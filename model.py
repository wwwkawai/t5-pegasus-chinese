import copy
from copy import deepcopy
from dataclasses import dataclass
import torch
import warnings
import jieba
import torch.distributed as dist
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from typing import Optional, List, Tuple

from transformers import MT5ForConditionalGeneration, MT5Config, BartTokenizer, AutoModel, BartPretrainedModel, PreTrainedModel, \
    BartModel, BertTokenizer
from transformers.models.bart.configuration_bart import BartConfig
from transformers.models.bart.modeling_bart import shift_tokens_right, BartEncoder
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput, SequenceClassifierOutput




@dataclass
class StylePoemOutput(Seq2SeqLMOutput):
    content_last_hidden_state: Optional[torch.FloatTensor] = None
    content_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    content_attentions: Optional[Tuple[torch.FloatTensor]] = None
    style_ref: Optional[torch.FloatTensor] = None
    style_repr: Optional[Tuple[torch.FloatTensor]] = None
    style_attentions: Optional[Tuple[torch.FloatTensor]] = None
    content_ref: Optional[torch.FloatTensor] = None
    content_repr: Optional[Tuple[torch.FloatTensor]] = None
    style_content_repr: Optional[Tuple[torch.FloatTensor]] = None
    label: Optional[torch.LongTensor] = None
    mask: Optional[torch.LongTensor] = None

@dataclass
class DualEncoderOutPut(BaseModelOutput):
    content_ref: Optional[torch.FloatTensor] = None
    style_content_repr: Optional[Tuple[torch.FloatTensor]] = None
    label: Optional[torch.LongTensor] = None
    mask: Optional[torch.LongTensor] = None

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.unsqueeze(-1)
        gamma, beta = torch.chunk(h, chunks=2, dim=2)
        gamma = gamma.squeeze()
        beta = beta.squeeze()
        return (1 + gamma) * self.norm(x.unsqueeze(1)).squeeze() + beta

class StylePoemConfig(MT5Config):
    def __init__(self,
                 mlm_probability=0.5,
                 with_adain=False,
                **kwargs):
        super().__init__(**kwargs)
        self.mlm_probability = mlm_probability
        self.with_adain = with_adain

class T5PegasusTokenizer(BertTokenizer):
    """结合中文特点完善的Tokenizer
    基于词颗粒度的分词，如词表中未出现，再调用BERT原生Tokenizer
    """

    def __init__(self, pre_tokenizer=lambda x: jieba.cut(x, HMM=False), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_tokenizer = pre_tokenizer

    def _tokenize(self, text, *arg, **kwargs):
        split_tokens = []
        for text in self.pre_tokenizer(text):
            if text in self.vocab:
                split_tokens.append(text)
            else:
                split_tokens.extend(super()._tokenize(text))
        return split_tokens

# add a encoder wrapper
class DualEncoderWrapper(nn.Module):
    def __init__(self, content_encoder, style_encoder, mlm, d_model=None) -> None:
        super().__init__()
        self.content_encoder = content_encoder
        self.style_encoder = style_encoder
        self.mlm = mlm
        self.with_adain = False
        if d_model:
            self.adain = AdaIN(d_model, d_model)
            self.with_adain = True
        # self.mlm = MaskLM()
        # self.mlm.to(self.style_encoder.device)

    def forward(
            self,
            input_ids,
            attention_mask,
            input_ids_ref,
            attention_mask_ref,
            shuffle: Optional[bool] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        # 1. encode style text
        style_encoder_outputs = self.style_encoder(
            input_ids=input_ids_ref,
            attention_mask=attention_mask_ref,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        style_last_hidden_state, style_hidden_states = style_encoder_outputs.last_hidden_state, style_encoder_outputs.hidden_states
        style_ref = mean_pooling(style_last_hidden_state, attention_mask_ref)
        # 2. encode content text
        labels = None
        batch_size = attention_mask.size(0)
        encoder_attention_mask = torch.cat((attention_mask,torch.ones(batch_size,1).long().cuda()), dim=1)
        if self.training and not shuffle:
            input_ids, attention_mask, labels = self.mlm.torch_mask_tokens(input_ids, attention_mask)

        content_encoder_outputs = self.content_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        content_last_hidden_state, content_hidden_states = content_encoder_outputs.last_hidden_state, content_encoder_outputs.hidden_states
        content_ref = mean_pooling(content_last_hidden_state, attention_mask)
        # concat style and content hidden states
        # encoder_hidden_state = torch.cat((content_last_hidden_state, style_last_hidden_state), dim=1)

        style_content_encoder_outputs = self.content_encoder(
            input_ids=input_ids_ref,
            attention_mask=attention_mask_ref,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        style_content_last_hidden_state = style_content_encoder_outputs.last_hidden_state
        style_content_repr = mean_pooling(style_content_last_hidden_state, attention_mask_ref)
        if self.with_adain:
            encoder_hidden_state = self.adain(content_last_hidden_state, style_last_hidden_state)
        else:
            # encoder_hidden_state = content_last_hidden_state
            encoder_hidden_state = torch.cat((content_last_hidden_state, style_ref.unsqueeze(1)), dim=1)
            # encoder_hidden_state = torch.cat((content_last_hidden_state, style_last_hidden_state), dim=1)
        # encoder_hidden_state = style_last_hidden_state


        # return encoder_hidden_state, encoder_attention_mask
        return DualEncoderOutPut(
            last_hidden_state=encoder_hidden_state,
            content_ref=content_ref,
            style_content_repr=style_content_repr,
            label=labels,
            mask=encoder_attention_mask
        )
    def get_style_repr(self, inputs_embeds, attention_mask):
        style_encoder_outputs = self.style_encoder(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=True
        )
        style_last_hidden_state = style_encoder_outputs.last_hidden_state
        style_repr = mean_pooling(style_last_hidden_state, attention_mask)
        return style_repr
    def get_content_repr(self, inputs_embeds, attention_mask):
        content_encoder_outputs = self.content_encoder(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=True
        )
        content_last_hidden_state = content_encoder_outputs.last_hidden_state
        content_repr = mean_pooling(content_last_hidden_state, attention_mask)
        return content_repr




class MaskLM(nn.Module):
    def __init__(self, mlm_probability=0.50):
        super().__init__()
        self.mlm_probability = mlm_probability
        self.tokenizer = T5PegasusTokenizer.from_pretrained('./t5_pegasus_pretrain')

    def torch_mask_tokens(self, inputs, masks, special_tokens_mask=None):
        labels = inputs.clone()
        if self.mlm_probability <= 0.0:
            return inputs, masks, labels
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        masks[masked_indices] = 0
        labels[~masked_indices] = self.tokenizer.pad_token_id  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        # inputs[masked_indices] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # # # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.1)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(low=103, high=self.tokenizer.vocab_size, size=labels.shape,
                                     dtype=torch.long).cuda()
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, masks, labels


class StylePoem(MT5ForConditionalGeneration):

    def __init__(self, config):
        super().__init__(config)
        self.post_initialize(mlm_probability=config.mlm_probability, with_adain = config.with_adain)

    def post_initialize(self, mlm_probability=0.5, with_adain = False):
        self.content_encoder = deepcopy(self.encoder)

        self.style_encoder = self.encoder
        # for param in self.style_encoder.parameters():
        #     param.requires_grad = False
        self.style_encoder.to(self.device)

        # self.extractor_type = extractor_type
        # if self.extractor_type == 'simcse':
        #     self.style_extractor = AutoModel.from_pretrained('princeton-nlp/unsup-simcse-roberta-base',
        #                                                      cache_dir='./pretrain-models')
        # elif self.extractor_type == 'style_emb':
        #     self.style_extractor = AutoModel.from_pretrained('AnnaWegmann/Style-Embedding',
        #                                                      cache_dir='./pretrain-models')
        # else:
        #     raise NotImplementedError(f'extractor type {self.extractor_type} not implemented.')

        # freeze extractor parameters
        # for param in self.style_extractor.parameters():
        #     param.requires_grad = False
        # self.style_extractor.to(self.device)
        self.mlm = MaskLM(mlm_probability=mlm_probability)
        self.mlm.to(self.device)
        self.with_adain = with_adain
        if with_adain:
            self.encoders = DualEncoderWrapper(self.content_encoder, self.style_encoder, self.mlm, self.model_dim)
        else:
            self.encoders = DualEncoderWrapper(self.content_encoder, self.style_encoder, self.mlm)
        # for param in self.decoder.parameters():
        #     param.requires_grad = False


    def get_encoder(self):
        return self.encoders

    def forward(
            self,
            input_ids,
            attention_mask,
            input_ids_ref,
            attention_mask_ref,
            shuffle: Optional[bool] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            decoder_head_mask: Optional[torch.Tensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[List[torch.FloatTensor]] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                # warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoders(
                input_ids=input_ids,
                attention_mask=attention_mask,
                input_ids_ref=input_ids_ref,
                attention_mask_ref=attention_mask_ref,
                shuffle = shuffle,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states, label, mask, content_ref, style_content_repr = encoder_outputs.last_hidden_state, encoder_outputs.label, encoder_outputs.mask, encoder_outputs.content_ref, encoder_outputs.style_content_repr
        style_ref = hidden_states[:,-1,:]
        style_ref = style_ref.squeeze()
        # encoder_attention_mask = torch.cat((attention_mask, attention_mask_ref), dim=1)
        # encoder_attention_mask = attention_mask_ref
        if self.with_adain:
            encoder_attention_mask = attention_mask
        else:
            encoder_attention_mask = mask
            # encoder_attention_mask = torch.cat((attention_mask, attention_mask_ref), dim=1)
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)
        style_repr = None
        content_repr = None
        if input_ids is not None:
            style_repr = self.encoders.get_style_repr(sequence_output, decoder_attention_mask)
            content_repr = self.encoders.get_content_repr(sequence_output, decoder_attention_mask)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return StylePoemOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs[0],
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            style_ref=style_ref,
            style_repr=style_repr,
            content_ref=content_ref,
            content_repr=content_repr,
            style_content_repr=style_content_repr,
            label=label,
            mask=mask
        )

    def prepare_inputs_for_generation(
            self,
            decoder_input_ids,
            past=None,
            attention_mask=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            use_cache=None,
            encoder_outputs=None,
            **kwargs,
    ):
        # cut decoder_input_ids if past_key_values is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        input_ids_ref = kwargs['input_ids_ref'] if 'input_ids_ref' in kwargs.keys() else None
        attention_mask_ref = kwargs['attention_mask_ref'] if 'attention_mask_ref' in kwargs.keys() else None

        return {
            "input_ids": None,
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "input_ids_ref": input_ids_ref,  # to be set
            "attention_mask_ref": attention_mask_ref,  # to be set
            "decoder_attention_mask": decoder_attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }


class StyleExtractor(MT5ForConditionalGeneration):

    def __init__(self, config):
        super().__init__(config)
        self.classification_head = nn.Linear(config.d_model, 1)
        self.loss_fn = nn.LogSigmoid()
    def freeze_param(self):
        for param in self.parameters():
            param.requires_grad = False
    def get_score(self,
                  input_ids: torch.LongTensor = None,
                  generate_ids: torch.LongTensor = None,
                  attention_mask: Optional[torch.Tensor] = None,
                  inputs_embeds: Optional[torch.FloatTensor] = None,
                  output_attentions: Optional[bool] = None,
                  output_hidden_states: Optional[bool] = None,
                  return_dict: Optional[bool] = None,
                  ):
        anchor_embeds = self.encoder.embed_tokens(input_ids)
        if generate_ids is not None:
            gen_embeds = self.encoder.embed_tokens(generate_ids)
            inputs_embeds = inputs_embeds + (gen_embeds - inputs_embeds).detach()
        inputs_embeds = torch.cat((anchor_embeds,inputs_embeds),dim=1)
        outputs = self.encoder(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sentence_embeddings = mean_pooling(outputs[0], attention_mask)
        rank = self.classification_head(sentence_embeddings)
        return rank
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            input_ids_ref: torch.LongTensor = None,
            attention_mask_ref: Optional[torch.Tensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.LongTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        # 1. encode anchor_pos text
        pos_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 2. encode anchor_neg text
        neg_outputs = self.encoder(
            input_ids=input_ids_ref,
            attention_mask=attention_mask_ref,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )



        # 3. mean_pooling and add
        pos_sentence_embeddings = mean_pooling(pos_outputs[0], attention_mask)
        neg_sentence_embeddings = mean_pooling(neg_outputs[0], attention_mask_ref)

        # 4. classification
        pos_rank = self.classification_head(pos_sentence_embeddings)
        neg_rank = self.classification_head(neg_sentence_embeddings)
        loss = -torch.mean(self.loss_fn(pos_rank-neg_rank))

        return SequenceClassifierOutput(
            loss=loss,
            logits=torch.cat((pos_rank, neg_rank),dim=1),
        )


