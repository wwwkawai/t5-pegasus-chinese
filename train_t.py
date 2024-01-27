import os
import re
import rouge
import jieba
import time
import torch
import argparse
import numpy as np
from tqdm.auto import tqdm
from bert4torch.models import *
from torch.utils.data import DataLoader, Dataset
from torch._six import container_abcs, string_classes, int_classes
from transformers import MT5ForConditionalGeneration, BertTokenizer, MT5Config
from model import StylePoem, StylePoemConfig, StyleExtractor
import json
from info_nce import InfoNCE, info_nce


def load_data(filename):
    """加载数据
    单条格式：(标题, 正文)
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f.readlines():
            cur = l.strip().split('\t')
            if len(cur) == 2:
                title, content = cur[0], cur[1]
                D.append((content, title, title))
                # D.append((title, title, content))
            elif len(cur) == 1:
                content = cur[0]
                D.append(content)
            elif len(cur) == 3:
                style, title, content = cur[0], cur[1], cur[2]
                D.append((style, title, content))
    return D


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


class KeyDataset(Dataset):
    def __init__(self, dict_data):
        self.data = dict_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def create_data(data, tokenizer, max_len=512, term='train'):
    """调用tokenizer.encode编码正文/标题，每条样本用dict表示数据域
    """
    ret, flag = [], True
    for style, title, content in tqdm(data):
        text_ids = tokenizer.encode(content, max_length=max_len, truncation='only_first')
        style_ids = tokenizer.encode(style, max_length=max_len, truncation='only_first')
        if flag and term == 'train':
            flag = False
            print(content)
        if term == 'train':
            summary_ids = tokenizer.encode(title, max_length=max_len, truncation='only_first')
            features = {'input_ids': text_ids,
                        'decoder_input_ids': summary_ids,
                        'attention_mask': [1] * len(text_ids),
                        'input_ids_ref': style_ids,
                        'attention_mask_ref': [1] * len(style_ids),
                        'decoder_attention_mask': [1] * len(summary_ids),
                        }

        elif term == 'dev':
            features = {'input_ids': text_ids,
                        'attention_mask': [1] * len(text_ids),
                        'input_ids_ref': style_ids,
                        'attention_mask_ref': [1] * len(style_ids),
                        'title': title
                        }

        ret.append(features)
    return ret


def sequence_padding(inputs, length=None, padding=0):
    """Numpy函数，将序列padding到同一长度
    """
    if length is None:
        length = max([len(x) for x in inputs])

    pad_width = [(0, 0) for _ in np.shape(inputs[0])]
    outputs = []
    for x in inputs:
        x = x[:length]
        pad_width[0] = (0, length - len(x))
        x = np.pad(x, pad_width, 'constant', constant_values=padding)
        outputs.append(x)

    return np.array(outputs, dtype='int64')


def default_collate(batch):
    """组batch
    各个数据域分别转换为tensor，tensor第一个维度等于batch_size
    """
    np_str_obj_array_pattern = re.compile(r'[SaUO]')
    default_collate_err_msg_format = (
        "default_collate: batch must contain tensors, numpy arrays, numbers, "
        "dicts or lists; found {}")
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))
            return default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch, dtype=torch.long)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            batch = sequence_padding(batch)
        return default_collate([default_collate(elem) for elem in batch])
    raise TypeError(default_collate_err_msg_format.format(elem_type))


def prepare_data(args, data_path, tokenizer, term='train'):
    """准备batch数据
    """
    data = load_data(data_path)
    data = create_data(data, tokenizer, args.max_len, term)
    data = KeyDataset(data)
    data = DataLoader(data, batch_size=args.batch_size, collate_fn=default_collate)
    return data


def compute_rouge(source, target):
    """计算rouge-1、rouge-2、rouge-l
    """
    source, target = ' '.join(source), ' '.join(target)
    try:
        scores = rouge.Rouge().get_scores(hyps=source, refs=target)
        return {
            'rouge-1': scores[0]['rouge-1']['f'],
            'rouge-2': scores[0]['rouge-2']['f'],
            'rouge-l': scores[0]['rouge-l']['f'],
        }
    except ValueError:
        return {
            'rouge-1': 0.0,
            'rouge-2': 0.0,
            'rouge-l': 0.0,
        }


def compute_rouges(sources, targets):
    scores = {
        'rouge-1': 0.0,
        'rouge-2': 0.0,
        'rouge-l': 0.0,
    }
    for source, target in zip(sources, targets):
        score = compute_rouge(source, target)
        for k, v in scores.items():
            scores[k] = v + score[k]

    return {k: v / len(targets) for k, v in scores.items()}


def train_model(model, style_model, adam, train_data, dev_data, tokenizer, device, args):
    args.save_dir = args.model_dir #+ '/' + str(time.strftime('%b%d%H%M%S', time.localtime()))
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    best = 0
    for epoch in range(args.num_epoch):
        model.train()
        for i, cur in enumerate(tqdm(train_data, desc='Epoch {}:'.format(epoch))):
            cur = {k: v.to(device) for k, v in cur.items()}
            outputs = model(**cur)
            prob, style_ref, style_repr, content_repr, content_ref, style_content_repr = outputs.logits, outputs.style_ref, outputs.style_repr, outputs.content_repr, outputs.content_ref, outputs.style_content_repr
            mask = cur['decoder_attention_mask'][:, 1:].reshape(-1).bool()
            # mask = mask[:,1:].reshape(-1).bool()
            prob = prob[:, :-1]
            prob = prob.reshape((-1, prob.size(-1)))[mask]
            labels = cur['decoder_input_ids'][:, 1:].reshape(-1)[mask]
            # labels = labels[:,1:].reshape(-1)[mask]
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            # s_loss_fct = nn.CosineEmbeddingLoss() #InfoNCE()
            # loss = loss_fct(prob, labels) + args.style_factor * s_loss_fct(style_repr, style_ref, torch.ones(style_ref.shape[0]).cuda())

            gen_loss = loss_fct(prob, labels)

            # c_loss = args.content_factor * c_loss_fct(content_repr, content_ref, style_content_repr.unsqueeze(1))
            # s_loss = args.style_factor * s_loss_fct(style_repr, style_ref, torch.ones(style_ref.shape[0]).cuda())
            # if i%5 == 0:
            #     loss = s_loss + c_loss
            # else:
            #     loss = s_loss + mlm_loss
            if i%args.freq == 0:
                cur['shuffle'] = True
                idx = torch.randperm(cur['input_ids'].shape[0])
                cur['input_ids_ref'] = cur['input_ids_ref'][idx, :]
                cur['attention_mask_ref'] = cur['attention_mask_ref'][idx, :]
                _cur = {k:v for k,v in cur.items() if k not in ['decoder_input_ids', 'decoder_attention_mask'] }
                with torch.no_grad():
                    token_ids = model.generate(
                        max_length=args.max_len,
                        eos_token_id=tokenizer.sep_token_id,
                        decoder_start_token_id=tokenizer.cls_token_id,
                        **_cur
                    )
                token_attention_mask = (token_ids!=0).long()
                cur['decoder_input_ids'] = token_ids
                cur['decoder_attention_mask'] = token_attention_mask
                new_outputs = model(**cur)
                hidden_states, content_repr, content_ref, style_content_repr = new_outputs.decoder_hidden_states, new_outputs.content_repr, new_outputs.content_ref, new_outputs.style_content_repr
                anchorpos_mask = torch.cat((cur['attention_mask_ref'],token_attention_mask),dim=-1)
                # logits = style_model.get_score(input_ids=cur['input_ids_ref'], generate_ids=token_ids, inputs_embeds=hidden_states, attention_mask=anchorpos_mask)
                logits = style_model.get_score(input_ids=cur['input_ids_ref'],
                                               inputs_embeds=hidden_states, attention_mask=anchorpos_mask)
                score_fn = nn.Sigmoid()
                scores = score_fn(logits)
                s_loss = args.style_factor * torch.mean(1-scores)
                c_loss_fct = InfoNCE(negative_mode='paired')
                c_loss = args.content_factor * c_loss_fct(content_repr, content_ref, style_content_repr.unsqueeze(1))
                loss = s_loss + gen_loss + c_loss
            else:
                loss = gen_loss
            if i % 100 == 0:
                print("Iter {}:  Training Loss: {},  MLM Loss: {},  Style Loss: {},  Content Loss: {}".format(i, loss.item(), gen_loss.item(), s_loss.item(), c_loss.item()))
            loss.backward()
            adam.step()
            adam.zero_grad()

            # second step
            # idx = torch.randperm(style_ref.shape[0])
            # style_ref_random = style_ref.detach()
            # style_ref_random = style_ref_random[idx,:]
            # cur['style_ref'] = style_ref_random
            # outputs = model(**cur)
            # style_repr, content_repr, content_ref, style_content_repr = outputs.style_repr, outputs.content_repr, outputs.content_ref, outputs.style_content_repr
            # s_loss = args.style_factor * s_loss_fct(style_repr, style_ref_random)
            # c_loss = args.content_factor * c_loss_fct(content_repr, content_ref, style_content_repr.unsqueeze(1))
            # loss = s_loss + c_loss
            # if i % 100 == 0:
            #     print("Iter {}:  Training Loss: {},  Content Loss: {},  Style Loss: {}".format(i, loss.item(),
            #                                                                                                   c_loss.item(),
            #                                                                                                   s_loss.item()))
            # loss.backward()
            # adam.step()
            # adam.zero_grad()


        # 验证
        model.eval()
        gens = []
        summaries = []
        for feature in tqdm(dev_data):
            title = feature['title']
            content = {k: v.to(device) for k, v in feature.items() if k != 'title'}
            if args.data_parallel and torch.cuda.is_available():
                gen = model.module.generate(max_length=args.max_len_generate,
                                            eos_token_id=tokenizer.sep_token_id,
                                            decoder_start_token_id=tokenizer.cls_token_id,
                                            **content)
            else:
                gen = model.generate(max_length=args.max_len_generate,
                                     eos_token_id=tokenizer.sep_token_id,
                                     decoder_start_token_id=tokenizer.cls_token_id,
                                     **content)
            gen = tokenizer.batch_decode(gen, skip_special_tokens=True)
            gen = [item.replace(' ', '') for item in gen]
            # print(title)
            # print(gen)
            gens.extend(gen)
            summaries.extend(title)
        scores = compute_rouges(gens, summaries)
        print("Validation Loss: {}".format(scores))
        rouge_l = scores['rouge-l']
        if rouge_l > best:
            best = rouge_l
            if args.data_parallel and torch.cuda.is_available():
                torch.save(model.module, os.path.join(args.save_dir, 'stylepoem_model'))
            else:
                model.save_pretrained(args.save_dir)
        output_dir = f"epoch_{epoch}"
        output_dir = os.path.join(args.save_dir,output_dir)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        with open(os.path.join(output_dir, "all_results.json"), "w") as f:
            json.dump(scores, f)
        model.save_pretrained(output_dir)


def init_argument():
    parser = argparse.ArgumentParser(description='t5-pegasus-chinese')
    parser.add_argument('--train_data', default='./data/train.tsv')
    parser.add_argument('--dev_data', default='./data/dev.tsv')
    parser.add_argument('--pretrain_model', default='./t5_pegasus_pretrain')
    parser.add_argument('--model_dir', default='./saved_model')

    parser.add_argument('--num_epoch', default=5, type=int, help='number of epoch')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('--lr', default=2e-4, help='learning rate')
    parser.add_argument('--data_parallel', default=False)
    parser.add_argument('--max_len', default=32, type=int, help='max length of inputs')
    parser.add_argument('--max_len_generate', default=32, type=int, help='max length of outputs')
    parser.add_argument('--style_model', default='./style_model')
    parser.add_argument('--mlm_probability', default=0.5, type=float)
    parser.add_argument('--with_adain', default=False, action='store_true')
    parser.add_argument('--style_factor', default=1.0, type=float, help='style loss factor')
    parser.add_argument('--content_factor', default=1.0, type=float, help='content loss factor')
    parser.add_argument('--freq', default=5, type=int, help='shuffle frequency')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    # step 1. init argument
    args = init_argument()

    # step 2. prepare training data and validation data
    tokenizer = T5PegasusTokenizer.from_pretrained('./t5_pegasus_pretrain')
    train_data = prepare_data(args, args.train_data, tokenizer, term='train')
    dev_data = prepare_data(args, args.dev_data, tokenizer, term='dev')

    # step 3. load pretrain model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    style_model_config = MT5Config.from_pretrained(args.style_model)
    style_model = StyleExtractor.from_pretrained(args.style_model, config=style_model_config).to(device)
    style_model.freeze_param()
    config = StylePoemConfig.from_pretrained(args.pretrain_model, mlm_probability=args.mlm_probability, with_adain=args.with_adain)
    model = StylePoem \
        .from_pretrained(args.pretrain_model, config=config).to(device)
    if args.data_parallel and torch.cuda.is_available():
        device_ids = range(torch.cuda.device_count())
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # step 4. finetune
    adam = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_model(model, style_model, adam, train_data, dev_data, tokenizer, device, args)