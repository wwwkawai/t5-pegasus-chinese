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
from model import StyleExtractor
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
                anchor, pos, neg = cur[0], cur[1], cur[2]
                D.append((anchor, pos, neg))
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
    for anchor, pos, neg in tqdm(data):
        anchor_ids = tokenizer.encode(anchor, max_length=max_len, truncation='only_first')
        pos_ids = tokenizer.encode(pos, max_length=max_len, truncation='only_first')
        neg_ids = tokenizer.encode(neg, max_length=max_len, truncation='only_first')
        anchor_pos = anchor_ids + pos_ids
        anchor_neg = anchor_ids + neg_ids

        features = {'input_ids': anchor_pos,
                    'attention_mask': [1] * len(anchor_pos),
                    'input_ids_ref': anchor_neg,
                    'attention_mask_ref': [1] * len(anchor_neg),
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


def eval_model(model, eval_data, tokenizer, device, args):
    best = 0


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
    total = 0
    correct = 0
    pos_avg = 0
    neg_avg =0
    cnt = 0
    for feature in tqdm(eval_data):
        content = {k: v.to(device) for k, v in feature.items()}
        rwfn = nn.Sigmoid()
        logits = rwfn(model(**content).logits)
        total += logits.size(0)
        correct += sum([logits[i][0] > logits[i][1] for i in range(logits.size(0))]).item()
        pos_avg += torch.mean(logits,dim=0)[0].item()
        neg_avg += torch.mean(logits,dim=0)[1].item()
        cnt += 1
    pos_avg /= cnt
    neg_avg /= cnt

    eval_metric = correct/total
    print("Validation Acc: {}, Pos Avg: {}, Neg Avg: {}".format(eval_metric, pos_avg, neg_avg))


def init_argument():
    parser = argparse.ArgumentParser(description='t5-pegasus-chinese')
    parser.add_argument('--test_data', default='./data/test.tsv')
    parser.add_argument('--model', default='./t5_pegasus_pretrain')
    parser.add_argument('--model_dir', default='./saved_model')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('--max_len', default=32, type=int, help='max length of inputs')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    # step 1. init argument
    args = init_argument()

    # step 2. prepare test_data
    tokenizer = T5PegasusTokenizer.from_pretrained('./t5_pegasus_pretrain')
    eval_data = prepare_data(args, args.test_data, tokenizer, term='train')
    # step 3. load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = MT5Config.from_pretrained(args.model)
    model = StyleExtractor.from_pretrained(args.model, config=config).to(device)

    eval_model(model, eval_data, tokenizer, device, args)