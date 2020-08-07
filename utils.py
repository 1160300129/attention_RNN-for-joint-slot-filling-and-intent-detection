# coding: UTF-8
import os
import torch
import torch.nn as nn
import numpy as np
import pickle as pkl
import time
from sklearn.utils import shuffle
from config import Config
from datetime import timedelta

UNK, PAD = '<UNK>', '<PAD>'
cls_duiying = dict()
slot_duiying = dict()
config = Config()


def get_cls(filename):
    with open(filename, 'r', encoding='utf8') as f:
        i = 0
        b = f.read()
        lines = b.split('\n')
        for line in lines:
            if line != '':
                cls_duiying[line] = i
                i += 1


full_train_path = os.path.join('./data', config.dataset, config.train_data_path)
full_test_path = os.path.join('./data', config.dataset, config.test_data_path)
full_valid_path = os.path.join('./data', config.dataset, config.valid_data_path)


def read_cls(filename):
    with open(filename, 'r', encoding='utf8') as f:
        b = f.read()
        cls = b.strip().split('\n')
    return cls


def build_vocab(input_path, output_path, pad=True, unk=True):
    if not isinstance(input_path, str):
        raise TypeError('input_path should be string')

    if not isinstance(output_path, str):
        raise TypeError('output_path should be string')

    vocab = {}
    with open(input_path, 'r', encoding='utf8') as fd, \
            open(output_path, 'w+', encoding='utf8') as out:
        for line in fd:
            line = line.rstrip('\r\n')
            words = line.split()

            for w in words:
                if w == '<UNK>':
                    break
                if str.isdigit(w):
                    w = '0'
                if w in vocab:
                    vocab[w] += 1
                else:
                    vocab[w] = 1
        init_vocab = []
        if pad:
            init_vocab.append('<PAD>')
        if unk:
            init_vocab.append('<UNK>')
            init_vocab.append('<SOS>')
        vocab1 = init_vocab + sorted(vocab, key=vocab.get, reverse=True)
        for v in vocab1:
            out.write(v + '\n')


def load_vocabulary(path):
    vocab = []
    rev = []
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            line = line.rstrip('\r\n')
            rev.append(line)
        vocab = dict([(x, y) for (y, x) in enumerate(rev)])
    return {'vocab': vocab, 'rev': rev}


def shuffle_data(in_path, slot_path, intent_path):
    return shuffle(
        open(in_path, 'r', encoding='utf8').readlines(),
        open(slot_path, 'r', encoding='utf8').readlines(),
        open(intent_path, 'r', encoding='utf8').readlines()
    )


def read_seg_sentences(filename):
    with open(filename, 'r', encoding='utf8') as f:
        seg_sentences = []
        b = f.read()
        sentences = b.strip().split('\n')
        for each_sentence in sentences:
            words = each_sentence.strip().split()
            seg_sentences.append(words)
    return seg_sentences


def build_dataset(vocab, slot_vocab, intent_vocab):
    print(f"Vocab size: {len(vocab)}")

    def wl_load_dataset(path, path_slot, pad_size=16, train_data=None, train_slot=None, train_label=None):
        # if path.__contains__('train'):
        #     seg_sen = []
        #     for each_sentence in train_data:
        #         words = each_sentence.strip('\n').split()
        #         seg_sen.append(words)

        contents = []
        seg_sen = read_seg_sentences(path)
        seg_slot = read_seg_sentences(path_slot)

        if 'valid' in path:
            label = read_cls(os.path.join(full_valid_path, config.intent_file))
        elif 'test' in path:
            label = read_cls(os.path.join(full_test_path, config.intent_file))
        else:
            label = read_cls(os.path.join(full_train_path, config.intent_file))
        i = 0
        for token, slot in zip(seg_sen, seg_slot):
            if len(token) > 0:
                words_line = []
                slot_line = []
                seq_len = len(token)
                if pad_size:
                    if len(token) < pad_size:
                        token.extend([PAD] * (pad_size - len(token)))
                        slot.extend([PAD] * (pad_size - len(slot)))
                    else:
                        token = token[:pad_size]
                        slot = slot[:pad_size]
                        seq_len = pad_size
                # word to id
                for word, s in zip(token, slot):
                    words_line.append(vocab.get(word, vocab.get(UNK)))
                    slot_line.append(slot_vocab.get(s, 1))
                contents.append((words_line, intent_vocab.get(label[i], 0), seq_len, slot_line))
                i += 1
        return contents  # [([...], 0), ([...], 1), ...]

    train = wl_load_dataset(os.path.join(full_train_path, config.input_file),
                            os.path.join(full_train_path, config.slot_file), config.pad_size)
    dev = wl_load_dataset(os.path.join(full_valid_path, config.input_file),
                          os.path.join(full_valid_path, config.slot_file), config.pad_size)
    test = wl_load_dataset(os.path.join(full_test_path, config.input_file),
                           os.path.join(full_test_path, config.slot_file), config.pad_size)
    return train, dev, test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        # for _ in datas:
        #     for i in range(len(_[3])):
        #         if _[3][i] is None:
        #             _[3][i] = 1

        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        s = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len), y, s

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index > self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
