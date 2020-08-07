import torch
import numpy as np


class Config(object):
    def __init__(self):
        self.hidden_size = 256
        self.embed_dim = 1024
        self.intent_dim = 128
        self.model_type = 'full'
        self.num_rnn = 1
        self.optimizer = 'rmsprop'
        self.batch_size = 4
        self.learning_rate = 0.001
        self.max_epochs = 5
        self.run_name = 'Attentinon_RNN'
        self.dataset = 'snips'
        self.model_path = './model'
        self.vocab_path = './vocab'
        self.train_data_path = 'train'
        self.test_data_path = 'test'
        self.valid_data_path = 'valid'
        self.input_file = 'seq.in'
        self.slot_file = 'seq.out'
        self.intent_file = 'label'
        self.pad_size = 16
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_vocab = 0
        self.slot_size = len([x.strip() for x in open('vocab/slot_vocab').readlines()])
        self.intent_size = len([x.strip() for x in open('vocab/intent_vocab').readlines()])
        self.decoder_embed_size = self.slot_size // 3