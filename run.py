import os
from config import Config
import time
from importlib import import_module
import torch
from train_and_test import init_network, train, test
from utils import build_dataset, build_iterator, get_time_dif, load_vocabulary, build_vocab


config = Config()

full_train_path = os.path.join('./data', config.dataset, config.train_data_path)
full_test_path = os.path.join('./data', config.dataset, config.test_data_path)
full_valid_path = os.path.join('./data', config.dataset, config.valid_data_path)

build_vocab(os.path.join(full_train_path, config.input_file), os.path.join(config.vocab_path, 'in_vocab'))
build_vocab(os.path.join(full_train_path, config.slot_file), os.path.join(config.vocab_path, 'slot_vocab'), unk=False)
build_vocab(os.path.join(full_train_path, config.intent_file), os.path.join(config.vocab_path, 'intent_vocab'), pad=False, unk=False)
if config.dataset == 'snips':
    print('use snips dataset')
elif config.dataset == 'atis':
    print('use atis dataset')

model_name = 'Attention_RNN'


torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True
start_time = time.time()
print('加载数据...')

in_vocab = load_vocabulary(os.path.join(config.vocab_path, 'in_vocab'))
slot_vocab = load_vocabulary(os.path.join(config.vocab_path, 'slot_vocab'))
intent_vocab = load_vocabulary(os.path.join(config.vocab_path, 'intent_vocab'))
train_data, dev_data, test_data = build_dataset(in_vocab['vocab'], slot_vocab['vocab'], intent_vocab['vocab'])

train_iter = build_iterator(train_data)
dev_iter = build_iterator(dev_data)
test_iter = build_iterator(test_data)
time_dif = get_time_dif(start_time)
print('time usage:', time_dif)

config.n_vocab = len(in_vocab['vocab'])

x = import_module(model_name)
encoder = x.Encoder(config).to(config.device)
decoder = x.Decoder(config).to(config.device)

init_network(encoder)
init_network(decoder)
print(encoder.parameters)
print(decoder.parameters)

# train(config, encoder, decoder, train_iter, dev_iter, test_iter)
test(config, encoder, decoder, test_iter)





