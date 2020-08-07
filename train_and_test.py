import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif


def init_network(model, method='xavier', exclude='embedding'):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name and 'batch' not in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(config, encoder, decoder, train_iter, dev_iter, test_iter):
    start_time = time.time()
    encoder.train()
    decoder.train()
    optimizer_encoder = torch.optim.Adam(encoder.parameters(), lr=config.learning_rate)
    optimizer_decoder = torch.optim.Adam(decoder.parameters(), lr=config.learning_rate)
    scheduler_encoder = torch.optim.lr_scheduler.ExponentialLR(optimizer_encoder, gamma=0.9)
    scheduler_decoder = torch.optim.lr_scheduler.ExponentialLR(optimizer_decoder, gamma=0.9)
    total_batch = 0
    dev_best_loss = float('inf')
    last_improve = 0
    flag = False
    for epoch in range(config.max_epochs):
        losses = []
        print('Epoch [{}/{}]'.format(epoch + 1, config.max_epochs))
        for i, (trains, labels, slot) in enumerate(train_iter):
            bsz = len(trains[0])
            if trains[0].shape[0] == 0 or labels.shape[0] == 0:
                continue
            x_mask = torch.cat([(torch.tensor(tuple(map(lambda s: s == 0, t.data))).byte()).to(config.device) for t in trains[0]]).view(bsz, -1)

            outputs, hidden_c = encoder(trains[0], x_mask)

            start_decoder = torch.tensor([[2]*bsz]).long().to(config.device).transpose(1, 0)

            slot_score, intent_score = decoder(start_decoder, hidden_c, outputs, x_mask)

            encoder.zero_grad()
            decoder.zero_grad()

            loss_slot = F.cross_entropy(slot_score, slot.view(-1))
            loss_intent = F.cross_entropy(intent_score, labels)

            loss = loss_slot+loss_intent
            losses.append(loss.item())
            loss.backward()

            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5.0)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5.0)

            optimizer_encoder.step()
            optimizer_decoder.step()
            scheduler_encoder.step(epoch=epoch)
            scheduler_decoder.step(epoch=epoch)

            if total_batch % 100 == 0:
                true_intent = labels.data.cpu()
                true_slot = slot.data.cpu().view(-1)
                predict_intent = torch.max(intent_score.data, 1)[1].cpu()
                predict_slot = torch.max(slot_score.data, 1)[1].cpu()
                train_acc_intent = metrics.accuracy_score(true_intent, predict_intent)
                train_acc_slot = metrics.accuracy_score(true_slot, predict_slot)
                dev_intent_acc, dev_loss, dev_slot_acc = evaluate(config, encoder, decoder, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(encoder.state_dict(), 'model/' + config.run_name + '_encoder.ckpt')
                    torch.save(decoder.state_dict(), 'model/' + config.run_name + '_decoder.ckpt')
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg_intent = 'Iter: {0:>6},  Train Intent Loss: {1:>5.2},  Train Intent Acc: {2:>6.2%},  ' \
                             'Val Intent Loss: {3:>5.2},  Val Intent Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg_intent.format(total_batch, loss_intent.item(), train_acc_intent, dev_loss,
                                        dev_intent_acc, time_dif, improve))
                msg_slot = 'Iter: {0:>6},  Train Slot Loss: {1:>5.2},  Train Slot Acc: {2:>6.2%},  ' \
                           'Val Slot Loss: {3:>5.2},  Val Slot Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg_slot.format(total_batch, loss_slot.item(), train_acc_slot, dev_loss, dev_slot_acc,
                                      time_dif, improve))
                encoder.train()
                decoder.train()
            total_batch += 1
            if total_batch - last_improve > 600:
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    test(config, encoder, decoder, test_iter)


def test(config, encoder, decoder, test_iter):
    encoder.load_state_dict(torch.load('model/' + config.run_name + '_encoder.ckpt'))
    decoder.load_state_dict(torch.load('model/' + config.run_name + '_decoder.ckpt'))
    encoder.eval()
    decoder.eval()
    start_time = time.time()
    test_acc_intent, test_loss, test_intent_confusion, test_acc_slot, test_slot_confusion = evaluate(
        config, encoder, decoder, test_iter, test=True)
    msg_intent = 'Test Loss: {0:>5.2},  Test Intent Acc: {1:>6.2%}'
    msg_slot = 'Test Loss: {0:>5.2},  Test Slot Acc: {1:>6.2%}'
    print(msg_intent.format(test_loss, test_acc_intent))
    print(msg_slot.format(test_loss, test_acc_slot))
    # print("Precision, Recall and F1-score...")
    # print(test_intent_report)
    print("Confusion Matrix...")
    print(test_intent_confusion)
    print(test_slot_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage: ", time_dif)


def evaluate(config, encoder, decoder, data_iter, test=False):
    encoder.eval()
    decoder.eval()
    loss_total = 0
    predict_slot_all = np.array([], dtype=int)
    predict_intent_all = np.array([], dtype=int)
    labels_slot_all = np.array([], dtype=int)
    labels_intent_all = np.array([], dtype=int)
    with torch.no_grad():
        i = 0
        for texts, labels, slot in data_iter:
            bsz = len(texts[0])
            # print(i)
            if texts[0].shape[0] == 0 or labels.shape[0] == 0:
                continue
            x_mask = torch.cat([(torch.tensor(tuple(map(lambda s: s == 0, t.data))).byte()).to(config.device) for t in
                                texts[0]]).view(bsz, -1)

            outputs, hidden_c = encoder(texts[0], x_mask)

            start_decoder = torch.tensor([[2] * bsz]).long().to(config.device).transpose(1, 0)

            slot_score, intent_score = decoder(start_decoder, hidden_c, outputs, x_mask)

            slot = slot.view(-1)
            loss_intent = F.cross_entropy(intent_score, labels)
            loss_slot = F.cross_entropy(slot_score, slot)
            loss_total += loss_slot+loss_intent
            labels = labels.data.cpu().numpy()
            slot = slot.data.cpu().numpy()
            predict_intent = torch.max(intent_score.data, 1)[1].cpu()
            predict_slot = torch.max(slot_score.data, 1)[1].cpu()
            labels_intent_all = np.append(labels_intent_all, labels)
            labels_slot_all = np.append(labels_slot_all, slot)
            predict_intent_all = np.append(predict_intent_all, predict_intent)
            predict_slot_all = np.append(predict_slot_all, predict_slot)
            i += 1
    acc_intent = metrics.accuracy_score(labels_intent_all, predict_intent_all)
    new_labels_slot_all = []
    new_predict_slot_all = []
    for a, b in zip(labels_slot_all, predict_slot_all):
        if a == b and a == 0:
            continue
        else:
            new_labels_slot_all.append(a)
            new_predict_slot_all.append(b)
    new_labels_slot_all = np.array(new_labels_slot_all)
    new_predict_slot_all = np.array(new_predict_slot_all)
    acc_slot = metrics.accuracy_score(new_labels_slot_all, new_predict_slot_all)
    if test:
        import os
        from utils import load_vocabulary, build_vocab
        full_test_path = os.path.join('./data', config.dataset, config.test_data_path)
        # build_vocab(os.path.join(full_test_path, config.intent_file),
        #             os.path.join(config.vocab_path, 'test_intent_vocab'))
        # build_vocab(os.path.join(full_test_path, config.slot_file),
        #             os.path.join(config.vocab_path, 'test_slot_vocab'))
        # slot_vocab = load_vocabulary(os.path.join(config.vocab_path, 'test_slot_vocab'))
        # slot_vocab['rev'] = slot_vocab['rev'][0:72]
        # intent_vocab = load_vocabulary(os.path.join(config.vocab_path, 'test_intent_vocab'))
        # report_intent = metrics.classification_report(labels_intent_all, predict_intent_all,
        #                                               target_names=intent_vocab['rev'], digits=4)
        # report_slot = metrics.classification_report(labels_slot_all, predict_slot_all,
        #                                             target_names=slot_vocab['rev'], digits=4)
        # print(report_slot)
        # print(report_intent)
        confusion_intent = metrics.confusion_matrix(labels_intent_all, predict_intent_all)
        confusion_slot = metrics.confusion_matrix(labels_slot_all, predict_slot_all)
        return acc_intent, loss_total / len(data_iter), confusion_intent, acc_slot, confusion_slot
    return acc_intent, loss_total / len(data_iter), acc_slot
