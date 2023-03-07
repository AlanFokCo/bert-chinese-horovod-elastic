import torch
import numpy as np
import argparse
from utils import build_dataset, build_iterator, get_time_dif
import horovod.torch as hvd
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from pytorch_pretrained.optimization import BertAdam
import models.bert_one as x
import os
from torch.utils.tensorboard import SummaryWriter
import time
import kubeai.handler as kubeai

parser = argparse.ArgumentParser(description='Bert Chinese Text Classification')
parser.add_argument('--model', type=str, default="bert")
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--learning-rate', type=float, default=5e-5)
parser.add_argument('--log-dir', default='./logs')
parser.add_argument('--batches-per-commit', type=int, default=100)
parser.add_argument('--batches-per-host-check', type=int, default=10)
parser.add_argument('--checkpoint-format', default='./checkpoint-{epoch}.pth.tar')

args = parser.parse_args()


def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if len(w.size()) < 2:
                continue
            if 'weight' in name:
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


def train(state, train_iter, test_iter):
    model = state.model
    model.train()

    epoch = state.epoch
    train_acc = Metric('train_accuracy')
    train_loss = Metric('train_loss')

    for idx, (trains, labels) in enumerate(train_iter):
        optimizer.zero_grad()
        data_batch = trains
        labels_batch = labels
        outputs = model(data_batch)
        model.zero_grad()
        loss = F.cross_entropy(outputs, labels_batch)
        loss.backward()
        optimizer.step()
        true = labels_batch.data.cpu()
        predict = torch.max(outputs.data, 1)[1].cpu()

        train_acc.update(torch.Tensor([np.float32(metrics.accuracy_score(true, predict))])[0])
        train_loss.update(loss)

        msg = 'Epoch: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%}'
        if hvd.rank() == 0:
            print(msg.format(epoch, train_loss.avg.item(), train_acc.avg.item()))
        if log_writer:
            log_writer.add_scalar('train/loss', train_loss.avg)
            log_writer.add_scalar('train/accuracy', train_acc.avg)
        state.commit()
        evaluate(model, test_iter, state.epoch)
    model.train()


def evaluate(model, data_iter, epoch):
    model.eval()
    loss_total = 0

    val_loss = Metric('val_loss')
    val_accuracy = Metric('val_accuracy')
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()

            predict_all = np.array([], dtype=int)
            labels_all = np.array([], dtype=int)
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

            val_loss.update(loss)
            val_accuracy.update(torch.Tensor([np.float32(metrics.accuracy_score(labels_all, predict_all))])[0])

    msg = 'Epoch: {0:>6},  Val Loss: {1:>5.2},  Val Acc: {2:>6.2%}'
    if hvd.rank() == 0:
        print(msg.format(epoch, val_loss.avg.item(), val_accuracy.avg.item()))

    if log_writer:
        log_writer.add_scalar('val/loss', val_loss.avg)
        log_writer.add_scalar('val/accuracy', val_accuracy.avg)


class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)

    def update(self, val):
        self.sum += hvd.allreduce(val.detach().cpu(), name=self.name)
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n


@hvd.elastic.run
def full_train(state):
    while state.epoch < args.epochs:
        train(state, train_iter, test_iter)
        save_checkpoint(state.epoch)
        end_epoch(state)


def end_epoch(state):
    state.epoch += 1
    state.commit()


def save_checkpoint(epoch):
    if hvd.rank() == 0:
        filepath = args.checkpoint_format.format(epoch=epoch + 1)
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, filepath)


if __name__ == '__main__':

    hvd.init()
    kubeai.init()

    dataset = '/examples/elastic/pytorch/THUCNews'

    model_name = args.model
    config = x.Config(dataset, args.batch_size, args.learning_rate)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True

    log_writer = SummaryWriter(args.log_dir) if hvd.rank() == 0 else None

    start_time = time.time()
    print("Loading data...")
    train_data, dev_data, test_data = build_dataset(config)

    train_dataset_size = int(len(train_data) / hvd.size())
    test_dataset_size = int(len(test_data) / hvd.size())

    train_iter = build_iterator(train_data[hvd.rank() * train_dataset_size:(hvd.rank() + 1) * train_dataset_size],
                                config)
    test_iter = build_iterator(test_data[hvd.rank() * test_dataset_size:(hvd.rank() + 1) * test_dataset_size], config)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    model = x.Model(config).cuda()

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=config.learning_rate,
                         warmup=0.05,
                         t_total=len(train_iter) * config.num_epochs)

    compression = hvd.Compression.fp16
    optimizer = hvd.DistributedOptimizer(
        optimizer, named_parameters=model.named_parameters(),
        compression=compression,
        backward_passes_per_step=1,
        op=hvd.Average,
        gradient_predivide_factor=1.0)

    resume_from_epoch = 0
    if hvd.rank() == 0:
        for try_epoch in range(args.epochs, 0, -1):
            if os.path.exists(args.checkpoint_format.format(epoch=try_epoch)):
                resume_from_epoch = try_epoch
                break

        if resume_from_epoch > 0:
            filepath = args.checkpoint_format.format(epoch=resume_from_epoch)
            checkpoint = torch.load(filepath)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])

    state = hvd.elastic.TorchState(model=model,
                                   optimizer=optimizer,
                                   batch=0,
                                   epoch=0)

    full_train(state)
