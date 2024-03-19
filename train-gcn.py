from __future__ import division, print_function

import argparse
import random
import time
import uuid

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import *

from gcn import GCN

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1500,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')
parser.add_argument('--wd1', type=float, default=0.01,
                    help='weight decay (L2 loss on parameters).')
parser.add_argument('--wd2', type=float, default=5e-4,
                    help='weight decay (L2 loss on parameters).')
parser.add_argument('--layer', type=int, default=64, help='Number of layers.')
parser.add_argument('--hidden', type=int, default=64,
                    help='hidden dimensions.')
parser.add_argument('--dropout', type=float, default=0.6,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--patience', type=int, default=200, help='Patience')
parser.add_argument('--data', default='cora', help='dateset')
parser.add_argument('--dev', type=int, default=0, help='device id')
parser.add_argument('--test', action='store_true',
                    default=False, help='evaluation on test set.')
parser.add_argument('--batchnorm', action='store_true')
args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_citation(args.data)
cudaid = "cuda:"+str(args.dev)
# device = torch.device(cudaid)
device = torch.device("cpu")  # cuda:0
features = features.to(device)
adj = adj.to(device)
checkpt_file = 'pretrained/'+uuid.uuid4().hex+'.pt'
print(cudaid, checkpt_file)

model = GCN(nfeat=features.shape[1],
            nlayers=args.layer,
            nhidden=args.hidden,
            nclass=int(labels.max()) + 1,
            dropout=args.dropout,
            batch_norm=args.batchnorm)


optimizer = optim.Adam([
    {'params': model.params1, 'weight_decay': args.wd1},
    {'params': model.params2, 'weight_decay': args.wd2},
], lr=args.lr)


def train():
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    acc_train = accuracy(output[idx_train], labels[idx_train].to(device))
    loss_train = F.nll_loss(output[idx_train], labels[idx_train].to(device))
    loss_train.backward()
    optimizer.step()
    return loss_train.item(), acc_train.item()


def validate():
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        loss_val = F.nll_loss(output[idx_val], labels[idx_val].to(device))
        acc_val = accuracy(output[idx_val], labels[idx_val].to(device))
        return loss_val.item(), acc_val.item()


def test():
    model.load_state_dict(torch.load(checkpt_file))
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test].to(device))
        acc_test = accuracy(output[idx_test], labels[idx_test].to(device))
        return loss_test.item(), acc_test.item()


t_total = time.time()
bad_counter = 0
best_val = 999999999
best_epoch_val = 0
best_train = 999999999
best_train_acc = 0
best_epoch_train = 0
acc = 0
for epoch in range(args.epochs):
    loss_tra, acc_tra = train()
    loss_val, acc_val = validate()
    if epoch % 20 == 0:
        print('Epoch:{:04d}'.format(epoch+1),
              'train',
              'loss:{:.3f}'.format(loss_tra),
              'acc:{:.2f}'.format(acc_tra*100),
              '| val',
              'loss:{:.3f}'.format(loss_val),
              'acc:{:.2f}'.format(acc_val*100))
    if loss_tra < best_train:
        best_train = loss_tra
        best_epoch_train = epoch
        best_train_acc = acc_tra
    if loss_val < best_val:
        best_val = loss_val
        best_epoch_val = epoch
        acc = acc_val
        torch.save(model.state_dict(), checkpt_file)
        bad_counter = 0
    else:
        bad_counter += 1

    if bad_counter == args.patience:
        print("bad counter is", bad_counter, "patience is reached")
        break

acc_test = test()[1]

print("Train cost: {:.4f}s".format(time.time() - t_total))
print('Load {}th epoch which has the best training accuracy'.format(best_epoch_train))
print("Train", "acc.:{:.1f}".format(best_train_acc*100))
print('Load {}th epoch which has the best val accuracy'.format(best_epoch_val))
print("Val", "acc.:{:.1f}".format(acc*100))
print("Test", "acc.:{:.1f}".format(acc_test*100))
