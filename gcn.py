import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.nn import GCNConv


class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(
            self.in_features, self.out_features
        ))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        hi = torch.spmm(adj, input)
        output = torch.mm(hi, self.weight)
        return output


class GCN(nn.Module):
    def __init__(self, nfeat, nlayers, nhidden, nclass, dropout, batch_norm=False):
        print("Use batch norm?", batch_norm)
        super(GCN, self).__init__()
        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(
                # GCNConv(nhidden, nhidden)
                GraphConvolution(nhidden, nhidden)
            )
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bns = nn.ModuleList()
            for _ in range(nlayers):
                self.bns.append(torch.nn.BatchNorm1d(nhidden, nhidden))

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, self.training)
        layer_inner = self.act_fn(self.fcs[0](x))

        for i, con in enumerate(self.convs):
            layer_inner = F.dropout(
                layer_inner, self.dropout, self.training)  # dropout
            layer_inner = self.act_fn(
                con(layer_inner, adj))  # graph_convolution
            # if self.batch_norm:
            #     layer_inner = self.bns[i](layer_inner)  # apply batch norm
            # layer_inner = self.act_fn(layer_inner)  # apply activation function

        layer_inner = F.dropout(layer_inner, self.dropout, self.training)
        layer_inner = self.fcs[-1](layer_inner)
        return F.log_softmax(layer_inner, dim=1)
