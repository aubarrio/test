import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.optim as optim

class GCNLPA(nn.Module):
    def __init__(self, nfeat, nhid, nclass, adj, dropout_rate):
        super(GCNLPA, self).__init__()

        self.gc1 = GCNLPAConv(nfeat, nhid, adj)
        self.gc2 = GCNLPAConv(nhid, nclass, adj)
        self.dropout = dropout_rate

    def forward(self, x, adj, y):
        x, y_hat = self.gc1(x, adj, y)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x, y_hat = self.gc2(x, adj, y_hat)
        return F.log_softmax(x, dim=1), F.log_softmax(y_hat,dim=1)

class GCNLPAConv(nn.Module):
    """
    A GCN-LPA layer. Please refer to: https://arxiv.org/abs/2002.06755
    """

    def __init__(self, in_features, out_features, adj, bias=True):
        super(GCNLPAConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.adjacency_mask = Parameter(adj.clone()).to_dense()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj, y):
        adj = adj.to_dense()
        # W * x
        support = torch.mm(x, self.weight)
        # Hadamard Product: A' = Hadamard(A, M)
        adj = adj * self.adjacency_mask
        # Row-Normalize: D^-1 * (A')
        adj = F.normalize(adj, p=1, dim=1)

        # output = D^-1 * A' * X * W
        output = torch.mm(adj, support)
        # y' = D^-1 * A' * y
        y_hat = torch.mm(adj, y)

        if self.bias is not None:
            return output + self.bias, y_hat
        else:
            return output, y_hat

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
