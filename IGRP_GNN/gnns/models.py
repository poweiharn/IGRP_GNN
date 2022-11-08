import torch.nn as nn
import torch.nn.functional as F
from gnns.layers import GraphConvolution, UGS_GraphConvolution
#from torch.nn.parameter import Parameter
import utils
import torch
import math
import random

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, adj, prune_adj, cuda):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid, adj)
        self.gc2 = GraphConvolution(nhid, nclass, adj)
        self.dropout = dropout
        self.prune_adj = prune_adj
        self.apply_adj_mask = False
        self.is_cuda = cuda
        self.normalize = utils.torch_normalize_adj_sparse
        if self.prune_adj:
          self.adj_weight = nn.parameter.Parameter(torch.FloatTensor(adj.to_sparse().values()), requires_grad=True)

          self.best_adj_weight = torch.ones(self.adj_weight.shape)
          self.best_adj_grad = torch.ones(self.adj_weight.shape)
          if self.is_cuda:  
            self.best_adj_weight = self.best_adj_weight.cuda()
            self.best_adj_grad = self.best_adj_grad.cuda()
          self.top_k_adj_list = []

    def forward(self, x, adj):
        if self.prune_adj:
          if(self.apply_adj_mask == False):
            adj_weight = torch.sparse_coo_tensor(adj.to_sparse().indices(), self.adj_weight, adj.size()).to_dense()
            adj_temp = torch.mul(adj, torch.mul(adj_weight, self.adj_mask))
          else:
            adj_temp = torch.mul(adj, self.adj_mask)
          
          adj_temp = adj_temp.to_sparse()
          adj_temp = self.normalize(adj_temp, self.is_cuda)
          adj_temp = adj_temp.to_dense()
          adj_temp[torch.isnan(adj_temp)] = random.uniform(0, 1)
          #adj_temp[torch.isnan(adj_temp)] = 0.0
        else:
          adj_temp = adj

        x = F.relu(self.gc1(x, adj_temp))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj_temp)
        return F.log_softmax(x, dim=1)

    def set_adj_mask(self, adj_mask):
      self.adj_mask = adj_mask


class UGS_GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, adj, cuda):
        super(UGS_GCN, self).__init__()

        self.gc1 = UGS_GraphConvolution(nfeat, nhid, adj)
        self.gc2 = UGS_GraphConvolution(nhid, nclass, adj)
        self.dropout = dropout
        self.is_cuda = cuda
        self.normalize = utils.torch_normalize_adj_sparse
        self.adj_nonzero = torch.nonzero(adj, as_tuple=False).shape[0]
        self.adj_mask1_train = nn.Parameter(self.generate_adj_mask(adj.to_sparse().values()))
        self.adj_mask2_fixed = nn.Parameter(self.generate_adj_mask(adj.to_sparse().values()), requires_grad=False)

    def forward(self, x, adj):
        mask_train = torch.sparse_coo_tensor(adj.to_sparse().indices(), self.adj_mask1_train, adj.size()).to_dense()
        mask_fixed = torch.sparse_coo_tensor(adj.to_sparse().indices(), self.adj_mask2_fixed, adj.size()).to_dense()
        adj = torch.mul(adj, mask_train)
        adj = torch.mul(adj, mask_fixed)
        adj = adj.to_sparse()
        adj = self.normalize(adj, self.is_cuda)
        adj = adj.to_dense()
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1) 

    def generate_adj_mask(self, input_adj): 
        sparse_adj = input_adj
        zeros = torch.zeros_like(sparse_adj)
        ones = torch.ones_like(sparse_adj)
        mask = torch.where(sparse_adj != 0, ones, zeros)
        return mask 

class Random_GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, adj, cuda):
        super(Random_GCN, self).__init__()

        self.gc1 = UGS_GraphConvolution(nfeat, nhid, adj)
        self.gc2 = UGS_GraphConvolution(nhid, nclass, adj)
        self.dropout = dropout
        self.is_cuda = cuda
        self.normalize = utils.torch_normalize_adj_sparse
        self.adj_nonzero = torch.nonzero(adj, as_tuple=False).shape[0]
        self.adj_mask1_train = nn.Parameter(self.generate_adj_mask(adj))
        self.adj_mask2_fixed = nn.Parameter(self.generate_adj_mask(adj), requires_grad=False)

    def forward(self, x, adj):
        #self.adj_mask1_train = torch.sparse_coo_tensor(adj.to_sparse().indices(), self.adj_mask1_train, adj.size()).to_dense()
        #self.adj_mask2_fixed = torch.sparse_coo_tensor(adj.to_sparse().indices(), self.adj_mask2_fixed, adj.size()).to_dense()
        adj = torch.mul(adj, self.adj_mask1_train)
        adj = torch.mul(adj, self.adj_mask2_fixed)
        adj = adj.to_sparse()
        adj = self.normalize(adj, self.is_cuda)
        adj = adj.to_dense()
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1) 

    def generate_adj_mask(self, input_adj): 
        sparse_adj = input_adj
        zeros = torch.zeros_like(sparse_adj)
        ones = torch.ones_like(sparse_adj)
        mask = torch.where(sparse_adj != 0, ones, zeros)
        return mask  

