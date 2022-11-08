import torch
import torch.nn.utils.prune as prune
import numpy as np
import torch.nn.functional as F
from utils import process_data
class IGRP_HighOrder_GradientScoreBasedPruning():
  def __init__(self, model, threshold, schedule, sparsity, separate_compression, weight_sparsity, adj_sparsity, bias_sparsity, optimizer, dataset, prune_weight, prune_bias, prune_adj, cuda, top_k_adj, top_k_weight, top_k_bias):
          self.model = model
          self.threshold = threshold
          self.schedule = schedule
          self.sparsity  = sparsity
          self.separate_compression = separate_compression
          self.weight_sparsity = weight_sparsity
          self.adj_sparsity = adj_sparsity
          self.bias_sparsity = bias_sparsity
          self.optimizer = optimizer
          self.dataset = dataset
          self.prune_weight = prune_weight
          self.prune_bias = prune_bias
          self.prune_adj = prune_adj
          self.cuda = cuda
          if self.prune_weight:
            self.gc1_weight_score_sum = torch.zeros(self.model.gc1.weight.shape)
            self.gc2_weight_score_sum = torch.zeros(self.model.gc2.weight.shape)
          if self.prune_adj:
            self.adj_score_sum = torch.zeros(self.model.adj_weight.shape)
          if self.prune_bias:
            self.gc1_bias_score_sum = torch.zeros(self.model.gc1.bias.shape)
            self.gc2_bias_score_sum = torch.zeros(self.model.gc2.bias.shape)
          self.top_k_adj = top_k_adj
          self.top_k_weight = top_k_weight
          self.top_k_bias = top_k_bias
          

  def calc_threshold(self, scores, sparsity_value):
        if self.schedule == 'exponential':
          sparse = sparsity_value**(1)
        elif self.schedule == 'linear':
          sparse = 1.0 - (1.0 - sparsity_value)*(1)
        k = int((1.0 - sparse) * scores.numel())
        if not k < 1:
            threshold, _ = torch.kthvalue(scores, k)
        return threshold
  def compute_mask(self, adj, features, labels, idx_train, idx_val, idx_test):
        zero = torch.tensor([0.])
        one = torch.tensor([1.])
        adj_nonzero_count = torch.sum(adj != 0)
        #print('adj', adj)
        if self.cuda:
          zero = zero.cuda()
          one = one.cuda()
          self.model.cuda()
          features = features.cuda()
          adj = adj.cuda()
          labels = labels.cuda()
          idx_train = idx_train.cuda()
          idx_val = idx_val.cuda()
          idx_test = idx_test.cuda()
          if self.prune_weight:
            self.gc1_weight_score_sum = self.gc1_weight_score_sum.cuda()
            self.gc2_weight_score_sum = self.gc2_weight_score_sum.cuda()
          if self.prune_adj:
            self.adj_score_sum = self.adj_score_sum.cuda()
          if self.prune_bias:
            self.gc1_bias_score_sum = self.gc1_bias_score_sum.cuda()
            self.gc2_bias_score_sum = self.gc2_bias_score_sum.cuda()
          
        self.model.train()
        self.optimizer.zero_grad()
          
        output = self.model(features, adj)
        
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        
        self.model.gc1.weight.retain_grad()
        self.model.gc2.weight.retain_grad()
        if(self.prune_bias):
          self.model.gc1.bias.retain_grad()
          self.model.gc2.bias.retain_grad()
        if self.prune_adj:
          self.model.adj_weight.retain_grad()

        loss_train.backward()
        self.optimizer.step()

        #init masks
        mask_gc1_weight = torch.ones(self.model.gc1.weight.shape)
        mask_gc2_weight = torch.ones(self.model.gc2.weight.shape)
        mask_gc1_bias = torch.ones(self.model.gc1.bias.shape)
        mask_gc2_bias = torch.ones(self.model.gc2.bias.shape)
        mask_adj = torch.ones(adj.shape) 

        scores = {}
        if(self.prune_weight):
          for gc1_weight_param in self.model.gc1.top_k_weight_list:
            self.gc1_weight_score_sum = torch.add(self.gc1_weight_score_sum, gc1_weight_param['score'])
          for gc2_weight_param in self.model.gc2.top_k_weight_list:
            self.gc2_weight_score_sum = torch.add(self.gc2_weight_score_sum, gc2_weight_param['score'])

          scores['gc1_weight'] = torch.clone(torch.div(self.gc1_weight_score_sum, self.top_k_weight)).detach().abs_()
          scores['gc2_weight'] = torch.clone(torch.div(self.gc2_weight_score_sum, self.top_k_weight)).detach().abs_()
          weight_scores = torch.cat([torch.flatten(scores['gc1_weight']), torch.flatten(scores['gc2_weight'])])

        if(self.prune_bias):
          for gc1_bias_param in self.model.gc1.top_k_bias_list:
            self.gc1_bias_score_sum = torch.add(self.gc1_bias_score_sum, gc1_bias_param['score'])
          for gc2_bias_param in self.model.gc2.top_k_bias_list:
            self.gc2_bias_score_sum = torch.add(self.gc2_bias_score_sum, gc2_bias_param['score'])
          
          scores['gc1_bias'] = torch.clone(torch.div(self.gc1_bias_score_sum, self.top_k_bias)).detach().abs_()
          scores['gc2_bias'] = torch.clone(torch.div(self.gc2_bias_score_sum, self.top_k_bias)).detach().abs_()
          bias_scores = torch.cat([torch.flatten(scores['gc1_bias']), torch.flatten(scores['gc2_bias'])])

        if self.prune_adj:
          for adj_param in self.model.top_k_adj_list:
            self.adj_score_sum = torch.add(self.adj_score_sum, adj_param['score'])
          score_adj_weight = torch.clone(torch.div(self.adj_score_sum, self.top_k_adj)).detach().abs_()
          temp_adj_score = torch.clone(score_adj_weight)
          temp_adj_score = torch.flatten(temp_adj_score)
          temp_adj_score, indices = torch.sort(temp_adj_score, descending=True)
          scores['adj_weight'] = temp_adj_score[0:adj_nonzero_count]
          adj_scores = torch.cat([torch.flatten(scores['adj_weight'])])
        global_scores = torch.cat([torch.flatten(v) for v in scores.values()])

        if self.separate_compression:
          if(self.prune_weight):
            threshold_weight = self.calc_threshold(weight_scores, self.weight_sparsity)
            mask_gc1_weight = torch.where(scores['gc1_weight']  <= threshold_weight, zero, one)
            mask_gc2_weight = torch.where(scores['gc2_weight'] <= threshold_weight, zero, one)
          if(self.prune_bias):
            threshold_bias = self.calc_threshold(bias_scores, self.bias_sparsity)
            mask_gc1_bias = torch.where(scores['gc1_bias']  <= threshold_bias, zero, one)
            mask_gc2_bias = torch.where(scores['gc2_bias'] <= threshold_bias, zero, one)
          if self.prune_adj:
            threshold_adj = self.calc_threshold(adj_scores, self.adj_sparsity)
            mask_adj = torch.where(score_adj_weight <= threshold_adj, zero, one)
            mask_adj = torch.sparse_coo_tensor(adj.to_sparse().indices(), mask_adj, adj.size()).to_dense()
        else:
          threshold = self.calc_threshold(global_scores, self.sparsity)

          if(self.prune_weight):
            mask_gc1_weight = torch.where(scores['gc1_weight']  <= threshold, zero, one)
            mask_gc2_weight = torch.where(scores['gc2_weight'] <= threshold, zero, one)
          if(self.prune_bias):
            mask_gc1_bias = torch.where(scores['gc1_bias']  <= threshold, zero, one)
            mask_gc2_bias = torch.where(scores['gc2_bias'] <= threshold, zero, one)
          if self.prune_adj:
            mask_adj = torch.where(score_adj_weight <= threshold, zero, one)
            mask_adj = torch.sparse_coo_tensor(adj.to_sparse().indices(), mask_adj, adj.size()).to_dense() 
           
        return mask_gc1_weight, mask_gc2_weight, mask_gc1_bias, mask_gc2_bias, mask_adj
  
