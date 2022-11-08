import torch
import torch.nn.utils.prune as prune
import numpy as np
import torch.nn.functional as F
from utils import process_data

class SynflowPruning():

  def __init__(self, model, prune_epochs, epoch, threshold, schedule, sparsity, separate_compression, weight_sparsity, adj_sparsity, bias_sparsity, optimizer, dataset, prune_weight, prune_bias, prune_adj, cuda):
          self.model = model
          self.prune_epochs = prune_epochs
          self.epoch = epoch
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

  @torch.no_grad()
  def linearize(self, model):
      # model.double()
      signs = {}
      for name, param in model.state_dict().items():
          signs[name] = torch.sign(param)
          param.abs_()
      return signs
  
  @torch.no_grad()
  def nonlinearize(self, model, signs):
      # model.float()
      for name, param in model.state_dict().items():
          param.mul_(signs[name])

  def calc_threshold(self, scores, sparsity_value):

        if self.schedule == 'exponential':
          sparse = sparsity_value**((self.epoch + 1) / self.prune_epochs)
        elif self.schedule == 'linear':
          sparse = 1.0 - (1.0 - sparsity_value)*((self.epoch + 1) / self.prune_epochs)

        k = int((1.0 - sparse) * scores.numel())
        if not k < 1:
            threshold, _ = torch.kthvalue(scores, k)
        return threshold

  def compute_mask(self, adj, features, labels, idx_train, idx_val, idx_test):
        zero = torch.tensor([0.])
        one = torch.tensor([1.])
        adj_nonzero_count = torch.sum(adj != 0)

        #print('adj', adj)
        ones = torch.ones(features.shape)
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
                  
          ones = ones.cuda()

        signs = self.linearize(self.model)
        self.model.train()

        output = self.model(ones, adj)

        self.model.gc1.weight.retain_grad()
        self.model.gc2.weight.retain_grad()
        if(self.prune_bias):
          self.model.gc1.bias.retain_grad()
          self.model.gc2.bias.retain_grad()
        if self.prune_adj:
          self.model.adj_weight.retain_grad()
        torch.sum(output).backward()

        #init masks
        mask_gc1_weight = torch.ones(self.model.gc1.weight.shape)
        mask_gc2_weight = torch.ones(self.model.gc2.weight.shape)
        mask_gc1_bias = torch.ones(self.model.gc1.bias.shape)
        mask_gc2_bias = torch.ones(self.model.gc2.bias.shape)
        mask_adj = torch.ones(adj.shape)
        
        scores = {}
        if(self.prune_weight):
          scores['gc1_weight'] = torch.clone(self.model.gc1.weight.grad * self.model.gc1.weight).detach().abs_()
          scores['gc2_weight'] = torch.clone(self.model.gc2.weight.grad * self.model.gc2.weight).detach().abs_()
          weight_scores = torch.cat([torch.flatten(scores['gc1_weight']), torch.flatten(scores['gc2_weight'])])
        if(self.prune_bias):
          scores['gc1_bias'] = torch.clone(self.model.gc1.bias.grad * self.model.gc1.bias).detach().abs_()
          scores['gc2_bias'] = torch.clone(self.model.gc2.bias.grad * self.model.gc2.bias).detach().abs_()
          bias_scores = torch.cat([torch.flatten(scores['gc1_bias']), torch.flatten(scores['gc2_bias'])])

        if self.prune_adj:
          score_adj_weight = torch.clone(self.model.best_adj_grad * self.model.best_adj_weight).detach().abs_()
          temp_adj_score = torch.clone(score_adj_weight)
          temp_adj_score = torch.flatten(temp_adj_score)
          temp_adj_score, indices = torch.sort(temp_adj_score, descending=True)
          scores['adj_weight'] = temp_adj_score[0:adj_nonzero_count]
          adj_scores = torch.cat([torch.flatten(scores['adj_weight'])])

        global_scores = torch.cat([torch.flatten(v) for v in scores.values()])

        self.nonlinearize(self.model, signs)

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



  