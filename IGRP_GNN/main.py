from __future__ import division
from __future__ import print_function

import sys
import time
import math
import argparse
import numpy as np
from tensorflow.python.framework.test_util import for_all_test_methods

import torch
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import grad

from torchprofile import profile_macs
import warnings

from Args_JSON import Args_JSON

from utils import process_data, accuracy, plot_confusion, plot_tsne
from gnns.models import GCN
from pruning.synflow_pruning import SynflowPruning
from pruning.igrp_high_order_score_based_pruning import IGRP_HighOrder_GradientScoreBasedPruning


warnings.filterwarnings('ignore')
# Training settings
parser = argparse.ArgumentParser()

parser.add_argument('--args_mode', type=str, default="default", choices=["default", "load_config", "export_config"], 
            help='default: User default args, load_config: load config from json file, export_config: export args to json file')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--gpu', type=int, default='0',
                        help='number of GPU device to use (default: 0)')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--data', type=str, default="cora", choices=["cora", "pubmed", "citeseer", "wiki"], help='dataset.')

#wiki dataset
parser.add_argument('--wiki_split_index', type=int, default=0, help='20(0-19) different training splits')

parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--weight_pruning_threshold', type=float, default=0.1,
                    help='Dropout rate (1 - keep probability).')

#new parameters 
parser.add_argument('--experiment', type=str, default='singleshot', 
                        choices=['singleshot','multishot'],
                        help='experiment name (default: example)') 
parser.add_argument('--pre_epochs', type=int, default=100,
                    help='Number of epochs to pre-train.')
parser.add_argument('--post_epochs', type=int, default=100,
                    help='Number of epochs to post-train.')                   
parser.add_argument('--pruner', type=str, default='igrp_high_order',
                    choices = ['synflow', 'igrp_high_order'], help='Prune strategy')
parser.add_argument('--prune_epochs', type=int, default=1,
                    help='no. of iterations for scoring')
parser.add_argument('--top_k_adj', type=int, default=1,
                    help='value of top k adj')
parser.add_argument('--top_k_weight', type=int, default=1,
                    help='value of top k weight')
parser.add_argument('--top_k_bias', type=int, default=10,
                    help='value of top k bias')
parser.add_argument('--connectivity_order', type=int, default=1,
                    help='value of connectivity_order of igrp_high_order score calculation')

parser.add_argument('--prune-batchnorm', type=bool, default=False,
                    help='whether to prune batchnorm layers')
parser.add_argument('--prune-residual', type=bool, default=False,
                    help='whether to prune residual connections')
parser.add_argument('--compression-schedule', type=str, default='exponential', choices=['linear','exponential'],
                        help='whether to use a linear or exponential compression schedule (default: exponential)')
parser.add_argument('--compression', type=float, default=1.0,
                    help='power of 10(should not be set to 0.0)') 
parser.add_argument('--separate_compression', type=bool, default=False,
                    help='whether to use separate compression')                  
parser.add_argument('--compression_weight', type=float, default=0,
                    help='power of 10') 
parser.add_argument('--compression_adj', type=float, default=0.25,
                    help='power of 10') 
parser.add_argument('--compression_bias', type=float, default=0,
                    help='power of 10') 

# At least one needs to be True for pruning
parser.add_argument('--prune_weight', type=bool, default=False,
                    help='whether to prune weight')
parser.add_argument('--prune_bias', type=bool, default=False,
                    help='whether to prune bias params')
parser.add_argument('--prune_adj', type=bool, default=True,
                    help='whether to prune adj matrix')

#new parameters used in multishot experiment
parser.add_argument('--compression-list', type=float, nargs='*', default=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
                        help='list of compression ratio exponents for singleshot/multishot (default: [])')
parser.add_argument('--compression-list_weight', type=float, nargs='*', default=[0.5, 1.0],
                        help='list of compression ratio exponents for singleshot/multishot (default: [])')
parser.add_argument('--compression-list_adj', type=float, nargs='*', default=[0.5, 1.0],
                        help='list of compression ratio exponents for singleshot/multishot (default: [])')
parser.add_argument('--compression-list_bias', type=float, nargs='*', default=[0.5, 1.0],
                        help='list of compression ratio exponents for singleshot/multishot (default: [])')
parser.add_argument('--level-list', type=int, nargs='*', default=[1,2,3],
                        help='list of number of prune-train cycles (levels) for multishot (default: [])')
parser.add_argument('--result-dir', type=str, default='results/',
                        help='path to directory to save results (default: "results/")')

args = parser.parse_args()

args_json = Args_JSON(args)

print("Config Mode: ",args.args_mode)
if args.args_mode == "load_config":
  args = args_json.read(parser,"configs/temp_config.json")
elif args.args_mode == "export_config":
  args_json.export(args,"configs/export_config.json")
  print("Finished export config to json!!")
  sys.exit()


args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device(("cuda:" + str(args.gpu)) if args.cuda else "cpu")
print('device: ', device)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, features, labels, idx_train, idx_val, idx_test = process_data("./data/", args.data)

# Model and optimizer
gcn = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout, adj=adj, prune_adj = args.prune_adj, cuda = args.cuda)


optimizer = optim.Adam(gcn.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

torch.save(optimizer.state_dict(),"{}/optimizer.pt".format(args.result_dir))

if args.cuda:
    gcn.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    if args.data == 'wiki':
      idx_train = idx_train[args.wiki_split_index].cuda()
      idx_val = idx_val[args.wiki_split_index].cuda()
    else:
      idx_train = idx_train.cuda()
      idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

if args.data == 'wiki':
  if args.prune_adj == False:
    adj = adj.to_sparse()
    adj = gcn.normalize(adj, args.cuda)
    adj = adj.to_dense()

def init_mask(model):
  mask_gc1_w = torch.ones(model.gc1.weight.shape).to(device)
  mask_gc2_w = torch.ones(model.gc2.weight.shape).to(device)
  mask_gc1_b = torch.ones(model.gc1.bias.shape).to(device)
  mask_gc2_b = torch.ones(model.gc2.bias.shape).to(device)
  adj_mask = adj

  return mask_gc1_w, mask_gc2_w, mask_gc1_b, mask_gc2_b, adj_mask

def apply_mask(model, mask_gc1_w, mask_gc2_w, mask_gc1_b, mask_gc2_b):
  #apply_mask
  if args.prune_weight:
    prune.custom_from_mask(model.gc1, name='weight', mask = mask_gc1_w)
    prune.custom_from_mask(model.gc2, name='weight', mask = mask_gc2_w)
  if(args.prune_bias):
    prune.custom_from_mask(model.gc1, name='bias', mask = mask_gc1_b)
    prune.custom_from_mask(model.gc2, name='bias', mask = mask_gc2_b)

def calc_sparsity(compression):
  sparsity = 10**(-float(compression))
  return sparsity

def calc_sparsity_multishot(compression, l, level):
  sparsity = (10**(-float(compression)))**((l + 1) / level)
  return sparsity

def compression_list_check():
  length = 0
  if args.prune_weight:
      if(len(args.compression_list_weight) == 0):
        print('The compression_list_weight is empty')
        sys.exit()
      else:
        length = len(args.compression_list_weight)
  if args.prune_adj:
    if(len(args.compression_list_adj) == 0):
      print('The compression_list_adj is empty')
      sys.exit()
    else:
      length = len(args.compression_list_adj)
  if args.prune_bias:
    if(len(args.compression_list_bias) == 0):
      print('The compression_list_bias is empty')
      sys.exit()
    else:
      length = len(args.compression_list_bias)

  if args.prune_weight and args.prune_adj:
    if(len(args.compression_list_weight) != len(args.compression_list_adj)):
      print('The compression_lists lengths are not equal')
      sys.exit()
  if args.prune_adj and args.prune_bias:
    if(len(args.compression_list_adj) != len(args.compression_list_bias)):
      print('The compression_lists lengths are not equal')
      sys.exit()
  if args.prune_bias and args.prune_weight:
    if(len(args.compression_list_bias) != len(args.compression_list_weight)):
      print('The compression_lists lengths are not equal')
      sys.exit()
  if args.prune_weight and args.prune_adj and args.prune_bias:
    if(not(len(args.compression_list_weight) == len(args.compression_list_adj) == len(args.compression_list_bias))):
      print('The compression_lists lengths are not equal')
      sys.exit()
    
  return length


def nth_derivative(f, wrt, n):

    for i in range(n):

        grads = grad(f, wrt, create_graph=True)[0]
        f = grads.sum()

    return grads


def connectivity_score(w, n, loss):
  total_score = torch.zeros_like(w) 
  while n > 0:
    total_w = w
    for _ in range(n-1):
      total_w = total_w * w
    score = 1/math.factorial(n) * total_w * nth_derivative(f=loss, wrt=w, n=n)
    if n % 2 == 1:
      total_score = total_score + score
    else:
      total_score = total_score - score
    n = n-1
  return total_score

def top_k_selection(top_k_list, param_dict, top_k_value):
    if len(top_k_list) < top_k_value:
      top_k_list.append(param_dict)
      top_k_list = sorted(top_k_list, key=lambda param: param['val_acc'], reverse=True)
    else:
      is_found = False
      for param in top_k_list:
        if is_found is False:
          if param_dict['val_acc'] > param['val_acc']:
            is_found = True
            top_k_list.pop()
            top_k_list.append(param_dict)
            top_k_list = sorted(top_k_list, key=lambda param: param['val_acc'], reverse=True)

def train(model, epoch, optim ,train_type):
    t = time.time()
    model.train()
    optim.zero_grad()
    macs = profile_macs(model, (features, adj))
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])

    gc1_weight_param_dict = {}
    gc2_weight_param_dict = {}
    adj_param_dict = {}
    gc1_bias_param_dict = {}
    gc2_bias_param_dict = {}

    if args.pruner == 'igrp_high_order' and train_type == 'pre_train':
      if args.prune_weight:
        gc1_weight_param_dict['score'] = connectivity_score(model.gc1.weight, args.connectivity_order, loss_train).detach()
        gc2_weight_param_dict['score'] = connectivity_score(model.gc2.weight, args.connectivity_order, loss_train).detach()
      if args.prune_adj:
        adj_param_dict['score'] = connectivity_score(model.adj_weight, 1, loss_train).detach()
      if args.prune_bias:
        gc1_bias_param_dict['score'] = connectivity_score(model.gc1.bias, args.connectivity_order, loss_train).detach()
        gc2_bias_param_dict['score'] = connectivity_score(model.gc2.bias, args.connectivity_order, loss_train).detach()

    model.gc1.weight.retain_grad()
    model.gc2.weight.retain_grad()
    #model.gc1.bias.retain_grad()
    #model.gc2.bias.retain_grad()
    if args.prune_adj:
      model.adj_weight.retain_grad()
    
    
    loss_train.backward()

    optim.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])

    if train_type == 'pre_train':
      if args.pruner == 'igrp_high_order':
        if args.prune_weight:
          gc1_weight_param_dict['val_acc'] = acc_val
          gc2_weight_param_dict['val_acc'] = acc_val
          gc1_weight_param_dict['weight'] = torch.clone(model.gc1.weight).detach()
          gc2_weight_param_dict['weight'] = torch.clone(model.gc2.weight).detach()
          

          top_k_selection(model.gc1.top_k_weight_list, gc1_weight_param_dict, args.top_k_weight)
          top_k_selection(model.gc2.top_k_weight_list, gc2_weight_param_dict, args.top_k_weight)
        if args.prune_adj:
          adj_param_dict['val_acc'] = acc_val
          adj_param_dict['adj_weight'] = torch.clone(model.adj_weight).detach()
          
          top_k_selection(model.top_k_adj_list, adj_param_dict, args.top_k_adj)
        if args.prune_bias:
          gc1_bias_param_dict['val_acc'] = acc_val
          gc1_bias_param_dict['bias'] = torch.clone(model.gc1.bias).detach()
          gc2_bias_param_dict['val_acc'] = acc_val
          gc2_bias_param_dict['bias'] = torch.clone(model.gc2.bias).detach()

          top_k_selection(model.gc1.top_k_bias_list, gc1_bias_param_dict, args.top_k_bias)
          top_k_selection(model.gc2.top_k_bias_list, gc2_bias_param_dict, args.top_k_bias)

    with torch.no_grad():
      if acc_val > best_val_acc['val_acc']:
        best_val_acc['val_acc'] = acc_val
        if args.prune_adj:
          model.best_adj_weight.copy_(model.adj_weight)
          if model.adj_weight.grad is not None:
            model.best_adj_grad.copy_(model.adj_weight.grad)

    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))
    
    print("Best val acc:", best_val_acc['val_acc'])
    print('Inference MACs:[{:.2f}M]'.format(macs/1e6))

    print(
        "Sparsity in gc1.weight: {:.2f}%".format(
            100. * float(torch.sum(model.gc1.weight == 0))
            / float(model.gc1.weight.nelement())
        )
    )
    print(
        "Sparsity in gc2.weight: {:.2f}%".format(
            100. * float(torch.sum(model.gc2.weight == 0))
            / float(model.gc2.weight.nelement())
        )
    )
    #weight sparsity
    print(
        "Weight Sparsity: {:.2f}%".format(
            100. * (float(torch.sum(model.gc1.weight == 0)) + float(torch.sum(model.gc2.weight == 0)))
            / (float(model.gc1.weight.nelement()) + float(model.gc2.weight.nelement()))
        )
    )
    print(
        "Sparsity in gc1.bias: {:.2f}%".format(
            100. * float(torch.sum(model.gc1.bias == 0))
            / float(model.gc1.bias.nelement())
        )
    )
    print(
        "Sparsity in gc2.bias: {:.2f}%".format(
            100. * float(torch.sum(model.gc2.bias == 0))
            / float(model.gc2.bias.nelement())
        )
    )
    if args.prune_adj:
      print(
          "Sparsity in adj_weight: {:.2f}%".format(
              100. * float(torch.sum(model.adj_weight == 0))
              / float(model.adj_weight.nelement())
          )
      )
    print(
        "Sparsity in adj_matrix: {:.2f}%".format(
            100. * float(torch.sum(adj == 0))
            / float(adj.nelement())
        )
    )
    
    print(
        "Sparsity in adj_mask: {:.2f}%".format(
            100. * float(torch.sum(model.adj_mask == 0))
            / float(model.adj_mask.nelement())
        )
    )

    print("Prune adj percent: {:.2f}%".format(100. * (1 - (float(torch.sum(model.adj_mask != 0)) / float(torch.sum(adj != 0))))))
    #added adj into global sparsity
    '''print(
        "Global sparsity: {:.2f}%".format(
            100. * float(
                + torch.sum(model.gc1.weight == 0)
                + torch.sum(model.gc2.weight == 0)
                + torch.sum(model.gc1.bias == 0)
                + torch.sum(model.gc2.bias == 0)
                + torch.sum(adj != 0 )
                - torch.sum(model.adj_mask != 0)
            )
            / float(
                + model.gc1.weight.nelement()
                + model.gc2.weight.nelement()
                + model.gc1.bias.nelement()
                + model.gc2.bias.nelement()
                + torch.sum(adj !=0)
            )
        )
    )'''


def test(model):
    model.eval()
    output = model(features, adj)
    #print(labels[idx_test])
    #print(output[idx_test])
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    #plot_confusion(output[idx_test], labels[idx_test])
    #plot_tsne(output[idx_test], labels[idx_test], features[idx_test])

    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

#initialize masks
mask_gc1_weight, mask_gc2_weight, mask_gc1_bias, mask_gc2_bias, adj_mask = init_mask(gcn)


#apply mask
apply_mask(gcn, mask_gc1_weight, mask_gc2_weight, mask_gc1_bias, mask_gc2_bias)
gcn.set_adj_mask(adj_mask)
torch.save(gcn.state_dict(),"{}/gcn_model.pt".format(args.result_dir))

best_val_acc = {'val_acc': 0}


if(args.experiment == 'singleshot'):

  # Pre-Train model
  print('--------------Pre training GCN--------------')

  t_total = time.time()
  for epoch in range(args.pre_epochs):
      train(gcn, epoch, optimizer, "pre_train")
  print("Optimization Finished!")
  print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
  # Pre-Testing
  test(gcn)
  #pruning
  print('--------------Pruning--------------')
  #calculate sparsity
  sparsity = calc_sparsity(args.compression)
  weight_sparsity = calc_sparsity(args.compression_weight)
  adj_sparsity = calc_sparsity(args.compression_adj)
  bias_sparsity = calc_sparsity(args.compression_bias)

  if args.pruner == 'igrp_high_order':
    top_k_gradient_pruning = IGRP_HighOrder_GradientScoreBasedPruning(
    model = gcn, threshold=args.weight_pruning_threshold, schedule = args.compression_schedule,
      sparsity=sparsity, separate_compression = args.separate_compression, weight_sparsity = weight_sparsity, adj_sparsity = adj_sparsity, bias_sparsity = bias_sparsity,
      optimizer=optimizer, dataset = args.data, prune_weight = args.prune_weight, prune_bias=args.prune_bias, prune_adj = args.prune_adj,
      cuda = args.cuda, top_k_adj = args.top_k_adj, top_k_weight = args.top_k_weight, top_k_bias = args.top_k_bias
      )
    mask_gc1_weight, mask_gc2_weight, mask_gc1_bias, mask_gc2_bias, adj_mask = top_k_gradient_pruning.compute_mask(adj, features, labels, idx_train, idx_val, idx_test)
  else:
    for epoch in range(args.prune_epochs):

      if args.pruner == 'synflow':
        synflow_pruning = SynflowPruning(
          model = gcn, prune_epochs=args.prune_epochs, epoch=epoch, threshold=args.weight_pruning_threshold, schedule = args.compression_schedule,
            sparsity=sparsity, separate_compression = args.separate_compression, weight_sparsity = weight_sparsity, adj_sparsity = adj_sparsity, bias_sparsity = bias_sparsity,
            optimizer=optimizer, dataset = args.data, prune_weight = args.prune_weight, prune_bias=args.prune_bias, prune_adj = args.prune_adj,
            cuda = args.cuda,
        )
      
      mask_gc1_weight, mask_gc2_weight, mask_gc1_bias, mask_gc2_bias, adj_mask = synflow_pruning.compute_mask(adj, features, labels, idx_train, idx_val, idx_test)
      

  print('--------------Post training GCN--------------')
  #post-train
  best_val_acc['val_acc'] = 0
  gcn.load_state_dict(torch.load("{}/gcn_model.pt".format(args.result_dir), map_location=device), strict=False)
  optimizer.load_state_dict(torch.load("{}/optimizer.pt".format(args.result_dir), map_location=device))

  #apply_mask
  if args.prune_adj:
    gcn.apply_adj_mask = True
    gcn.set_adj_mask(adj_mask)

  apply_mask(gcn, mask_gc1_weight, mask_gc2_weight, mask_gc1_bias, mask_gc2_bias)

  t_total = time.time()
  for epoch in range(args.post_epochs):
      train(gcn, epoch, optimizer, "post_train")
  
  print("Optimization Finished!")
  print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
  # Post-Testing
  
  test(gcn)


elif(args.experiment == 'multishot'):
  
  if args.separate_compression:
    compression_list_length = compression_list_check()
  else:
    compression_list_length = len(args.compression_list)

  for index in range(compression_list_length):
        for level in args.level_list:
          if args.separate_compression:
            print('{} compression weight ratio, compression adj ratio, compression bias ratio, {} train-prune levels'.format(args.compression_list_weight[index], 
              args.compression_list_adj[index], args.compression_list_bias[index], level))
          else:
            print('{} compression ratio, {} train-prune levels'.format(args.compression_list[index], level))

          #initialize masks
          mask_gc1_weight, mask_gc2_weight, mask_gc1_bias, mask_gc2_bias, adj_mask = init_mask(gcn)
          
          gcn.load_state_dict(torch.load("{}/gcn_model.pt".format(args.result_dir), map_location=device))
          optimizer.load_state_dict(torch.load("{}/optimizer.pt".format(args.result_dir), map_location=device))
          
          gcn.apply_adj_mask = False       

          for l in range(level):
            best_val_acc['val_acc'] = 0
            #apply mask
            gcn.set_adj_mask(adj_mask)

            apply_mask(gcn, mask_gc1_weight, mask_gc2_weight, mask_gc1_bias, mask_gc2_bias)

            
            # Pre-Train model
            print('--------------Pre training GCN--------------')
            for epoch in range(args.pre_epochs):
                train(gcn, epoch, optimizer, "pre_train")

            #pruning
            print('--------------Pruning--------------')
            weight_sparsity = 0
            adj_sparsity = 0
            bias_sparsity = 0
            sparsity = 0
            if args.separate_compression:
              weight_sparsity = calc_sparsity_multishot(args.compression_list_weight[index], l, level)
              adj_sparsity = calc_sparsity_multishot(args.compression_list_adj[index], l, level)
              bias_sparsity = calc_sparsity_multishot(args.compression_list_bias[index], l, level)
            else:
              sparsity = calc_sparsity_multishot(args.compression_list[index], l, level)
            if args.pruner == 'igrp_high_order':
              top_k_gradient_pruning = IGRP_HighOrder_GradientScoreBasedPruning(
                model = gcn, threshold=args.weight_pruning_threshold, schedule = args.compression_schedule,
                  sparsity=sparsity, separate_compression = args.separate_compression, weight_sparsity = weight_sparsity, adj_sparsity = adj_sparsity, bias_sparsity = bias_sparsity,
                  optimizer=optimizer, dataset = args.data, prune_weight = args.prune_weight, prune_bias=args.prune_bias, prune_adj = args.prune_adj,
                  cuda = args.cuda, top_k_adj = args.top_k_adj, top_k_weight = args.top_k_weight, top_k_bias = args.top_k_bias
                )
              mask_gc1_weight, mask_gc2_weight, mask_gc1_bias, mask_gc2_bias, adj_mask = top_k_gradient_pruning.compute_mask(adj, features, labels, idx_train, idx_val, idx_test)
            else:
              for epoch in range(args.prune_epochs):

                if args.pruner == 'synflow':
                  synflow_pruning = SynflowPruning(
                    model = gcn, prune_epochs=args.prune_epochs, epoch=epoch, threshold=args.weight_pruning_threshold, schedule = args.compression_schedule,
                      sparsity=sparsity, separate_compression = args.separate_compression, weight_sparsity = weight_sparsity, adj_sparsity = adj_sparsity, bias_sparsity = bias_sparsity,
                      optimizer=optimizer, dataset = args.data, prune_weight = args.prune_weight, prune_bias=args.prune_bias, prune_adj = args.prune_adj,
                      cuda = args.cuda,
                  )
                
                mask_gc1_weight, mask_gc2_weight, mask_gc1_bias, mask_gc2_bias, adj_mask = synflow_pruning.compute_mask(adj, features, labels, idx_train, idx_val, idx_test)

            #reset model's weights
            original_dict = torch.load("{}/gcn_model.pt".format(args.result_dir), map_location=device)
            
            original_weights = dict(filter(lambda v: (v[0].endswith(('adj_weight', '.weight', '.bias'))), original_dict.items()))
            
            
            gcn_dict = gcn.state_dict()
            gcn_dict.update(original_weights)

            
            gcn.load_state_dict(gcn_dict)
            
            optimizer.load_state_dict(torch.load("{}/optimizer.pt".format(args.result_dir), map_location=device))

          # Post-Train model
          print('--------------Post training GCN--------------')
          #apply_mask
          if args.prune_adj:
              gcn.apply_adj_mask = True
              gcn.set_adj_mask(adj_mask)
          apply_mask(gcn, mask_gc1_weight, mask_gc2_weight, mask_gc1_bias, mask_gc2_bias)
          t_total = time.time()
          
          for epoch in range(args.post_epochs):
              train(gcn, epoch, optimizer, "post_train")
          
          print("Optimization Finished!")
          print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
          # Post-Testing
          test(gcn)


