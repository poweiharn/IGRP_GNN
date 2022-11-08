from __future__ import division
from __future__ import print_function

import sys
import time
import copy
import argparse
import numpy as np

import torch
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import torch.optim as optim

from utils import process_data, accuracy, plot_confusion, plot_tsne
from gnns.models import UGS_GCN as GCN
from pruning import UGS_pruning
from torchprofile import profile_macs 
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--gpu', type=int, default='0',
                        help='number of GPU device to use (default: 0)')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--data', type=str, default="pubmed", choices=["cora", "pubmed", "citeseer"], help='dataset.')

parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--result-dir', type=str, default='results/',
                        help='path to directory to save results (default: "results/")')
 ###### Unify pruning settings #######
parser.add_argument('--s1', type=float, default=0.0001,help='scale sparse rate (default: 0.0001)')
parser.add_argument('--s2', type=float, default=0.0001,help='scale sparse rate (default: 0.0001)')
#parser.add_argument('--pruning_percent_wei', type=float, default=0.5)
#parser.add_argument('--pruning_percent_adj', type=float, default=0.5)
parser.add_argument('--compression_weight', type=float, default=1.0,
                    help='power of 10') 
parser.add_argument('--compression_adj', type=float, default=0,
                    help='power of 10') 
parser.add_argument('--pre_epochs', type=int, default=3,
                    help='Number of epochs to pre-train.')
parser.add_argument('--post_epochs', type=int, default=3,
                    help='Number of epochs to post-train.') 
parser.add_argument('--init_soft_mask_type', type=str, default='normal', help='all_one, kaiming, normal, uniform')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device(("cuda:" + str(args.gpu)) if args.cuda else "cpu")
print('device', device)
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
            dropout=args.dropout, adj=adj, cuda = args.cuda)

optimizer = optim.Adam(gcn.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

torch.save(optimizer.state_dict(),"{}/optimizer.pt".format(args.result_dir))
if args.cuda:
    gcn.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

def calc_sparsity(compression):
  sparsity = 10**(-float(compression))
  return sparsity

def train(model, epoch, optim, train_type):
    t = time.time()
    model.train()
    optim.zero_grad()
    macs = profile_macs(model, (features, adj))
    print('macs', macs/1e6)
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])

    model.gc1.weight.retain_grad()
    model.gc2.weight.retain_grad()
    #model.gc1.bias.retain_grad()
    #model.gc2.bias.retain_grad()
    
    loss_train.backward()
    if train_type == "pre_train":
      UGS_pruning.subgradient_update_mask(gcn, args)
    optim.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    with torch.no_grad():
      if acc_val > best_val_acc['val_acc']:
        best_val_acc['val_acc'] = acc_val
        if train_type == "pre_train":
          best_epoch_mask = UGS_pruning.get_final_mask_epoch(gcn, adj_percent=args.pruning_percent_adj, 
            wei_percent=args.pruning_percent_wei)
          rewind_weight['adj_mask1_train'] = best_epoch_mask['adj_mask']
          rewind_weight['adj_mask2_fixed'] = best_epoch_mask['adj_mask']
          rewind_weight['gc1.weight_mask_train'] = best_epoch_mask['weight1_mask']
          rewind_weight['gc1.weight_mask_fixed'] = best_epoch_mask['weight1_mask']
          rewind_weight['gc2.weight_mask_train'] = best_epoch_mask['weight2_mask']
          rewind_weight['gc2.weight_mask_fixed'] = best_epoch_mask['weight2_mask']

    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))
    
    print("Best val acc:", best_val_acc['val_acc'])

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
    '''
    print(
        "Sparsity in adj_matrix: {:.2f}%".format(
            100. * float(torch.sum(adj == 0))
            / float(adj.nelement())
        )
    )
    
    print(
        "Sparsity in adj_mask: {:.2f}%".format(
            100. * float(torch.sum(rewind_weight['adj_mask1_train'] == 0))
            / float(rewind_weight['adj_mask1_train'].nelement())
        )
    )

    print("Prune adj percent: {:.2f}%".format(100. * (1 - (float(torch.sum(rewind_weight['adj_mask1_train'] != 0)) / float(torch.sum(adj != 0))))))
    '''

    #added adj into global sparsity

    '''print(
        "Global sparsity: {:.2f}%".format(
            100. * float(
                + torch.sum(model.gc1.weight == 0)
                + torch.sum(model.gc2.weight == 0)
                + torch.sum(model.gc1.bias == 0)
                + torch.sum(model.gc2.bias == 0)
                + torch.sum(adj != 0 )
                - torch.sum(rewind_weight['adj_mask1_train'] != 0)
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

best_val_acc = {'val_acc': 0}

args.pruning_percent_wei = 1 - calc_sparsity(args.compression_weight)
args.pruning_percent_adj = 1 - calc_sparsity(args.compression_adj)
#initialize masks
UGS_pruning.add_mask(gcn)
UGS_pruning.soft_mask_init(gcn, args.init_soft_mask_type, args.seed)

# Pre-Train model
print('--------------Pre training GCN--------------')
rewind_weight = copy.deepcopy(gcn.state_dict())
t_total = time.time()
for epoch in range(args.pre_epochs):
    train(gcn, epoch, optimizer, "pre_train")
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
# Pre-Testing
test(gcn)

print('--------------Post training GCN--------------')
#post-train
best_val_acc['val_acc'] = 0
gcn.load_state_dict(rewind_weight)
#adj_spar, wei_spar = UGS_pruning.print_sparsity(gcn)
for name, param in gcn.named_parameters():
  if 'mask' in name:
      param.requires_grad = False
optimizer.load_state_dict(torch.load("{}/optimizer.pt".format(args.result_dir), map_location=device))

t_total = time.time()
for epoch in range(args.post_epochs):
    train(gcn, epoch, optimizer, "post_train")

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
# Post-Testing

test(gcn)