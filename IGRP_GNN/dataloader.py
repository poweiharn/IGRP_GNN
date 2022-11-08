import sys
import pickle as pkl
import numpy as np
import networkx as nx
import scipy.sparse as sp
import json
import torch
import itertools
import dgl
from dgl import DGLGraph

def norm_feat(features):
    row_sum_inv = np.power(np.sum(features, axis=1), -1)
    row_sum_inv[np.isinf(row_sum_inv)] = 0.
    deg_inv = np.diag(row_sum_inv)
    norm_features = np.dot(deg_inv, features)
    norm_features = np.array(norm_features, dtype=np.float32)

    return norm_features

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_citation_data(dataset_name, dataset_path):
    """
    Load Citation Networks Datasets.
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    dataset_path = dataset_path + dataset_name + '/'
    for i in range(len(names)):
        with open(dataset_path + 'ind.{}.{}'.format(dataset_name.lower(), names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(dataset_path + 'ind.{}.test.index'.format(dataset_name))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_name == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = features.toarray()
    features = norm_feat(features)

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = sp.csc_matrix(adj)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)
    idx_test = test_idx_range.tolist()

    return features, labels, adj, idx_train, idx_val, idx_test

def get_wiki_dataset(dataset_name, dataset_path):
  dataset_path = dataset_path + dataset_name + '/data.json'
  data = json.load(open(dataset_path))
  features = torch.FloatTensor(np.array(data['features']))
  labels = torch.LongTensor(np.array(data['labels']))
  if hasattr(torch, 'BoolTensor'):
      train_masks = [torch.BoolTensor(tr) for tr in data['train_masks']]
      val_masks = [torch.BoolTensor(val) for val in data['val_masks']]
      stopping_masks = [torch.BoolTensor(st) for st in data['stopping_masks']]
      test_mask = torch.BoolTensor(data['test_mask'])
  else:
      train_masks = [torch.ByteTensor(tr) for tr in data['train_masks']]
      val_masks = [torch.ByteTensor(val) for val in data['val_masks']]
      stopping_masks = [torch.ByteTensor(st) for st in data['stopping_masks']]
      test_mask = torch.ByteTensor(data['test_mask'])
  n_feats = features.shape[1]
  n_classes = len(set(data['labels']))

  g = DGLGraph()
  g.add_nodes(len(data['features']))
  edge_list = list(itertools.chain(*[[(i, nb) for nb in nbs] for i,nbs in enumerate(data['links'])]))
  n_edges = len(edge_list)
  # add edges two lists of nodes: src and dst
  src, dst = tuple(zip(*edge_list))
  g.add_edges(src, dst)
  # edges are directional in DGL; make them bi-directional
  g.add_edges(dst, src)
  g = dgl.to_networkx(g)
  adj = nx.adjacency_matrix(g)
  adj = sp.csc_matrix(adj)
  return adj, features, labels, train_masks, val_masks, test_mask

def main():
    dataset_name = 'citeseer'
    data_path = './data/'
    if dataset_name == 'cora' or dataset_name == 'citeseer' or dataset_name == 'pubmed':
        features, labels, dir_adj, idx_train, idx_val, idx_test = load_citation_data(dataset_name, data_path)
        print(features)
        print(labels)
        print(dir_adj)
        print(idx_train)
        print(idx_val)
        print(idx_test)


if __name__ == "__main__":
    main()