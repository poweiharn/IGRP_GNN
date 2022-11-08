import numpy as np
import scipy.sparse as sp
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

from sklearn.manifold import TSNE
from dataloader import load_citation_data, get_wiki_dataset


#torch.set_printoptions(profile="full")

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot




def process_data(data_path, dataset_name):
    if dataset_name == 'cora' or dataset_name == 'citeseer' or dataset_name == 'pubmed':
        features, labels, dir_adj, idx_train, idx_val, idx_test = load_citation_data(dataset_name, data_path)

        adj = dir_adj + dir_adj.T.multiply(dir_adj.T > dir_adj) - dir_adj.multiply(dir_adj.T > dir_adj)
        labels = np.argmax(labels, axis=1)

        features = torch.from_numpy(features)
        labels = torch.LongTensor(labels)
        adj = sparse_mx_to_torch_sparse_tensor(adj).to_dense()

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
    if dataset_name == 'wiki':
      adj, features, labels, idx_train, idx_val, idx_test = get_wiki_dataset(dataset_name, data_path)
      adj = sparse_mx_to_torch_sparse_tensor(adj).to_dense()
   
    return adj, features, labels, idx_train, idx_val, idx_test   



def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
    
def torch_normalize_adj(adj, cuda):
    if cuda:
      adj = adj + torch.eye(adj.shape[0]).cuda()
    else:
      adj = adj + torch.eye(adj.shape[0])
    rowsum = adj.sum(1)
    d_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_inv_sqrt[torch.isnan(d_inv_sqrt)] = 0.0
    if cuda:
      d_mat_inv_sqrt = torch.diag(d_inv_sqrt).cuda()
    else:
      d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    return adj.mm(d_mat_inv_sqrt).t().mm(d_mat_inv_sqrt)


def torch_normalize_adj_sparse(adj, cuda):
    if cuda:
      adj = adj + torch.eye(adj.shape[0]).to_sparse().cuda()
    else:
      adj = adj + torch.eye(adj.shape[0]).to_sparse()
    rowsum = torch.sparse.sum(adj,dim=1).to_dense()
    d_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_inv_sqrt[torch.isnan(d_inv_sqrt)] = 0.0
    if cuda:
      d_mat_inv_sqrt = torch.diag(d_inv_sqrt).to_sparse().cuda()
    else:
      d_mat_inv_sqrt = torch.diag(d_inv_sqrt).to_sparse()
    return torch.sparse.mm(torch.sparse.mm(adj,d_mat_inv_sqrt).t_(),d_mat_inv_sqrt)

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def plot_confusion(output, labels):
    preds = output.max(1)[1].type_as(labels)
    multiclass = np.array(confusion_matrix(labels, preds))
    fig, ax = plot_confusion_matrix(conf_mat=multiclass,
                                    colorbar=True,
                                    show_absolute=False,
                                    show_normed=True)
    #plt.savefig("confusion.png")
    plt.show()


def plot_tsne(output, labels, features):
    preds = output.max(1)[1].type_as(labels)
    tsne = TSNE(n_components=2)
    low_dim_embs = tsne.fit_transform(features)
    plt.title('Tsne result')
    plt.scatter(low_dim_embs[:, 0], low_dim_embs[:, 1], marker='o', c=preds, cmap="jet", alpha=0.7, )
    #plt.savefig("tsne.png")
    plt.show()



def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def torch_sparse_to_sparse_coo_tensor(torch_sparse_mx):
  nnz = torch_sparse_mx.nnz()
  row, col, value = torch_sparse_mx.coo()
  size = torch_sparse_mx.sparse_sizes()
  index = torch.vstack((row, col))
  values = torch.ones(nnz)
  return torch.sparse_coo_tensor(index, values, size=size)




