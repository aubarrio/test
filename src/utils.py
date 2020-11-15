import os
from sklearn.utils import shuffle
import scipy.sparse as sp
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import torch
import pandas as pd
import json

def read_cora_data(fp):
    data, edges = [], []
    for ro, di, files in os.walk(fp):
        for file in files:
            if '.content' in file:
                with open(os.path.join(ro, file),'r') as f:
                    data.extend(f.read().splitlines())
            elif 'cites' in file:
                with open(os.path.join(ro, file),'r') as f:
                    edges.extend(f.read().splitlines())
    data = shuffle(data)
    return data, edges

def read_twitch_data(fp):
    data, edges = [], []

    for ro, di, files in os.walk(fp):
        for file in files:
            if 'edges' in file:
                with open(os.path.join(ro, file), 'r') as f:
                    edges.extend(f.read().splitlines())
            elif 'features.json' in file:
                with open(os.path.join(ro, file),'r') as json_data:
                    j_data = json.load(json_data)
                    d = pd.DataFrame.from_dict(j_data, orient='index')
                    d['index'] = d.index
                    d['index'] = d['index'].astype(int)
                    d = d.reset_index(drop=True)
                    d = d.replace(np.NaN, 0)
            elif 'target.csv' in file:
                target_df = pd.read_csv(os.path.join(ro, file))
                target_df = target_df[['new_id', 'partner']]

    edges = edges[1:]
    combined = pd.merge(d, target_df, left_on='index', right_on='new_id').drop(['new_id'], axis=1)
    order = [combined.columns[-2]] + list(combined.columns[0:-2]) + [combined.columns[-1]]
    combined = combined[order]

    combined.to_csv(os.path.join(fp, 'features.csv'), '\t')

    for ro, di, files in os.walk(fp):
        for file in files:
            if 'features.csv' in file:
                with open(os.path.join(ro, file), 'r') as f:
                    data.extend(f.read().splitlines())

    data = data[1:]

    return data, edges

def read_facebook_data(fp):
    data, edges = [], []

    for ro, di, files in os.walk(fp):
        for file in files:
            if 'edges' in file:
                with open(os.path.join(ro, file), 'r') as f:
                    edges.extend(f.read().splitlines())
            elif 'features.json' in file:
                with open(os.path.join(ro, file),'r') as json_data:
                    j_data = json.load(json_data)
                    d = pd.DataFrame.from_dict(j_data, orient='index')
                    d['index'] = d.index
                    d['index'] = d['index'].astype(int)
                    d = d.reset_index(drop=True)
                    d = d.replace(np.NaN, 0)
            elif 'target.csv' in file:
                target_df = pd.read_csv(os.path.join(ro, file))
                target_df = target_df[['id', 'page_type']]

    edges = edges[1:]
    combined = pd.merge(d, target_df, left_on='index', right_on='id').drop(['id'], axis=1)
    order = [combined.columns[-2]] + list(combined.columns[0:-2]) + [combined.columns[-1]]
    combined = combined[order]

    combined.to_csv(os.path.join(fp, 'features.csv'), '\t')

    for ro, di, files in os.walk(fp):
        for file in files:
            if 'features.csv' in file:
                with open(os.path.join(ro, file), 'r') as f:
                    data.extend(f.read().splitlines())

    data = data[1:]

    return data, edges

def parse_data(data):
    labels, nodes, X = [], [], []
    for i, data in enumerate(data):
        features = data.split('\t')
        labels.append(features[-1])
        X.append(features[1:-1])
        nodes.append(features[0])

    X = np.array(X, dtype=float)
    X = np.array(X, dtype=int)
    return labels, np.array(nodes, dtype=np.int32), X

def parse_edges(edges):
    edge_list = []
    for edge in edges:
        e = edge.split('\t')
        edge_list.append([e[0],e[1]])
    return np.array(edge_list).astype('int32')

def build_features(X):
    features = sp.csr_matrix(X, dtype=np.float32)
    features = normalize(features)
    features = torch.FloatTensor(np.array(features.todense()))
    return features

def encode_label(labels):
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    labels = to_categorical(labels)
    return labels.astype('int32')

def build_edges(idx, edges):
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges.flatten())),
                        dtype=np.int32).reshape(edges.shape)
    return edges

def build_adj(edges, labels):
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
              shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj

# def build_idx():
#     idx_train = torch.LongTensor(range(140))
#     idx_val = torch.LongTensor(range(200, 500))
#     idx_test = torch.LongTensor(range(500, 1500))
#     return idx_train, idx_val, idx_test

def build_idx(shape):
    # print("Original Shape: " + str(shape))
    x1 = int(0.6 * shape)
    # print("x1 value: " + str(x1))
    x2 = x1 + int(0.2 * shape)
    # print("x2 value: " + str(x2))
    x3 = x2 + int(0.2 * shape)
    # print("x3 value: " + str(x3))
    idx_train = torch.LongTensor(range(x1))
    idx_val = torch.LongTensor(range(x1, x2))
    idx_test = torch.LongTensor(range(x2, x3 - 1))
    return idx_train, idx_val, idx_test

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
