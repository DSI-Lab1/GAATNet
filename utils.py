import numpy as np
import scipy.sparse as sp
import torch
import random
import torch.nn.functional as F


# def load_data(dataset, EMBEDDING_DIM, path="./data/"):
#     path += (dataset + '/')
#     print('Loading {} dataset...'.format(dataset))
#
#     if dataset == "Cora":
#         nodes_num = 2708
#         edges_num = 5429
#     elif dataset == "Yeast":
#         nodes_num = 2375
#         edges_num = 11693
#     elif dataset == "Celegans":
#         nodes_num = 297
#         edges_num = 2148
#     elif dataset == "Citeseer":
#         nodes_num = 3305
#         edges_num = 4732
#     elif dataset == "Sexual":
#         nodes_num = 288
#         edges_num = 291
#
#     # Load data
#     features = np.load("{}{}_features.npy".format(path, dataset))
#     idx_train = torch.load("{}{}_train_edges.pt".format(path, dataset))
#     idx_val = torch.load("{}{}_validation_edges.pt".format(path, dataset))
#     idx_test = torch.load("{}{}_test_edges.pt".format(path, dataset))
#     edges = np.genfromtxt("{}{}_edges_AD.txt".format(path, dataset), dtype=np.int32)
#
#
#     # Adjacency matrix
#     adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(nodes_num, nodes_num),
#                         dtype=np.float32)
#     adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
#     adj = normalize_adj(adj + sp.eye(adj.shape[0]))
#
#     random_features = np.random.rand(features.shape[0], EMBEDDING_DIM - features.shape[1])
#     features = np.hstack((features, random_features))
#
#     adj = torch.FloatTensor(np.array(adj.todense()))
#     features = torch.FloatTensor(features)
#
#     # idx_train = torch.LongTensor(idx_train)
#     # idx_test = torch.LongTensor(idx_test)
#     # idx_val = torch.LongTensor(idx_val)
#     #
#     # train_len = int(len(idx_train) / 2)
#     # val_len = int(len(idx_val) / 2)
#     # test_len = int(len(idx_test) / 2)
#     #
#     # labels_train = torch.cat(
#     #     [torch.ones(train_len, dtype=torch.float32), torch.zeros(train_len, dtype=torch.float32)], dim=0)
#     # labels_val = torch.cat(
#     #     [torch.ones(val_len, dtype=torch.float32), torch.zeros(val_len, dtype=torch.float32)], dim=0)
#     # labels_test = torch.cat(
#     #     [torch.ones(test_len, dtype=torch.float32), torch.zeros(test_len, dtype=torch.float32)], dim=0)
#
#
#     def get_negative_edges(edges, num_neg_samples, nodes_num):
#         existing_edges = set(tuple(e) for e in edges)
#         negative_edges = set()
#         while len(negative_edges) < num_neg_samples:
#             i = random.randint(0, nodes_num - 1)
#             j = random.randint(0, nodes_num - 1)
#             if i != j and (i, j) not in existing_edges and (j, i) not in existing_edges:
#                 negative_edges.add((i, j))
#         return list(negative_edges)
#
#     def create_labels_and_index(pos_index, nodes_num):
#         pos_labels = torch.ones(len(pos_index), dtype=torch.float32)
#         neg_index = get_negative_edges(pos_index, len(pos_index), nodes_num)
#         neg_labels = torch.zeros(len(neg_index), dtype=torch.float32)
#         index = torch.LongTensor(np.concatenate([pos_index, neg_index], axis=0))
#         labels = torch.cat([pos_labels, neg_labels], dim=0)
#         return index, labels
#
#     idx_train, labels_train = create_labels_and_index(idx_train[:int(len(idx_train)/2)], nodes_num)
#     idx_val, labels_val = create_labels_and_index(idx_val[:int(len(idx_val)/2)], nodes_num)
#     idx_test, labels_test = create_labels_and_index(idx_test[:int(len(idx_test)/2)], nodes_num)
#
#
#     return adj, features, labels_train, labels_val, labels_test, idx_train, idx_val, idx_test


def load_data(dataset, EMBEDDING_DIM, path="./data/"):
    path += (dataset + '/')
    print('Loading {} dataset...'.format(dataset))

    # Load data
    features = np.load("{}{}_features.npy".format(path, dataset))
    idx_train = torch.load("{}{}_train_edges.pt".format(path, dataset))
    idx_val = torch.load("{}{}_validation_edges.pt".format(path, dataset))
    idx_test = torch.load("{}{}_test_edges.pt".format(path, dataset))
    edges = np.genfromtxt("{}{}_edges_AD.txt".format(path, dataset), dtype=np.int32)

    idx_train = torch.cat([idx_train, idx_val], dim=0)
    nodes_num = features.shape[0]

    # Adjacency matrix
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(nodes_num, nodes_num),
                        dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    random_features = np.random.rand(features.shape[0], EMBEDDING_DIM - features.shape[1])
    features = np.hstack((features, random_features))

    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(features)

    def get_negative_edges(edges, num_neg_samples, nodes_num):
        existing_edges = set(tuple(e) for e in edges)
        negative_edges = set()
        while len(negative_edges) < num_neg_samples:
            i = random.randint(0, nodes_num - 1)
            j = random.randint(0, nodes_num - 1)
            if i != j and (i, j) not in existing_edges and (j, i) not in existing_edges:
                negative_edges.add((i, j))
        return list(negative_edges)

    def create_labels_and_index(pos_index, nodes_num):
        pos_labels = torch.ones(len(pos_index), dtype=torch.float32)
        neg_index = get_negative_edges(pos_index, len(pos_index), nodes_num)
        neg_labels = torch.zeros(len(neg_index), dtype=torch.float32)
        index = torch.LongTensor(np.concatenate([pos_index, neg_index], axis=0))
        labels = torch.cat([pos_labels, neg_labels], dim=0)
        return index, labels

    idx_train, labels_train = create_labels_and_index(idx_train[:int(len(idx_train)/2)], nodes_num)
    idx_test, labels_test = create_labels_and_index(idx_test[:int(len(idx_test)/2)], nodes_num)


    return adj, features, labels_train, labels_test, idx_train, idx_test

def normalize_adj(mx):
    # Row-normalize sparse matrix
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)



def make_one_hot(labels):
    # Make one-hot labels
    one_hot = torch.zeros(labels.shape[0], 2)
    for i in range(len(labels)):
        if labels[i] == 1:
            one_hot[i][1] = 1
        else:
            one_hot[i][0] = 1
    return one_hot



def compute_AUC(output):
    # Compute AUC value
    num = int(len(output) / 2)
    auc = 0
    for i in range(num):
        if output[i] > output[i + num]:
            auc += 1
        elif output[i] == output[i + num]:
            auc += 0.5
        else:
            auc += 0
    auc /= num
    return auc


def compute_contrastive_loss(features, idx_train, labels_train, temperature=0.5):
    features = F.normalize(features, p=2, dim=1)
    num_edges = idx_train.shape[0]
    pos_idx = idx_train[:num_edges // 2]
    neg_idx = idx_train[num_edges // 2:]
    pos_z_i, pos_z_j = features[pos_idx[:, 0]], features[pos_idx[:, 1]]
    neg_z_i, neg_z_j = features[neg_idx[:, 0]], features[neg_idx[:, 1]]
    pos_sim = F.cosine_similarity(pos_z_i, pos_z_j)
    all_z_i = features[idx_train[:, 0]]
    all_z_j = features[idx_train[:, 1]]
    all_sim_matrix = torch.mm(all_z_i, all_z_j.T) / temperature
    numerator = torch.exp(pos_sim / temperature)
    denominator_pos = torch.logsumexp(all_sim_matrix[:num_edges // 2], dim=1)
    contrastive_loss = -torch.mean(torch.log(numerator / denominator_pos))
    return contrastive_loss


def get_k_hop_neighbors(adj, features, k):
    N = adj.size(0)
    adj_k = adj.clone()
    all_neighbors_features = features.clone().unsqueeze(0)

    for hop in range(2, k + 1):
        adj_k = torch.matmul(adj_k, adj)
        neighbors_k = torch.matmul(adj_k, features)
        all_neighbors_features = torch.cat([all_neighbors_features, neighbors_k.unsqueeze(0)], dim=0)
    aggregated_features = all_neighbors_features.mean(dim=0)
    return aggregated_features


if __name__ == '__main__':
    adj, features, labels_train, labels_test, idx_train, idx_test = load_data("Email-Enron", 256)
    print("dadsd")
    print("dadsdad")

