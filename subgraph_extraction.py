import torch
import scipy.sparse as sp


def extract_subgraph(ind, adj, h=1):
    subgraph_nodes = set([ind[0].item(), ind[1].item()])

    for _ in range(h):
        neighbors = set()
        for node in subgraph_nodes:
            neighbors.update((adj[node] > 0).nonzero(as_tuple=True)[0].tolist())
        subgraph_nodes.update(neighbors)

    subgraph_nodes = list(subgraph_nodes)
    subgraph_adj = adj[subgraph_nodes][:, subgraph_nodes]
    return subgraph_adj, subgraph_nodes


def links2subgraphs(adj, idx_train, idx_val, h=1, features=None):
    def helper(adj, links, h, features):
        subgraphs = []
        node_features = []

        for idx, pair in enumerate(links):
            sub_adj, nodes = extract_subgraph(pair, adj, h=h)
            subgraphs.append(sub_adj)
            node_features.append(features[nodes])

        return subgraphs, node_features

    train_graphs, train_features = helper(adj, idx_train, h, features)
    val_graphs, val_features = helper(adj, idx_val, h, features)

    return train_graphs, val_graphs, train_features, val_features


# Example usage:
# train_graphs, val_graphs, test_graphs, train_features, val_features, test_features, train_labels, val_labels, test_labels = links2subgraphs(
#     adj, idx_train, idx_val, idx_test, h=1, features=features, labels_train=labels_train, labels_val=labels_val,
#     labels_test=labels_test
# )