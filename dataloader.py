import pandas as pd
import numpy as np
import networkx as nx
from node2vec import Node2Vec
from ogb.linkproppred import LinkPropPredDataset
from torch_geometric.data import Data
import os
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from torch_geometric.datasets import Amazon, Coauthor, Planetoid


def process_dataset(dataset_name, root_dir="data1", embedding_dim=256, walk_length=30, num_walks=50, p=0.5, q=2):

    output_dir = os.path.join(root_dir, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    if dataset_name == "Facebook":
        edges = pd.read_csv(
            "http://snap.stanford.edu/data/facebook_combined.txt.gz",
            compression="gzip",
            sep=" ",
            names=["start_node", "end_node"],
        )
        nodes_num = pd.concat([edges.iloc[:, 0], edges.iloc[:, 1]]).nunique()
        edges_num = edges.shape[0]

    elif dataset_name == "Computers":
        dataset = Amazon(root=os.path.join(root_dir, "Amazon"), name="computers")
        data = dataset[0]
        nodes_num = data.num_nodes
        edges_num = data.num_edges
        edges_index = data.edge_index
        edges = edges_index.t().cpu().numpy()
        edges = pd.DataFrame(edges, columns=["start_node", "end_node"])

    elif dataset_name == "Photos":
        dataset = Amazon(root=os.path.join(root_dir, "Amazon"), name="photo")
        data = dataset[0]
        nodes_num = data.num_nodes
        edges_num = data.num_edges
        edges_index = data.edge_index
        edges = edges_index.t().cpu().numpy()
        edges = pd.DataFrame(edges, columns=["start_node", "end_node"])

    elif dataset_name == "ChCh-Miner":
        df = pd.read_csv('data/ChCh-Miner_durgbank-chem-chem.tsv', sep='\t', header=None,
                         names=['start_node', 'end_node'])
        all_nodes = pd.concat([df["start_node"], df["end_node"]]).unique()
        node_to_id = {node: idx for idx, node in enumerate(all_nodes)}
        df['start_node'] = df['start_node'].map(node_to_id)
        df['end_node'] = df['end_node'].map(node_to_id)
        nodes_num = all_nodes.shape[0]
        edges_num = df.shape[0]
        edges = df

    elif dataset_name == "Coauthor-CS":
        dataset = Coauthor(root=os.path.join(root_dir, "Coauthor"), name="CS")
        data = dataset[0]
        nodes_num = data.num_nodes
        edges_num = data.num_edges
        edges_index = data.edge_index
        edges = edges_index.t().cpu().numpy()
        edges = pd.DataFrame(edges, columns=["start_node", "end_node"])

    elif dataset_name == "Email-Enron":
        edges = pd.read_csv("data1/Email-Enron.txt", sep='\t', header=None, names=["start_node", "end_node"])
        edges['ordered_start'] = edges[['start_node', 'end_node']].min(axis=1)
        edges['ordered_end'] = edges[['start_node', 'end_node']].max(axis=1)
        undirected_edges = edges[['ordered_start', 'ordered_end']].drop_duplicates()
        undirected_edges.columns = ['start_node', 'end_node']
        edges = undirected_edges

        node_degrees = edges["start_node"].value_counts() + edges["end_node"].value_counts()
        top_nodes = node_degrees.nlargest(15642).index
        edges = edges[
            edges["start_node"].isin(top_nodes) & edges["end_node"].isin(top_nodes)
        ]
        node_to_new_id = {node: idx for idx, node in enumerate(top_nodes)}
        edges.loc[:, 'start_node'] = edges['start_node'].map(node_to_new_id)
        edges.loc[:, 'end_node'] = edges['end_node'].map(node_to_new_id)
        nodes_num = len(top_nodes)
        edges_num = edges.shape[0]

    elif dataset_name == "ogbl-ddi":
        dataset = LinkPropPredDataset("ogbl-ddi")
        graph = dataset[0]
        edge_index = graph["edge_index"]
        nodes_num = graph["num_nodes"]
        edges = edge_index.T
        edges = pd.DataFrame(edges, columns=["start_node", "end_node"])

    elif dataset_name == "Cora":
        dataset = Planetoid(root=os.path.join(root_dir, "Cora"), name="Cora")
        data = dataset[0]
        nodes_num = data.num_nodes
        edges_num = data.num_edges
        edges_index = data.edge_index
        edges = edges_index.t().cpu().numpy()
        edges = pd.DataFrame(edges, columns=["start_node", "end_node"])

    elif dataset_name == "Citeseer":
        dataset = Planetoid(root=os.path.join(root_dir, "Citeseer"), name="Citeseer")
        data = dataset[0]
        nodes_num = data.num_nodes
        edges_num = data.num_edges
        edges_index = data.edge_index
        edges = edges_index.t().cpu().numpy()
        edges = pd.DataFrame(edges, columns=["start_node", "end_node"])

    else:
        raise ValueError("Unsupported dataset name.")

    graph = nx.from_pandas_edgelist(edges, source="start_node", target="end_node", create_using=nx.Graph())
    node2vec = Node2Vec(
        graph,
        dimensions=embedding_dim,
        walk_length=walk_length,
        num_walks=num_walks,
        workers=4,
        p=p,
        q=q,
    )
    model = node2vec.fit(window=10, min_count=1, batch_words=32)
    features = np.zeros((nodes_num, embedding_dim))
    for node in graph.nodes():
        features[node] = model.wv[str(node)]

    edges_shuffled = edges.sample(frac=1).reset_index(drop=True)
    all_edges = edges_shuffled.values
    edge_file_path = os.path.join(output_dir, f"{dataset_name}_edges_AD.txt")
    np.savetxt(edge_file_path, all_edges, fmt="%d", delimiter=" ")

    train_edges, temp_edges = train_test_split(all_edges, test_size=0.2, random_state=42)
    test_edges, val_edges = train_test_split(temp_edges, test_size=0.5, random_state=42)
    train_file_path = os.path.join(output_dir, f"{dataset_name}_train_edges.pt")
    test_file_path = os.path.join(output_dir, f"{dataset_name}_test_edges.pt")
    val_file_path = os.path.join(output_dir, f"{dataset_name}_validation_edges.pt")
    torch.save(torch.tensor(train_edges, dtype=torch.long), train_file_path)
    torch.save(torch.tensor(test_edges, dtype=torch.long), test_file_path)
    torch.save(torch.tensor(val_edges, dtype=torch.long), val_file_path)

    features_file_path = os.path.join(output_dir, f"{dataset_name}_features.npy")
    np.save(features_file_path, features)

    print(f"Dataset {dataset_name} processed and saved to {output_dir}")


# # process_dataset("Coauthor-CS")
# def process_dataset(dataset_name, root_dir="data1"):
#
#     output_dir = os.path.join(root_dir, dataset_name)
#     os.makedirs(output_dir, exist_ok=True)
#
#     dataset = Coauthor(root=os.path.join(root_dir, "Coauthor"), name="CS")
#     data = dataset[0]
#     nodes_num = data.num_nodes
#     edges_num = data.num_edges
#     edges_index = data.edge_index
#     edges = edges_index.t().cpu().numpy()
#     edges = pd.DataFrame(edges, columns=["start_node", "end_node"])
#
#     features = data.x.cpu().numpy()
#     pca = PCA(n_components=256)
#     features = pca.fit_transform(features)
#
#     edges_shuffled = edges.sample(frac=1).reset_index(drop=True)
#     all_edges = edges_shuffled.values
#     edge_file_path = os.path.join(output_dir, f"{dataset_name}_edges_AD.txt")
#     np.savetxt(edge_file_path, all_edges, fmt="%d", delimiter=" ")
#
#     train_edges, temp_edges = train_test_split(all_edges, test_size=0.2, random_state=42)
#     test_edges, val_edges = train_test_split(temp_edges, test_size=0.5, random_state=42)
#     train_file_path = os.path.join(output_dir, f"{dataset_name}_train_edges.pt")
#     test_file_path = os.path.join(output_dir, f"{dataset_name}_test_edges.pt")
#     val_file_path = os.path.join(output_dir, f"{dataset_name}_validation_edges.pt")
#     torch.save(torch.tensor(train_edges, dtype=torch.long), train_file_path)
#     torch.save(torch.tensor(test_edges, dtype=torch.long), test_file_path)
#     torch.save(torch.tensor(val_edges, dtype=torch.long), val_file_path)
#
#     features_file_path = os.path.join(output_dir, f"{dataset_name}_features.npy")
#     np.save(features_file_path, features)
#
#     print(f"Dataset {dataset_name} processed and saved to {output_dir}")


# # process_dataset("ogbl-collab")
# def process_dataset(dataset_name, root_dir="data1", embedding_dim=256, walk_length=100, num_walks=20, p=1, q=1.5):
#
#     output_dir = os.path.join(root_dir, dataset_name)
#     os.makedirs(output_dir, exist_ok=True)
#
#     dataset = LinkPropPredDataset(name="ogbl-collab")
#     graph = dataset[0]
#     edge_index = graph["edge_index"]
#     nodes_num = graph["num_nodes"]
#     edges = edge_index.T
#     edges = pd.DataFrame(edges, columns=["start_node", "end_node"])
#
#     graph = nx.from_pandas_edgelist(edges, source="start_node", target="end_node", create_using=nx.Graph())
#     node_batches = np.array_split(list(graph.nodes), 10)
#     features = np.zeros((nodes_num, embedding_dim))
#     for batch in node_batches:
#         subgraph = graph.subgraph(batch)
#         node2vec = Node2Vec(
#             subgraph,
#             dimensions=embedding_dim,
#             walk_length=walk_length,
#             num_walks=num_walks,
#             workers=4,
#             p=p,
#             q=q,
#         )
#         model = node2vec.fit(window=10, min_count=1, batch_words=32)
#
#         for node in batch:
#             features[node] = model.wv[str(node)]
#
#     edges_shuffled = edges.sample(frac=1).reset_index(drop=True)
#     all_edges = edges_shuffled.values
#     edge_file_path = os.path.join(output_dir, f"{dataset_name}_edges_AD.txt")
#     np.savetxt(edge_file_path, all_edges, fmt="%d", delimiter=" ")
#
#     train_edges, temp_edges = train_test_split(all_edges, test_size=0.2, random_state=42)
#     test_edges, val_edges = train_test_split(temp_edges, test_size=0.5, random_state=42)
#     train_file_path = os.path.join(output_dir, f"{dataset_name}_train_edges.pt")
#     test_file_path = os.path.join(output_dir, f"{dataset_name}_test_edges.pt")
#     val_file_path = os.path.join(output_dir, f"{dataset_name}_validation_edges.pt")
#     torch.save(torch.tensor(train_edges, dtype=torch.long), train_file_path)
#     torch.save(torch.tensor(test_edges, dtype=torch.long), test_file_path)
#     torch.save(torch.tensor(val_edges, dtype=torch.long), val_file_path)
#
#     features_file_path = os.path.join(output_dir, f"{dataset_name}_features.npy")
#     np.save(features_file_path, features)
#
#     print(f"Dataset {dataset_name} processed and saved to {output_dir}")



# get data
# process_dataset("Facebook")  # process Facebook
# process_dataset("Computers")  # process Amazon-Computers
# process_dataset("Photos")  # process Amazon-Photos
# process_dataset("ChCh-Miner")  # process ChCh-Miner
# process_dataset("Coauthor-CS")  # process CS
# process_dataset("ogbl-collab")  # process ogbl-collab
# process_dataset("Email-Enron")  # process Email-Enron
# process_dataset("ogbl-ddi")  # process ogbl-ddi
process_dataset("Cora")  # process Cora
process_dataset("Citeseer")  # process Citeseer













# if nodes_num > 10000 or edges_num > 200000:
#     dimensions = 512
#     walk_length = 50
#     num_walks = 100
# else:
#     dimensions = 256
#     walk_length = 30
#     num_walks = 50








# # Get Facebook
# edges = pd.read_csv(
#     #"data/facebook_combined.txt.gz",
#     "http://snap.stanford.edu/data/facebook_combined.txt.gz",
#     compression="gzip",
#     sep=" ",
#     names=["start_node", "end_node"],
# )
# nodes_num = pd.concat([edges.iloc[:, 0], edges.iloc[:, 1]]).nunique()
# edges_num = edges.shape[0]
#
# # # adj matrix
# # adj = np.zeros((nodes_num, nodes_num), dtype=int)
# # for _, row in edges.iterrows():
# #     start_node, end_node = row["start_node"], row["end_node"]
# #     adj[start_node, end_node] = 1
# #     adj[end_node, start_node] = 1
#
# # use node2vec get features
# graph = nx.from_pandas_edgelist(edges, source="start_node", target="end_node", create_using=nx.Graph())
# node2vec = Node2Vec(
#     graph,
#     dimensions=256,
#     walk_length=30,
#     num_walks=50,
#     workers=4,
#     p=0.5,
#     q=2
# )
# model = node2vec.fit(window=10, min_count=1, batch_words=4)
# features = np.zeros((nodes_num, 256))
# for node in graph.nodes():
#     features[node] = model.wv[str(node)]








# # Get Amazon Computer
# graph = nx.from_pandas_edgelist(edges, source="start_node", target="end_node", create_using=nx.Graph())
# node2vec = Node2Vec(
#     graph,
#     dimensions=256,
#     walk_length=30,
#     num_walks=50,
#     workers=4,
#     p=0.5,
#     q=2
# )
# model = node2vec.fit(window=10, min_count=1, batch_words=4)
# features = np.zeros((nodes_num, 256))
# for node in graph.nodes():
#     features[node] = model.wv[str(node)]






