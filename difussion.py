import torch
from utils import load_data

def normalize_adj(adj):
    rowsum = adj.sum(1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt

class GraphDiffusion:
    def __init__(self, alpha=0.4, diffusion_steps=10):
        self.alpha = alpha
        self.diffusion_steps = diffusion_steps

    def diffuse_adj(self, adj):
        adj = normalize_adj(adj)
        diffused_adj = (1 - self.alpha) * torch.eye(adj.size(0)).to(adj.device)
        temp = adj.clone()
        for _ in range(self.diffusion_steps):
            diffused_adj += self.alpha * temp
            temp = temp @ adj
        return diffused_adj

    def diffuse_features(self, adj, features):
        adj = normalize_adj(adj)
        diffused_features = features.clone()
        for _ in range(self.diffusion_steps):
            diffused_features = self.alpha * (adj @ diffused_features) + (1 - self.alpha) * features
        return diffused_features

# def main():
#     dataset = "Cora"
#     adj, features, labels_train, labels_val, labels_test, idx_train, idx_val, idx_test = load_data(dataset)
#     diffusion = GraphDiffusion(alpha=0.4, diffusion_steps=10)
#     diffused_adj = diffusion.diffuse_adj(adj)
#     diffused_features = diffusion.diffuse_features(adj, features)
#     print("Diffused adj shape:", diffused_adj.shape)
#     print("Diffused features shape:", diffused_features.shape)



