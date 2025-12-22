import torch
import torch.nn as nn
from layer import GraphAttentionLayer
import torch.nn.functional as F

from utils import get_k_hop_neighbors


class Model(nn.Module):
    def __init__(self, mode, nfeat, nhid, ndimension, dropout, alpha, nheads, adapter_size=8):
        super(Model, self).__init__()
        self.mode = mode
        self.dropout = dropout

        if mode == "finetune":
            # Multiple attention mechanism
            self.attention = GraphAttentionLayer(nfeat, nhid * 4, dropout=dropout, alpha=alpha, concat=True)
            self.dropout = nn.Dropout(dropout)
            self.linear1 = nn.Linear(nhid*4, nhid*2)
            self.adapter = Adapter(nhid * 2, nhid * 2, hidden_dim=adapter_size)
            self.norm = nn.LayerNorm(nhid * 2, eps=1e-5)
            self.linear2 = nn.Linear(nhid * 2, 16)
            # self.linear1 = nn.Linear(nhid * nheads, 64)
            # self.activation = nn.GELU()
            # self.adapter1 = Adapter(64, 64, hidden_dim=adapter_size)
            # self.linear2 = nn.Linear(64, 16)
            # self.dropout2 = nn.Dropout(dropout)
        elif mode == "useTransf":
            # pretrain2---TransformerEncoderLayer
            self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                               range(nheads)]
            for i, attention in enumerate(self.attentions):
                self.add_module('attention_{}'.format(i), attention)
            self.dropout1 = nn.Dropout(dropout)

            # transformer layer
            self.transformer_encoder = TransformerEncoderLayer(
                input_dim=nhid * nheads,
                hidden_dim=64,
                output_dim=16,
                dropout=dropout,
            )
        else:
            # pre-train
            self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                               range(nheads)]
            for i, attention in enumerate(self.attentions):
                self.add_module('attention_{}'.format(i), attention)
            self.dropout1 = nn.Dropout(dropout)

            # transformer layer
            self.transformer_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=nhid * nheads,
                    nhead=2,
                    dim_feedforward=128,
                    dropout=dropout,
                ),
                num_layers=1,
            )

            self.positional_encoding = PositionalEncoding(nfeat, nhid * nheads, dropout)

            self.linear3 = nn.Linear(nhid * nheads, 64)
            self.activation = nn.GELU()
            self.linear4 = nn.Linear(64, 16)

        # The output layer of attention
        self.out_att = GraphAttentionLayer(16, ndimension, dropout=dropout, alpha=alpha, concat=False)
        # Parameter initialization
        self.gamma_1 = nn.Parameter(torch.ones(size=(ndimension, 1)))

    def forward(self, x, adj, edge_start_node, edge_end_node):

        if self.mode == "finetune":
            x = self.dropout(x)
            x = self.attention(x, adj)
            x = self.dropout(x)
            x = self.linear1(x)
            x = self.linear2(self.norm(self.adapter(x)))
        elif self.mode == "useTransf":
            # pretrain2---TransformerEncoderLayer
            x = self.dropout1(x)
            x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
            x = self.dropout1(x)

            x = self.transformer_encoder(x)
        else:
            # pre-train use transform
            x = self.dropout1(x)
            x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
            x = self.dropout1(x)

            positional_embedding = self.positional_encoding(x, adj)

            x = self.transformer_encoder(x + positional_embedding)

            x = self.linear4(self.activation(self.linear3(x)))


        x = F.elu(self.out_att(x, adj))

        # Computing link features
        rmb_train = torch.exp(-F.relu(torch.mm((x[edge_start_node] - x[edge_end_node]).pow(2), self.gamma_1)))
        return torch.squeeze(rmb_train, 1)


    def train_adapter(self):
        for param in self.parameters():     # Freeze parameters
            param.requires_grad = False

        tune_layers = [self.attention, self.adapter, self.norm, self.out_att]
        for layer in tune_layers:
            for param in layer.parameters():
                param.requires_grad = True

        self.gamma_1.requires_grad = True


class TransformerEncoderLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(TransformerEncoderLayer, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(hidden_dim, int(hidden_dim/4))
        self.linear3 = nn.Linear(int(hidden_dim/4), hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim, eps=1e-5)
        self.linear4 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(self.activation(self.linear1(x)))
        x = self.activation(self.linear2(x))
        x = self.linear3(x)
        x = self.norm(x)
        x = self.linear4(x)
        return x



class Adapter(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=8):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, out_dim)
        self.gelu = nn.GELU()

    def forward(self, x):
        z = self.gelu(self.linear1(x))
        z = self.linear3(self.linear2(z))

        return x+z


class PositionalEncoding(nn.Module):
    def __init__(self, nfeat, nhid, dropout, k=2):
        super(PositionalEncoding, self).__init__()
        self.nfeat = nfeat
        self.nhid = nhid
        self.dropout = nn.Dropout(dropout)
        self.k = k

        # Learnable bias terms for positional encoding
        self.phi_dist = nn.Parameter(torch.zeros(1, nhid))

    def forward(self, x, adj):
        k_hop_features = get_k_hop_neighbors(adj, x, self.k)
        positional_embedding = torch.matmul(k_hop_features, self.phi_dist.T)
        return positional_embedding

