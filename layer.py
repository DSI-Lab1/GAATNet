import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        # Weights and Bias
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)


    # def forward(self, h, adj):
    #     # Build attention layer
    #     Wh = torch.mm(h, self.W)
    #     a_input = self._prepare_attentional_mechanism_input(Wh)
    #     e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
    #
    #     zero_vec = -9e15*torch.ones_like(e)
    #     attention = torch.where(adj > 0, e, zero_vec)
    #     attention = F.softmax(attention, dim=1)
    #     attention = F.dropout(attention, self.dropout, training=self.training)
    #     h_prime = torch.matmul(attention, Wh)
    #
    #     if self.concat:
    #         return F.elu(h_prime)
    #     else:
    #         return h_prime

    def forward(self, h, adj):
        # Build attention layer
        Wh = torch.mm(h, self.W)

        a_input, edge_indices = self._prepare_attentional_mechanism_input(Wh, adj)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(1))

        attention = torch.zeros_like(adj, dtype=torch.float32)
        attention[edge_indices[:, 0], edge_indices[:, 1]] = e
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


    # def _prepare_attentional_mechanism_input(self, Wh):
    #     N = Wh.size()[0]
    #
    #     Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
    #     Wh_repeated_alternating = Wh.repeat(N, 1)
    #
    #     all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
    #
    #     return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def _prepare_attentional_mechanism_input(self, Wh, adj):
        edge_indices = adj.nonzero(as_tuple=False)

        Wh_i = Wh[edge_indices[:, 0]]
        Wh_j = Wh[edge_indices[:, 1]]

        all_combinations_matrix = torch.cat([Wh_i, Wh_j], dim=1)
        return all_combinations_matrix, edge_indices

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'



