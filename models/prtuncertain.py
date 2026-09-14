import copy
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=(1024, 512), activation='relu', discrim=False, dropout=-1):
        super(MLP, self).__init__()
        dims = []
        dims.append(input_dim)
        dims.extend(hidden_size)
        dims.append(output_dim)
        self.layers = nn.ModuleList()
        for i in range(len(dims)-1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()

        self.sigmoid = nn.Sigmoid() if discrim else None
        self.dropout = dropout

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i != len(self.layers)-1:
                x = self.activation(x)
                if self.dropout != -1:
                    x = nn.Dropout(min(0.1, self.dropout/3) if i == 1 else self.dropout)(x)
            elif self.sigmoid:
                x = self.sigmoid(x)
        return x



# ============================================================
# Pairwise Relational Transformer
# ============================================================

class RT(nn.Module):
    def __init__(
        self,
        num_layers=3,
        num_heads=4,
        node_dim=64,
        node_hidden_dim=128,
        edge_dim=64,
        edge_hidden_dim_1=128,
        edge_hidden_dim_2=128,
        dropout=0.0
    ):
        super().__init__()

        layer = RTTransformerLayer(
            num_heads,
            node_dim,
            node_hidden_dim,
            edge_dim,
            edge_hidden_dim_1,
            edge_hidden_dim_2,
            dropout
        )

        self.aggregation = 'cat'
        print('[INFO] PRT Agg: {}'.format(self.aggregation))

        self.layers = nn.ModuleList([
            copy.deepcopy(layer)
            for _ in range(num_layers)
        ])

        self.node2edge_mlp = MLP(
            input_dim=node_dim * 2 if self.aggregation == 'cat' else node_dim,
            output_dim=edge_dim,
            hidden_size=(node_hidden_dim,)
        )

    def init_adj(self, num_nodes, batch, device):
        off_diag = np.ones([num_nodes, num_nodes])

        rel_rec = np.array(
            encode_onehot(np.where(off_diag)[1]),
            dtype=np.float32
        )

        rel_send = np.array(
            encode_onehot(np.where(off_diag)[0]),
            dtype=np.float32
        )

        rel_rec = torch.tensor(
            rel_rec,
            dtype=torch.float32,
            device=device
        ).unsqueeze(0).repeat(batch, 1, 1)

        rel_send = torch.tensor(
            rel_send,
            dtype=torch.float32,
            device=device
        ).unsqueeze(0).repeat(batch, 1, 1)

        return rel_rec, rel_send

    def init_edge(self, x, rel_rec, rel_send):
        B, N, _ = x.shape

        if self.aggregation == 'cat':
            receivers = torch.matmul(rel_rec, x)
            senders = torch.matmul(rel_send, x)

            edges = self.node2edge_mlp(
                torch.cat([receivers, senders], dim=-1)
            )

        elif self.aggregation == 'avg':
            H = rel_rec + rel_send
            edges = self.node2edge_mlp(
                torch.matmul(H, x) / 2
            )

        elif self.aggregation == 'sum':
            H = rel_rec + rel_send
            edges = self.node2edge_mlp(
                torch.matmul(H, x)
            )

        else:
            raise NotImplementedError

        return edges.reshape(B, N, N, -1)

    def build_initial_edge(self, node_features):
        B, N, _ = node_features.shape

        rel_rec, rel_send = self.init_adj(
            N,
            B,
            node_features.device
        )

        return self.init_edge(
            node_features,
            rel_rec,
            rel_send
        )

    def reasoning(self, node_features, edge_features):
        for layer in self.layers:
            node_features, edge_features = layer(
                node_features,
                edge_features
            )

        return node_features, edge_features

    def forward(
        self,
        node_features,
        edge_features_=None,
        return_edge=False
    ):
        edge_features = self.build_initial_edge(
            node_features
        )

        node_features, edge_features = self.reasoning(
            node_features,
            edge_features
        )

        if return_edge:
            return node_features, edge_features

        return node_features


class RTNoEdgeInit(nn.Module):
    def __init__(
        self,
        num_layers=3,
        num_heads=4,
        node_dim=64,
        node_hidden_dim=128,
        edge_dim=64,
        edge_hidden_dim_1=128,
        edge_hidden_dim_2=128,
        dropout=0.0
    ):
        super().__init__()

        layer = RTTransformerLayer(
            num_heads,
            node_dim,
            node_hidden_dim,
            edge_dim,
            edge_hidden_dim_1,
            edge_hidden_dim_2,
            dropout
        )

        self.layers = nn.ModuleList([
            copy.deepcopy(layer)
            for _ in range(num_layers)
        ])

    def reasoning(self, node_features, edge_features):
        for layer in self.layers:
            node_features, edge_features = layer(
                node_features,
                edge_features
            )

        return node_features, edge_features

    def forward(
        self,
        node_features,
        edge_features,
        return_edge=False
    ):
        node_features, edge_features = self.reasoning(
            node_features,
            edge_features
        )

        if return_edge:
            return node_features, edge_features

        return node_features


class RTTransformerLayer(nn.Module):
    def __init__(
        self,
        num_heads=4,
        node_dim=64,
        node_hidden_dim=128,
        edge_dim=64,
        edge_hidden_dim_1=128,
        edge_hidden_dim_2=128,
        dropout=0.0
    ):
        super().__init__()

        self.edge_update = True

        self.attention_layer = RTAttentionLayer(
            num_heads,
            node_dim,
            edge_dim
        )

        self.dropout = nn.Dropout(dropout)

        self.linear_net_n = nn.Sequential(
            nn.Linear(node_dim, node_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(node_hidden_dim, node_dim)
        )

        self.norm1_n = nn.LayerNorm(node_dim)
        self.norm2_n = nn.LayerNorm(node_dim)

        if self.edge_update:
            self.linear_net1_e = nn.Sequential(
                nn.Linear(
                    node_dim * 2 + edge_dim * 2,
                    edge_hidden_dim_1
                ),
                nn.ReLU(inplace=True),
                nn.Linear(
                    edge_hidden_dim_1,
                    edge_dim
                )
            )

            self.linear_net2_e = nn.Sequential(
                nn.Linear(
                    edge_dim,
                    edge_hidden_dim_2
                ),
                nn.ReLU(inplace=True),
                nn.Linear(
                    edge_hidden_dim_2,
                    edge_dim
                )
            )

            self.norm1_e = nn.LayerNorm(edge_dim)
            self.norm2_e = nn.LayerNorm(edge_dim)

    def forward(self, node_features, edge_features):
        _, N, _ = node_features.shape

        # ====================================================
        # Node update
        # ====================================================
        attn_out = self.attention_layer(
            node_features,
            edge_features
        )

        node_features = self.norm1_n(
            node_features + self.dropout(attn_out)
        )

        linear_out = self.linear_net_n(
            node_features
        )

        node_features = self.norm2_n(
            node_features + self.dropout(linear_out)
        )

        # ====================================================
        # Edge update
        # ====================================================
        if self.edge_update:
            source_nodes = node_features.unsqueeze(1).repeat(
                1, N, 1, 1
            )

            target_nodes = node_features.unsqueeze(2).repeat(
                1, 1, N, 1
            )

            reversed_edge_features = edge_features.permute(
                0, 2, 1, 3
            )

            edge_input = torch.cat(
                [
                    edge_features,
                    reversed_edge_features,
                    source_nodes,
                    target_nodes
                ],
                dim=-1
            )

            edge_features = self.norm1_e(
                edge_features
                + self.dropout(
                    self.linear_net1_e(edge_input)
                )
            )

            edge_features = self.norm2_e(
                edge_features
                + self.dropout(
                    self.linear_net2_e(edge_features)
                )
            )

        return node_features, edge_features


class RTAttentionLayer(nn.Module):
    def __init__(
        self,
        num_heads,
        node_dim,
        edge_dim
    ):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = node_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.proj_qkv_n = nn.Linear(
            node_dim,
            3 * node_dim
        )

        self.proj_qkv_e = nn.Linear(
            edge_dim,
            3 * node_dim
        )

        self.proj_o = nn.Linear(
            node_dim,
            node_dim
        )

    def forward(self, node_features, edge_features):
        B, N, _ = node_features.shape

        # ====================================================
        # Node Q/K/V
        # [B,N,D] -> [B,H,N,3Dh]
        # ====================================================
        qkv_n = self.proj_qkv_n(
            node_features
        ).reshape(
            B,
            N,
            self.num_heads,
            3 * self.head_dim
        ).permute(
            0, 2, 1, 3
        )

        q_n, k_n, v_n = qkv_n.chunk(
            3,
            dim=-1
        )

        # ====================================================
        # Edge Q/K/V
        # [B,N,N,D] -> [B,H,N,N,3Dh]
        # ====================================================
        qkv_e = self.proj_qkv_e(
            edge_features
        ).reshape(
            B,
            N,
            N,
            self.num_heads,
            3 * self.head_dim
        ).permute(
            0, 3, 1, 2, 4
        )

        q_e, k_e, v_e = qkv_e.chunk(
            3,
            dim=-1
        )

        # ====================================================
        # Relational Q/K
        # ====================================================
        q = (
            q_n.unsqueeze(3)
            + q_e
        )

        k = (
            k_n.unsqueeze(2)
            + k_e
        )

        qk = (
            q * k
        ).sum(
            dim=-1
        )
        # [B,H,N,N]

        # ====================================================
        # Original relational attention
        # No uncertainty
        # ====================================================
        att_dist = F.softmax(
            qk * self.scale,
            dim=-1
        )
        # [B,H,N,N]

        # ====================================================
        # Relational V
        # ====================================================
        v = (
            v_n.unsqueeze(2)
            + v_e
        )
        # [B,H,N,N,Dh]

        new_nodes = torch.sum(
            att_dist.unsqueeze(-1) * v,
            dim=3
        )
        # [B,H,N,Dh]

        new_nodes = new_nodes.permute(
            0, 2, 1, 3
        ).reshape(
            B,
            N,
            self.num_heads * self.head_dim
        )

        return self.proj_o(
            new_nodes
        )