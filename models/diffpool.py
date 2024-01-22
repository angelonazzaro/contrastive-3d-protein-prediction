# This implementation is based on the one from the repository:
# https://github.com/diningphil/gnn-comparison, all rights reserved to authors and contributors.
# Copyright (C)  2020  University of Pisa
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
from math import ceil
from typing import Optional, Any
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn.dense import DenseSAGEConv, dense_diff_pool
from torch_geometric.utils import to_dense_batch, to_dense_adj
# from torch_geometric.transforms import ToDense
from torchmetrics.functional import accuracy, precision, recall, f1_score


NUM_SAGE_LAYERS = 3


class SAGEConvolutions(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 lin=True):
        super().__init__()

        self.conv1 = DenseSAGEConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = DenseSAGEConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.conv3 = DenseSAGEConv(hidden_channels, out_channels)

        if lin is True:
            self.lin = nn.Linear((NUM_SAGE_LAYERS - 1) * hidden_channels + out_channels, out_channels)
        else:
            # GNN's intermediate representation is given by the concatenation of SAGE layers
            self.lin = None

    def bn(self, i, x):
        batch_size, num_nodes, num_channels = x.size()

        x = x.view(-1, num_channels)
        x = getattr(self, 'bn{}'.format(i))(x)
        x = x.view(batch_size, num_nodes, num_channels)
        return x

    def forward(self, x, adj, mask=None):
        x0 = x
        x1 = self.bn(1, F.relu(self.conv1(x0, adj, mask)))
        x2 = self.bn(2, F.relu(self.conv2(x1, adj, mask)))
        x3 = self.conv3(x2, adj, mask)

        x = torch.cat([x1, x2, x3], dim=-1)

        # This is used by GNN_pool
        if self.lin is not None:
            x = self.lin(x)

        return x


class DiffPoolLayer(nn.Module):
    """
    Applies GraphSAGE convolutions and then performs pooling
    """
    def __init__(self, dim_input, dim_hidden, dim_embedding, no_new_clusters):
        """
        :param dim_input:
        :param dim_hidden: embedding size of first 2 SAGE convolutions
        :param dim_embedding: embedding size of 3rd SAGE convolutions (eq. 5, dim of Z)
        :param no_new_clusters: number of clusters after pooling (eq. 6, dim of S)
        """
        super().__init__()
        self.gnn_pool = SAGEConvolutions(dim_input, dim_hidden, no_new_clusters)
        self.gnn_embed = SAGEConvolutions(dim_input, dim_hidden, dim_embedding, lin=False)

    def forward(self, x, adj, mask=None):
        s = self.gnn_pool(x, adj, mask)
        x = self.gnn_embed(x, adj, mask)

        x, adj, l, e = dense_diff_pool(x, adj, s, mask)
        return x, adj, l, e


class DiffPoolMulticlassClassificationLoss(MulticlassClassificationLoss):
    """
    DiffPool - No Link Prediction Loss, that one is outputed by the DiffPool layer
    """

    def forward(self, targets: torch.Tensor, *outputs: torch.Tensor) -> torch.Tensor:
        if len(outputs) == 1:
            outputs = outputs[0]
        preds, lp_loss, ent_loss = outputs

        if targets.dim() > 1 and targets.size(1) == 1:
            targets = targets.squeeze(1)

        loss = self._loss(preds, targets)
        return loss + lp_loss + ent_loss


class DiffPool(GraphClassifier):
    """
    Computes multiple DiffPoolLayers
    """
    def __init__(self, dim_features, dim_target, config):
        super().__init__(dim_features, dim_target, config)

        self.max_num_nodes = config['max_num_nodes']
        num_diffpool_layers = config['num_layers']
        gnn_dim_hidden = config['gnn_dim_hidden']  # embedding size of first 2 SAGE convolutions
        dim_embedding = config['dim_embedding']  # embedding size of 3rd SAGE convolutions (eq. 5, dim of Z)
        dim_embedding_MLP = config['dim_embedding_MLP']  # hidden neurons of last 2 MLP layers

        self.num_diffpool_layers = num_diffpool_layers

        # Reproduce paper choice about coarse factor
        coarse_factor = 0.1 if num_diffpool_layers == 1 else 0.25

        gnn_dim_input = dim_features
        no_new_clusters = ceil(coarse_factor * self.max_num_nodes)
        gnn_embed_dim_output = (NUM_SAGE_LAYERS - 1) * gnn_dim_hidden + dim_embedding

        layers = []
        for i in range(num_diffpool_layers):
            diffpool_layer = DiffPoolLayer(gnn_dim_input, gnn_dim_hidden, dim_embedding, no_new_clusters)
            layers.append(diffpool_layer)

            # Update embedding sizes
            gnn_dim_input = gnn_embed_dim_output
            no_new_clusters = ceil(no_new_clusters * coarse_factor)

        self.diffpool_layers = nn.ModuleList(layers)

        # After DiffPool layers, apply again layers of GraphSAGE convolutions
        self.final_embed = SAGEConvolutions(gnn_embed_dim_output, gnn_dim_hidden, dim_embedding, lin=False)
        final_embed_dim_output = gnn_embed_dim_output * (num_diffpool_layers + 1)

        self.lin1 = nn.Linear(final_embed_dim_output, dim_embedding_MLP)
        self.lin2 = nn.Linear(dim_embedding_MLP, dim_target)

    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor):
        x, mask = to_dense_batch(x, batch=batch)
        adj = to_dense_adj(edge_index, batch=batch)
        # data = ToDense(data.num_nodes)(data)
        # TODO describe mask shape and how batching works

        # adj, mask, x = data.adj, data.mask, data.x
        x_all, l_total, e_total = [], 0, 0

        for i in range(self.num_diffpool_layers):
            if i != 0:
                mask = None

            x, adj, l, e = self.diffpool_layers[i](x, adj, mask)  # x has shape (batch, MAX_no_nodes, feature_size)
            x_all.append(torch.max(x, dim=1)[0])

            l_total += l
            e_total += e

        x = self.final_embed(x, adj)
        x_all.append(torch.max(x, dim=1)[0])

        x = torch.cat(x_all, dim=1)  # shape (batch, feature_size x diffpool layers)

        return x, l_total, e_total

    def forward(self, x, edge_index, batch, apply_first_linear: bool = True, apply_second_linear: bool = True):

        """x, mask = to_dense_batch(x, batch=batch)
        adj = to_dense_adj(edge_index, batch=batch)
        # data = ToDense(data.num_nodes)(data)
        # TODO describe mask shape and how batching works

        # adj, mask, x = data.adj, data.mask, data.x
        x_all, l_total, e_total = [], 0, 0

        for i in range(self.num_diffpool_layers):
            if i != 0:
                mask = None

            x, adj, l, e = self.diffpool_layers[i](x, adj, mask)  # x has shape (batch, MAX_no_nodes, feature_size)
            x_all.append(torch.max(x, dim=1)[0])

            l_total += l
            e_total += e

        x = self.final_embed(x, adj)
        x_all.append(torch.max(x, dim=1)[0])

        x = torch.cat(x_all, dim=1)  # shape (batch, feature_size x diffpool layers)"""
        x, l_total, e_total = self.get_embeddings(x=x, edge_index=edge_index, batch=batch)

        if apply_first_linear or apply_second_linear:
            x = F.relu(self.lin1(x))
        if apply_second_linear:
            x = self.lin2(x)
        return x, l_total, e_total

    def test(self,
             y,
             y_hat: Optional[Any] = None,
             x: Optional[torch.Tensor] = None,
             edge_index: Optional[torch.Tensor] = None,
             batch_index: torch.Tensor = None,
             criterion: ClassificationLoss = DiffPoolMulticlassClassificationLoss(),
             top_k: Optional[int] = None,
             *args, **kwargs) -> (float, Optional[float], float, float, float, float):
        """
        This function takes in a graph, and returns the loss, accuracy, top-k accuracy, precision, recall, and F1-score.

        :param x: torch.Tensor = The node features
        :type x: torch.Tensor
        :param edge_index: The edge indices of the graph
        :type edge_index: torch.Tensor
        :param y: The target labels
        :param batch_index: The batch index of the nodes
        :type batch_index: torch.Tensor
        :param criterion: The loss function to use
        :type criterion: Callable
        :param top_k: k for computing top_k accuracy, *args, **kwargs
        :type top_k: Optional[int]
        :return: The loss, accuracy, top-k accuracy, precision, recall, and F1-score.
        """

        # Get the number of classes
        n_classes = self.dim_target

        # Get predictions
        if y_hat is None:
            y_hat = self(x, edge_index, batch_index, *args, **kwargs)

        # Compute loss
        loss = self.loss(y_hat=y_hat, y=y, criterion=criterion)

        # Remove additional loss terms
        if isinstance(y_hat, tuple):
            y_hat = y_hat[0]

        # Compute the metrics
        acc = accuracy(preds=y_hat, target=y, task='multiclass', average="macro", num_classes=n_classes)
        if top_k is not None:
            top_k_acc = float(accuracy(preds=y_hat, target=y, task='multiclass', num_classes=n_classes, top_k=top_k,
                                       average="macro"))
        else:
            top_k_acc = None
        prec = precision(preds=y_hat, target=y, task='multiclass', num_classes=n_classes, average="macro")
        rec = recall(preds=y_hat, target=y, task='multiclass', num_classes=n_classes, average="macro")
        f1 = f1_score(preds=y_hat, target=y, task='multiclass', num_classes=n_classes, average="macro")

        return float(loss), float(acc), top_k_acc, prec, rec, f1


class DiffPoolEmbedding(DiffPool):
    """Custom DiffPool class to change default parameters in DiffPool forward()"""
    def __init__(self, dim_features, dim_target, config):
        super().__init__(dim_features, dim_target, config)

    def forward(self, x, edge_index, batch, apply_first_linear: bool = True, apply_second_linear: bool = False):
        return super().forward(x, edge_index, batch, apply_first_linear=apply_first_linear,
                               apply_second_linear=apply_second_linear)
