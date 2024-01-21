import gzip
import shutil
from typing import Optional, Union

import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform

from constants import NM_EIGENVALUES


def extract_compressed_file(file_path: str):
    """
    Extract a compressed file and return the path to the extracted file.
    """
    with gzip.open(file_path, 'rb') as f_in:
        extracted_file_path = file_path[:-3]
        with open(extracted_file_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    return extracted_file_path


# original source: https://github.com/Attornado/protein-representation-learning/
class NodeFeatureFormatter(BaseTransform):
    def __init__(self, feature_columns: Optional[list[str]] = None):
        self.feature_columns = feature_columns if feature_columns is not None else []

    def __call__(self, graph: Union[Data, HeteroData]):
        if not isinstance(graph["coords"], torch.Tensor):
            graph["coords"] = torch.Tensor(graph["coords"][0])
        graph["x"] = graph["coords"]

        for feature_col in self.feature_columns:
            if feature_col in graph:
                graph[feature_col] = torch.Tensor(graph[feature_col])  # convert to tensor
                graph["x"] = torch.cat([graph["x"], graph[feature_col]], dim=-1)  # combine node features

        if NM_EIGENVALUES in graph and len(graph[NM_EIGENVALUES].shape) > 1:
            graph[NM_EIGENVALUES] = graph[NM_EIGENVALUES][0]  # get only one copy not one for each node

        # Handle distance matrix, converting it to numpy array
        if "dist_mat" in graph:
            graph["dist_mat"] = graph["dist_mat"].values

        # Add renamed y column if required
        if "graph_y" in graph:
            graph["y"] = graph["graph_y"]

        return graph


class EdgeFeatureFormatter(BaseTransform):
    def __init__(self, feature_columns: Optional[list[str]] = None):
        self.feature_columns = feature_columns if feature_columns is not None else []

    def __call__(self, graph: Union[Data, HeteroData]):
        if graph.edge_attr is None:
            graph.edge_attr = torch.empty(graph.num_edges, graph.num_edge_features)

        for feature_col in self.feature_columns:
            if feature_col in graph:
                graph[feature_col] = torch.Tensor(graph[feature_col])  # convert to tensor
                graph.edge_attr = torch.cat([graph.edge_attr, graph[feature_col].unsqueeze(1)], dim=-1)  # combine edge
                # features

        return graph
