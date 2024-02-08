import gzip
import shutil
from typing import Optional, Union

import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform

from dataset.constants import NM_EIGENVALUES


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
        super().__init__()
        self.feature_columns = feature_columns if feature_columns is not None else []

    def __call__(self, graph: Union[Data, HeteroData]):
        if not isinstance(graph["coords"], torch.Tensor):
            graph["coords"] = torch.Tensor(graph["coords"][0])
        graph["x"] = graph["coords"]

        for feature_col in self.feature_columns:
            graph[feature_col] = torch.Tensor(graph[feature_col])  # convert to tensor
            graph["x"] = torch.cat([graph["x"], graph[feature_col]], dim=-1)  # combine node features

        graph["x"] = graph["x"].float()

        if NM_EIGENVALUES in graph and len(graph[NM_EIGENVALUES].shape) > 1:
            graph[NM_EIGENVALUES] = graph[NM_EIGENVALUES][0]  # get only one copy not one for each node

        # Handle distance matrix, converting it to numpy array
        if "dist_mat" in graph:
            graph["dist_mat"] = graph["dist_mat"].values

        # Add renamed y column if required
        if "graph_y" in graph:
            graph["y"] = graph["graph_y"]

        # graph = self.calculate_chemical_information(graph)

        return graph


class EdgeFeatureFormatter(BaseTransform):
    def __init__(self):
        super().__init__()

    def __call__(self, graph: Union[Data, HeteroData]):
        return self.__calculate_chemical_information(graph)

    def __calculate_chemical_information(self, graph: Data) -> Data:
        if graph.edge_attr is None: 
            graph.edge_attr = torch.empty(graph.num_edges, graph.num_edge_features)

        torsion_angles = self.__calculate_torsion_angles(graph)
        graph.edge_attr = torch.cat([graph.edge_attr, torsion_angles.view(-1, 1)], dim=-1)

        covalent_bonds = self.__identify_covalent_bonds(graph)
        graph.edge_attr = torch.cat([graph.edge_attr, covalent_bonds.view(-1, 1)], dim=-1)

        # repeat_patterns = self.__identify_repeat_patterns(graph)
        # graph.edge_attr = torch.cat([graph.edge_attr, repeat_patterns.view(-1, 1)], dim=-1)

        return graph

    def __calculate_torsion_angles(self, graph: Data) -> torch.Tensor:
        positions = graph.coords

        angles = []
        for edge in graph.edge_index.t().contiguous().tolist():
            atom1, atom2, atom3, atom4 = edge

            v1 = positions[atom2] - positions[atom1]
            v2 = positions[atom3] - positions[atom2]
            v3 = positions[atom4] - positions[atom3]

            angle = self.__calculate_dihedral_angle(v1, v2, v3)
            angles.append(angle.item())

        return torch.tensor(angles, dtype=torch.float32)

    def __calculate_dihedral_angle(self, v1, v2, v3):
        cross_product1 = torch.cross(v1, v2)
        cross_product2 = torch.cross(v2, v3)

        matmul_product = torch.dot(cross_product1, cross_product2)
        norm_product = torch.norm(cross_product1) * torch.norm(cross_product2)

        return torch.atan2(norm_product, dot_product)

    def __identify_covalent_bonds(self, graph: Data) -> torch.Tensor:
        bond_types = graph.edge_attr

        covalent_bonds = (bond_types == 1)
        return covalent_bonds

    # def __identify_repeat_patterns(self, graph: Data) -> torch.Tensor:
    #     return torch.zeros(graph.edge_attr.size(0), dtype=torch.float32)
    #     return repeat_patterns
