from __future__ import annotations

import gzip
import shutil
from typing import Iterable, Union
from typing import Optional

import networkx as nx
import numpy as np
import torch
from graphein.protein import add_edge, compute_distmat, filter_distmat
from graphein.protein.utils import filter_dataframe
from loguru import logger as log
from sklearn.neighbors import NearestNeighbors
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


INFINITE_DIST = 10_000.0  # np.inf leads to errors in some cases


def add_k_nn_edges(
        G: nx.Graph,
        long_interaction_threshold: int = 0,
        k: int = 5,
        exclude_edges: Iterable[str] = (),
        exclude_self_loops: bool = True,
        kind_name: str = "knn",
):
    """
    Adds edges to nodes based on K nearest neighbours. Long interaction
    threshold is used to specify minimum separation in sequence to add an edge
    between networkx nodes within the distance threshold

    :param G: Protein Structure graph to add distance edges to
    :type G: nx.Graph
    :param long_interaction_threshold: minimum distance in sequence for two
        nodes to be connected
    :type long_interaction_threshold: int
    :param k: Number of neighbors for each sample.
    :type k: int
    :param exclude_edges: Types of edges to exclude. Supported values are
        `inter` and `intra`.
        - `inter` removes inter-connections between nodes of the same chain.
        - `intra` removes intra-connections between nodes of different chains.
    :type exclude_edges: Iterable[str].
    :param exclude_self_loops: Whether to mark each sample as the first nearest neighbor to itself.
    :type exclude_self_loops: Union[bool, str]
    :param kind_name: Name for kind of edges in networkx graph.
    :type kind_name: str
    :return: Graph with knn-based edges added
    :rtype: nx.Graph
    """
    # Prepare dataframe
    pdb_df = filter_dataframe(
        G.graph["pdb_df"], "node_id", list(G.nodes()), True
    )
    if (
            pdb_df["x_coord"].isna().sum()
            or pdb_df["y_coord"].isna().sum()
            or pdb_df["z_coord"].isna().sum()
    ):
        raise ValueError("Coordinates contain a NaN value.")

    # Construct distance matrix
    dist_mat = compute_distmat(pdb_df)

    # Filter edges
    dist_mat = filter_distmat(pdb_df, dist_mat, exclude_edges)

    # Add self-loops if specified
    if not exclude_self_loops:
        k -= 1
        for n1, n2 in zip(G.nodes(), G.nodes()):
            add_edge(G, n1, n2, kind_name)

    # Reduce k if number of nodes is less (to avoid sklearn error)
    # Note: - 1 because self-loops are not included
    if G.number_of_nodes() - 1 < k:
        k = G.number_of_nodes() - 1

    if k == 0:
        return

    # Run k-NN search
    neigh = NearestNeighbors(n_neighbors=k, metric="precomputed")
    neigh.fit(dist_mat)
    nn = neigh.kneighbors_graph()

    # Create iterable of node indices
    outgoing = np.repeat(np.array(range(len(G.graph["pdb_df"]))), k)
    incoming = nn.indices
    interacting_nodes = list(zip(outgoing, incoming))
    log.info(f"Found: {len(interacting_nodes)} KNN edges")
    for a1, a2 in interacting_nodes:
        if dist_mat.loc[a1, a2] == INFINITE_DIST:
            continue

        # Get nodes IDs from indices
        n1 = G.graph["pdb_df"].iloc[a1]["node_id"]
        n2 = G.graph["pdb_df"].iloc[a2]["node_id"]

        # Get chains
        n1_chain = G.graph["pdb_df"].iloc[a1]["chain_id"]
        n2_chain = G.graph["pdb_df"].iloc[a2]["chain_id"]

        # Get sequence position
        n1_position = G.graph["pdb_df"].iloc[a1]["residue_number"]
        n2_position = G.graph["pdb_df"].iloc[a2]["residue_number"]

        # Check residues are not on same chain
        condition_1 = n1_chain != n2_chain
        # Check residues are separated by long_interaction_threshold
        condition_2 = (
                abs(n1_position - n2_position) > long_interaction_threshold
        )

        # If not on same chain add edge or
        # If on same chain and separation is sufficient add edge
        if condition_1 or condition_2:
            add_edge(G, n1, n2, kind_name)

# class EdgeFeatureFormatter(BaseTransform):
#     def __init__(self):
#         super().__init__()
#
#     def __call__(self, graph: Union[Data, HeteroData]):
#         return self.__calculate_chemical_information(graph)
#
#     def __calculate_chemical_information(self, graph: Data) -> Data:
#         if graph.edge_attr is None:
#             graph.edge_attr = torch.empty(graph.num_edges, graph.num_edge_features)
#
#         torsion_angles = self.__calculate_torsion_angles(graph)
#         graph.edge_attr = torch.cat([graph.edge_attr, torsion_angles.view(-1, 1)], dim=-1)
#
#         covalent_bonds = self.__identify_covalent_bonds(graph)
#         graph.edge_attr = torch.cat([graph.edge_attr, covalent_bonds.view(-1, 1)], dim=-1)
#
#         # repeat_patterns = self.__identify_repeat_patterns(graph)
#         # graph.edge_attr = torch.cat([graph.edge_attr, repeat_patterns.view(-1, 1)], dim=-1)
#
#         return graph
#
#     def __calculate_torsion_angles(self, graph: Data) -> torch.Tensor:
#         positions = graph.coords
#
#         angles = []
#         for edge in graph.edge_index.t().contiguous().tolist():
#             atom1, atom2, atom3, atom4 = edge
#
#             v1 = positions[atom2] - positions[atom1]
#             v2 = positions[atom3] - positions[atom2]
#             v3 = positions[atom4] - positions[atom3]
#
#             angle = self.__calculate_dihedral_angle(v1, v2, v3)
#             angles.append(angle.item())
#
#         return torch.tensor(angles, dtype=torch.float32)
#
#     def __calculate_dihedral_angle(self, v1, v2, v3):
#         cross_product1 = torch.cross(v1, v2)
#         cross_product2 = torch.cross(v2, v3)
#
#         matmul_product = torch.dot(cross_product1, cross_product2)
#         norm_product = torch.norm(cross_product1) * torch.norm(cross_product2)
#
#         return torch.atan2(norm_product, dot_product)
#
#     def __identify_covalent_bonds(self, graph: Data) -> torch.Tensor:
#         bond_types = graph.edge_attr
#
#         covalent_bonds = (bond_types == 1)
#         return covalent_bonds
#
#     # def __identify_repeat_patterns(self, graph: Data) -> torch.Tensor:
#     #     return torch.zeros(graph.edge_attr.size(0), dtype=torch.float32)
#     #     return repeat_patterns
