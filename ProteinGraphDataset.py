import torch
from torch.utils.data import Dataset, DataLoader
import os
import json
import networkx as nx

class ProteinGraphDataset(Dataset):
    def __init__(self, dataset_dir, response_format='json'):
        self.dataset_dir = dataset_dir
        self.response_format = response_format

        # Collect a list of paths for JSON files in the dataset directory
        # Filter the files based on the '_graph.json' suffix and root directory
        self.graph_files = [
            os.path.join(root, file)
            for root, dirs, files in os.walk(dataset_dir)
            for file in files
            if file.endswith('_graph.json')
        ]

    def __len__(self):
        # Return the length of the dataset
        return len(self.graph_files)

    def __getitem__(self, idx):
        # Retrieve the graph file path for a given index
        graph_file = self.graph_files[idx]
        protein_name = graph_file.split('_')[0]

        # Load graph data from the JSON file
        graph_json_path = os.path.join(self.dataset_dir, protein_name, graph_file)

        try:
            # Attempt to open and load the JSON file
            with open(graph_json_path, 'r') as json_file:
                graph_data = json.load(json_file)

            # Create a networkx graph from the loaded data
            graph = nx.node_link_graph(graph_data)

            # Convert the graph's adjacency matrix to a PyTorch tensor
            adjacency_matrix = torch.Tensor(nx.adjacency_matrix(graph).todense())

            return adjacency_matrix
        except FileNotFoundError as e:
            # Handle the case where the file is not found with a custom message
            raise FileNotFoundError(f"File not found: {graph_json_path}") from e
        except Exception as e:
            # Handle other exceptions with a custom message
            raise Exception(f"Error while reading file {graph_json_path}: {e}") from e
