import torch
from torch.utils.data import Dataset
import os
import json
import networkx as nx


class ProteinGraphDataset(Dataset):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir

        # Collect a list of paths for JSON files in the dataset directory
        # Filter the files based on the '_graph.json' suffix and root directory
        self.graph_files = [os.path.join(root, file) for root, dirs, files in os.walk(dataset_dir) for file in files \
                            if file.endswith('_graph.json')]

    def __len__(self):
        # Return the length of the dataset
        return len(self.graph_files)

    def __getitem__(self, idx):
        try:
            # Attempt to open and load the JSON file
            with open(self.graph_files[idx], 'r') as json_file:
                graph_data = json.load(json_file)

            # Create a networkx graph from the loaded data
            graph = nx.node_link_graph(graph_data)

            # Convert the graph's adjacency matrix to a PyTorch tensor
            return torch.Tensor(nx.adjacency_matrix(graph).todense())
        except FileNotFoundError as e:
            # Handle the case where the file is not found with a custom message
            raise FileNotFoundError(f"File not found: {self.graph_files[idx]}") from e
        except Exception as e:
            # Handle other exceptions with a custom message
            raise Exception(f"Error while reading file {self.graph_files[idx]}: {e}") from e
