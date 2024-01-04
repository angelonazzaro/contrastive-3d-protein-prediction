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
            if file.endswith('_graph.json') and root == 'preprocessing/data/'+file[:-11]
        ]

        print("List of JSON files:", self.graph_files)

    def __len__(self):
        # Return the length of the dataset
        print("Dataset length:", len(self.graph_files))
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
        except FileNotFoundError:
            # Handle the case where the file is not found
            return None
        except Exception as e:
            # Handle other exceptions and print an error message
            print(f"Error while reading file {graph_json_path}: {e}")
            return None

def custom_collate(batch):
    # Remove None values from the batch
    batch = [item for item in batch if item is not None]

    # Check if the batch is empty
    if len(batch) == 0:
        return torch.Tensor()  # Return an empty tensor

    # Stack the tensors in the batch
    return torch.stack(batch)

# Create an instance of the ProteinGraphDataset
custom_dataset = ProteinGraphDataset(dataset_dir='preprocessing/data/')

# Create a DataLoader with custom collation function
data_loader = DataLoader(custom_dataset, batch_size=64, shuffle=True, collate_fn=custom_collate)
