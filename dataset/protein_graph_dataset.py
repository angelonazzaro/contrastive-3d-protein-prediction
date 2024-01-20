import torch
import os
import os.path as osp

from torch_geometric.data import Dataset, download_url, extract_zip, extract_gz, extract_tar
from torch_geometric.utils.convert import from_networkx

from graphein.protein import amino_acid_one_hot, meiler_embedding, add_aromatic_interactions, \
    hydrogen_bond_donor, hydrogen_bond_acceptor, add_atomic_edges
from graphein.protein.graphs import construct_graph
from graphein.protein.config import ProteinGraphConfig

from preprocessing import extract_compressed_file, NodeFeatureFormatter
from tqdm import tqdm

EDGE_CONSTRUCTION_FUNCTIONS = [add_aromatic_interactions, add_atomic_edges]
NODE_METADATA_FUNCTIONS = [amino_acid_one_hot, meiler_embedding]


class ProteinGraphDataset(Dataset):
    raw_url = "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000000805_243232_METJA_v4.tar"

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.transform = transform

    @property
    def raw_file_names(self):
        if not os.path.exists(self.raw_dir):
            return []

        return [filename for filename in os.listdir(self.raw_dir)
                if os.path.isfile(os.path.join(self.raw_dir, filename)) and '.pdb' in filename]

    @property
    def processed_file_names(self):
        if not os.path.exists(self.processed_dir):
            return []

        processed_file_names = [filename for filename in os.listdir(self.processed_dir)]

        return processed_file_names if len(processed_file_names) == len(self.raw_paths) else []

    def download(self):
        file_path = download_url(self.raw_url, self.raw_dir)
        file_path_ext = os.path.splitext(file_path)[1]

        if file_path_ext == ".zip":
            extract_zip(file_path, self.raw_dir)
        elif file_path_ext == ".tar":
            extract_tar(file_path, self.raw_dir, mode='r')
        elif file_path_ext == ".gz":
            extract_gz(file_path, self.raw_dir)
        else:
            raise Exception(f"{file_path_ext} file not supported. Supported types are: ['.zip', '.tar', '.gz']")

        os.unlink(file_path)

    def process(self):
        for idx, raw_path in enumerate(tqdm(self.raw_paths, desc="Processing files", unit="file")):
            filename = os.path.basename(extract_compressed_file(raw_path))
            protein_name = filename.split('-')[1]

            config = ProteinGraphConfig(edge_construction_functions=EDGE_CONSTRUCTION_FUNCTIONS,
                                        node_metadata_functions=NODE_METADATA_FUNCTIONS)

            pyg_graph = from_networkx(construct_graph(uniprot_id=protein_name, config=config, verbose=False))

            torch.save(pyg_graph, osp.join(self.processed_dir, f'data_{idx}.pt'))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        if self.transform:
            return self.transform(data)
        return data
