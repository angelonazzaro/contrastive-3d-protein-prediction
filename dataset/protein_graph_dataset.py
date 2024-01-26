import os
import os.path as osp
import glob

import torch
from graphein.protein import amino_acid_one_hot, meiler_embedding, add_aromatic_interactions, \
    add_atomic_edges
from graphein.protein.config import ProteinGraphConfig
from graphein.protein.graphs import construct_graph
from torch_geometric.data import Dataset, download_url, extract_zip, extract_gz, extract_tar
from torch_geometric.utils.convert import from_networkx
from tqdm import tqdm

from dataset.preprocessing import extract_compressed_file

EDGE_CONSTRUCTION_FUNCTIONS = [add_aromatic_interactions, add_atomic_edges]
NODE_METADATA_FUNCTIONS = [amino_acid_one_hot, meiler_embedding]


class ProteinGraphDataset(Dataset):
    raw_url = "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000000805_243232_METJA_v4.tar"

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        if not osp.exists(self.raw_dir):
            return []

        return [filename for filename in os.listdir(self.raw_dir)
                if osp.isfile(osp.join(self.raw_dir, filename)) and filename.endswith(".pdb.gz")]

    @property
    def processed_file_names(self):
        if not osp.exists(self.processed_dir):
            return []

        processed_file_names = glob.glob(osp.join(self.processed_dir, "data_*.pt"))
        return [osp.basename(filename) for filename in processed_file_names]

    def download(self):
        file_path = download_url(self.raw_url, self.raw_dir)
        file_path_ext = osp.splitext(file_path)[1]

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
            filename = osp.basename(extract_compressed_file(raw_path))
            protein_name = filename.split('-')[1]

            config = ProteinGraphConfig()

            pyg_graph = from_networkx(construct_graph(uniprot_id=protein_name, config=config, verbose=False))

            torch.save(self.__apply_transform(pyg_graph, "pre_transform"),
                       osp.join(self.processed_dir, f'data_{idx}.pt'))

            os.unlink(osp.join(self.raw_dir, filename))

    def len(self):
        return len(self.processed_file_names)

    def __apply_transform(self, sample, attr):
        attr = getattr(self, attr)
        if attr:
            if isinstance(attr, list):
                for t in attr:
                    sample = t(sample)
            else:
                sample = attr(sample)
        return sample

    def get(self, idx):
        return self.__apply_transform(torch.load(osp.join(self.processed_dir, f'data_{idx}.pt')), "transform")