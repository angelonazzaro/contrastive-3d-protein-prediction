import glob
import os
import os.path as osp
from functools import partial

import torch
from graphein.protein import amino_acid_one_hot, meiler_embedding, add_aromatic_interactions, \
    expasy_protein_scale, add_bond_order, add_hydrogen_bond_interactions, add_peptide_bonds
from graphein.protein.config import ProteinGraphConfig
from graphein.protein.graphs import construct_graph
from torch_geometric.data import Dataset, download_url, extract_zip, extract_gz, extract_tar
from torch_geometric.utils.convert import from_networkx
from tqdm import tqdm

from dataset.preprocessing import extract_compressed_file, add_k_nn_edges

EDGE_CONSTRUCTION_FUNCTIONS = [
    partial(add_k_nn_edges, long_interaction_threshold=0),
    add_hydrogen_bond_interactions,
    add_peptide_bonds,
    # add_ionic_interactions,
    # add_disulfide_interactions, 
    # add_aromatic_sulphur_interactions, 
    add_aromatic_interactions,
    add_bond_order,
]
NODE_METADATA_FUNCTIONS = {
    "meiler": meiler_embedding,
    "expasy": expasy_protein_scale,
    "amino_acid_one_hot": amino_acid_one_hot,
    # "hbond_donors": hydrogen_bond_donor,
    # "hbond_acceptors": hydrogen_bond_acceptor
}


class ProteinGraphDataset(Dataset):
    raw_url = "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000005640_9606_HUMAN_v4.tar"

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
        config = ProteinGraphConfig(node_metadata_functions=list(NODE_METADATA_FUNCTIONS.values()), edge_construction_functions=EDGE_CONSTRUCTION_FUNCTIONS)

        for idx, raw_path in enumerate(tqdm(self.raw_paths, desc="Processing files", unit="file")):
            filename = osp.basename(extract_compressed_file(raw_path))
            protein_name = filename.split('-')[1]

            try: 
                pyg_graph = from_networkx(construct_graph(uniprot_id=protein_name, config=config, verbose=False))

                torch.save(self.__apply_transform(pyg_graph, "pre_transform"), osp.join(self.processed_dir, f'data_{idx}.pt'))
            except:
                pass
            
            os.unlink(osp.join(self.raw_dir, filename))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        return self.__apply_transform(torch.load(osp.join(self.processed_dir, f'data_{idx}.pt')), "transform")
    
    def __apply_transform(self, sample, attr):
        attr = getattr(self, attr)
        if attr:
            if isinstance(attr, list):
                for t in attr:
                    sample = t(sample)
            else:
                sample = attr(sample)
        return sample
