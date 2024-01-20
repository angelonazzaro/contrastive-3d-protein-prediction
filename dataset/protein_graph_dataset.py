import torch
import os
import os.path as osp
import networkx as nx
import numpy as np

from torch_geometric.data import Dataset, download_url, extract_zip, extract_gz, extract_tar
from torch_geometric.utils.convert import from_networkx

from graphein.protein.graphs import construct_graph
from graphein.protein.config import ProteinGraphConfig

from tqdm import tqdm
from dataset.preprocessing import extract_compressed_file

class ProteinGraphDataset(Dataset):
    raw_url = "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000000805_243232_METJA_v4.tar"

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

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

            config = ProteinGraphConfig()

            pyg_graph = self.construct_detailed_graph(uniprot_id=protein_name, config=config, verbose=False)

            torch.save(pyg_graph, osp.join(self.processed_dir, f'data_{idx}.pt'))

    def construct_detailed_graph(self, uniprot_id, config, verbose=False):
        G = construct_graph(uniprot_id=uniprot_id, config=config, verbose=verbose)

        for model in G:
            for chain in model:
                residues = list(chain)
                self.calculate_bond_distances_angles(G, residues)

                for i in range(len(residues) - 1):
                    distance = np.linalg.norm(residues[i + 1]["CA"].get_coord() - residues[i]["CA"].get_coord())
                    G.add_edge(residues[i].id, residues[i + 1].id, distance=distance)

                    if i >= 2:
                        angle = self.calculate_dihedral_angle(
                            residues[i - 2]["CA"].get_coord(),
                            residues[i - 1]["CA"].get_coord(),
                            residues[i]["CA"].get_coord(),
                            residues[i + 1]["CA"].get_coord()
                        )
                        G.edges[(residues[i - 1].id, residues[i].id)]['torsion_angle'] = angle

                    covalent_bonds = ["C-N", "N-C", "C-O", "O-C", "N-CA", "CA-N"]
                    for bond_type in covalent_bonds:
                        bond_atoms = bond_type.split('-')
                        if bond_atoms[0] in residues[i] and bond_atoms[1] in residues[i + 1]:
                            G.edges[(residues[i].id, residues[i + 1].id)][bond_type] = True

        return G

    def calculate_bond_distances_angles(self, G, residues):
        for i in range(len(residues) - 1):
            distance = np.linalg.norm(residues[i + 1]["CA"].get_coord() - residues[i]["CA"].get_coord())
            G.add_edge(residues[i].id, residues[i + 1].id, distance=distance)

            if i >= 2:
                angle = self.calculate_dihedral_angle(
                    residues[i - 2]["CA"].get_coord(),
                    residues[i - 1]["CA"].get_coord(),
                    residues[i]["CA"].get_coord(),
                    residues[i + 1]["CA"].get_coord()
                )
                G.edges[(residues[i - 1].id, residues[i].id)]['torsion_angle'] = angle

            covalent_bonds = ["C-N", "N-C", "C-O", "O-C", "N-CA", "CA-N"]
            for bond_type in covalent_bonds:
                bond_atoms = bond_type.split('-')
                if bond_atoms[0] in residues[i] and bond_atoms[1] in residues[i + 1]:
                    G.edges[(residues[i].id, residues[i + 1].id)][bond_type] = True

    def calculate_dihedral_angle(self, p1, p2, p3, p4):
        b1 = -1.0 * (p2 - p1)
        b2 = p3 - p2
        b3 = p4 - p3

        b2 /= np.linalg.norm(b2)

        v = b1 - np.dot(b1, b2) * b2
        w = b3 - np.dot(b3, b2) * b2

        x = np.dot(v, w)
        y = np.dot(np.cross(b2, v), w)

        return np.degrees(np.arctan2(y, x))

    def get_coordinates_from_atom(self, atom):
        return atom.get_coord().tolist()

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        return torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
