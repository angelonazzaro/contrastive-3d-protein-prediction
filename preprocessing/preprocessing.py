import requests
import gzip
import os
import shutil
import networkx as nx
import json

from tqdm import tqdm
from Bio.PDB import PDBParser

BASE_URL = 'https://rest.uniprot.org/uniprotkb/'


def extract_compressed_file(file_path: str):
    """
    Extract a compressed file and return the path to the extracted file.
    """
    with gzip.open(file_path, 'rb') as f_in:
        extracted_file_path = file_path[:-3]
        with open(extracted_file_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    return extracted_file_path


def process_pdb_file(file_path: str, out_dir: str, response_format: str = 'json') -> bool:
    """
    Process a PDB file: get protein uniqid, extract the graph from the pdb file and
    move everything to the appropriate directory.
    """
    filename = os.path.basename(file_path)
    protein_name = filename.split('-')[1]
    r = requests.get(url=BASE_URL + protein_name + '.' + response_format)

    if r.status_code == 200:
        protein_dir = os.path.join(out_dir, protein_name)

        os.makedirs(protein_dir, exist_ok=True)
        new_file_path = os.path.join(protein_dir, filename)
        # Move pdb file to protein dir
        shutil.move(file_path, new_file_path)
        # Write metadata file to protein dir
        with open(os.path.join(protein_dir, f'{protein_name}.{response_format}'), 'wb') as f:
            f.write(r.content)

        # Create json file containing the graph from the pdb file
        # Extract information from PDB
        parser = PDBParser()
        structure = parser.get_structure(protein_name, new_file_path)

        # Create a graph using networkx
        graph = nx.Graph()

        for model in structure:
            for chain in model:
                # Add nodes for each residue
                for residue in chain:
                    for atom in residue:
                        node_attrs = {
                            'atom_type': atom.get_name(),
                            'amino_acid': residue.resname,
                            'coordinates': atom.get_coord().tolist()
                        }
                        graph.add_node(residue.id, **node_attrs)

                # Add edges between consecutive residues in the chain
                residues = list(chain)
                for i in range(len(residues) - 1):
                    graph.add_edge(residues[i].id, residues[i + 1].id)

        # Save the graph to a JSON file
        graph_json_path = protein_dir + os.sep + protein_name + '_graph.json'
        # Serialize the graph to JSON format
        graph_data = nx.node_link_data(graph)
        with open(graph_json_path, 'w') as json_file:
            json.dump(graph_data, json_file)
        return True
    else:
        return False


def get_pdbs(dir_path: str, out_dir: str = 'data', response_format: str = 'json'):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    pdb_file_path_list = [os.path.join(dir_path, filename) for filename in os.listdir(dir_path)
                          if os.path.isfile(os.path.join(dir_path, filename)) and '.pdb' in filename]

    for file_path in tqdm(pdb_file_path_list, desc="Processing files", unit="file"):
        # Extract compressed files
        if file_path.endswith(('.zip', '.gz', '.tar.gz', '.rar')):
            file_path = extract_compressed_file(file_path)

        # Check if the file still exists before processing
        if os.path.exists(file_path):
            # Process PDB files
            if not process_pdb_file(file_path, out_dir, response_format):
                # Separate files that could not be parsed due to API calls limitations
                failed_dir = "FAILED_" + out_dir
                if not os.path.exists(failed_dir):
                    os.makedirs(failed_dir, exist_ok=True)
                # Use os.path.basename to get the filename without the path
                shutil.move(file_path, os.path.join(failed_dir, os.path.basename(file_path)))
