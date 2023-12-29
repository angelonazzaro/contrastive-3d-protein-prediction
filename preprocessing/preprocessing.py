import requests
import gzip
import os
import shutil
from tqdm import tqdm

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


def process_pdb_file(file_path: str, out_dir: str, response_format: str = 'json'):
    """
    Process a PDB file: get protein uniqid and move the file to the appropriate directory.
    """
    filename = os.path.basename(file_path)
    protein_name = filename.split('-')[1]
    r = requests.get(url=BASE_URL + protein_name + '.' + response_format)

    if r.status_code == 200:
        protein_dir = os.path.join(out_dir, protein_name)

        os.makedirs(protein_dir, exist_ok=True)

        # Move pdb file to protein dir
        shutil.move(file_path, os.path.join(protein_dir, filename))
        # Write metadata file to protein dir
        with open(os.path.join(protein_dir, f'{protein_name}.{response_format}'), 'wb') as f:
            f.write(r.content)
    else:
        print(r.status_code)
        print(r.content)
        return



def get_pdbs(dir_path: str, out_dir: str = 'data', response_format: str = 'json'):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    pdb_file_path_list = [os.path.join(dir_path, filename) for filename in os.listdir(dir_path)
                          if os.path.isfile(os.path.join(dir_path, filename)) and '.pdb' in filename]

    for file_path in tqdm(pdb_file_path_list, desc="Processing files", unit="file"):
        # Extract compressed files
        if file_path.endswith(('.zip', '.gz', '.tar.gz', '.rar')):
            file_path = extract_compressed_file(file_path)

        # Process PDB files
        process_pdb_file(file_path, out_dir, response_format)
