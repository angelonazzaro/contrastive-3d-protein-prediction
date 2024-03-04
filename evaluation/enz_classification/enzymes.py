import os
import csv
import torch
from graphein.protein.graphs import construct_graph
from pypdb.clients.pdb.pdb_client import get_pdb_file
from torch_geometric.utils import from_networkx
from dataset.preprocessing import NodeFeatureFormatter


# Open the input files
# with open('amino_enzymes.txt', 'r') as enzymes_file:
#    enzymes = [line.strip() for line in enzymes_file]

# with open('amino_no_enzymes.txt', 'r') as no_enzymes_file:
#    no_enzymes = [line.strip() for line in no_enzymes_file]

# Combine the lists and create a list of (pdb_code, enzyme) tuples
# data = [(pdb_code, 'yes') for pdb_code in enzymes] + [(pdb_code, 'no') for pdb_code in no_enzymes]

# Write the output file
# with open('enzymes_classification.csv', 'w', newline='') as output_file:
#    writer = csv.writer(output_file)
#    writer.writerow(['pdb_code', 'enzyme'])
#    writer.writerows(data)

def load_enzyme_data(file_path):
    """
    Load enzyme data from a CSV file.

    Args:
    file_path (str): The path to the CSV file containing enzyme data.

    Returns:
    list: A list of tuples containing (pdb_code, enzyme_label) data.
    """
    enzymes_data = []
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        # Check if the required columns exist in the CSV file
        if 'pdb_code' in reader.fieldnames and 'enzyme' in reader.fieldnames:
            # Read data from each row and append to enzymes_data list
            enzymes_data = [(row['pdb_code'], row['enzyme']) for row in reader]
        else:
            print("Error: The CSV file lacks the columns 'pdb_code' and 'enzyme' required by the script.")
    return enzymes_data


def save_pdb_file(pdb_code, pdb_content, out_dir):
    """
    Save PDB content to a file.

    Args:
    pdb_code (str): The PDB code.
    pdb_content (str): The content of the PDB file.
    out_dir (str): The output directory.

    Returns:
    str: The path to the saved PDB file.
    """
    pdb_path = os.path.join(out_dir, f"{pdb_code}.pdb")
    with open(pdb_path, "w") as file:
        file.write(pdb_content)
    return pdb_path


def process_enzymes(enzymes_data, out_dir):
    """
    Process enzyme data and save graph representations.

    Args:
    enzymes_data (list): A list of tuples containing (pdb_code, enzyme_label) data.
    out_dir (str): The output directory.
    """
    failed_enzymes = []
    n = NodeFeatureFormatter()

    # Loop through each enzyme data
    for pdb_code, _ in enzymes_data:
        pdb_content = get_pdb_file(pdb_code)
        pdb_path = save_pdb_file(pdb_code, pdb_content, out_dir)
        try:
            # Construct graph from PDB file
            pyg_graph = from_networkx(construct_graph(path=pdb_path, verbose=False))
            pyg_graph = n(pyg_graph)

            # Check for sequence_A, sequence_B, and sequence_L
            sequences = [key for key in pyg_graph.keys() if "sequence_" in key]

            if "sequence_A" not in sequences:
                pyg_graph["sequence_A"] = ""
                for sequence in sequences:
                    pyg_graph["sequence_A"] += pyg_graph[sequence]

            for sequence in sequences:
                if "sequence_A" != sequence:
                    del pyg_graph[sequence]

            if _ == "yes":
                pyg_graph.y = torch.tensor(1)
            else:
                pyg_graph.y = torch.tensor(0)

            # Save the preprocessed graph as a PyTorch tensor
            torch.save(pyg_graph, os.path.join(out_dir, f"{pdb_code}.pt"))
        except KeyError:
            # If KeyError occurs, skip the enzyme and add it to the list of failed enzymes
            failed_enzymes.append(pdb_code)
        finally:
            # Delete the temporary PDB file
            os.unlink(pdb_path) if os.path.exists(pdb_path) else None

    # Write the list of failed enzymes to a file
    with open(os.path.join(out_dir, "failed.txt"), "w") as f:
        f.write("\n".join(failed_enzymes))


def main():
    """
    Main function.
    """
    # Define the paths
    csv_file_path = 'enzymes_classification.csv'
    out_dir = os.path.join(os.getcwd(), "dataset")
    # Create the output directory if it does not exist
    os.makedirs(out_dir, exist_ok=True)
    # Load enzyme data from the CSV file
    enzymes_data = load_enzyme_data(csv_file_path)
    # If enzyme data is available, process it
    if enzymes_data:
        process_enzymes(enzymes_data, out_dir)


if __name__ == "__main__":
    main()
