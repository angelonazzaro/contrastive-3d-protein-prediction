import os

import torch
from graphein.protein.graphs import construct_graph
from pypdb.clients.pdb.pdb_client import get_pdb_file
from torch_geometric.utils import from_networkx

from dataset.preprocessing import NodeFeatureFormatter

# create an empty list to hold the pdb codes and fold_number
pdb_fold_pairs = []

# loop through the file numbers
# for i in range(10):
#     # open the file
#     filename = f"amino_fold_{i}.txt"
#     with open(filename, "r") as f:
#         # read the pdb codes and add them to the list
#         for line in f:
#             pdb_code = line.strip()
#             fold_no = i
#             pair = [pdb_code, fold_no]
#             pdb_fold_pairs.append(pair)
#
# # create the CSV file
# with open("fold_classification.csv", "w", newline="") as f:
#     writer = csv.writer(f)
#     # write the header row
#     writer.writerow(["pdb_code", "fold_number"])
#     # write the data rows
#     for pair in pdb_fold_pairs:
#         writer.writerow(pair)

failed_enzymes = []
out_dir = os.path.join(os.getcwd(), "dataset")

os.makedirs(out_dir, exist_ok=True)
idx = 0
n = NodeFeatureFormatter()
for i in range(10):
    # open the file
    filename = f"amino_fold_{i}.txt"
    with open(filename, "r") as f:
        # read the pdb codes and add them to the list
        for line in f:
            pdb_code = line.strip()
            pdb_file = get_pdb_file(pdb_code)
            pdb_path = os.path.join(out_dir, f"{pdb_code}.pdb")
            with open(pdb_path, "w") as file:
                file.write(pdb_file)
            try:
                pyg_graph = from_networkx(construct_graph(path=pdb_path, verbose=False))
            except:
                failed_enzymes.append(pdb_code)
                os.unlink(pdb_path)
                continue

            pyg_graph = n(pyg_graph)
            pyg_graph.y = torch.tensor([i])
            torch.save(pyg_graph, os.path.join(out_dir, f"{pdb_code}.pt"))
            idx += 1
            os.unlink(pdb_path)

with open(os.path.join(out_dir, "failed.txt")) as f:
    for s in failed_enzymes:
        f.write(f"{s}\n")