from torch_geometric.loader import DataLoader

from dataset import ProteinGraphDataset
from models.c3dp import C3DPNet
import numpy as np
import torch
from torch_geometric.datasets import TUDataset
from torchmetrics.functional import f1_score, auroc, accuracy, precision, recall
from torch_geometric.utils import scatter

import torch

from training.utils import compute_accuracy, compute_running_accuracy

if __name__ == '__main__':
    torch.manual_seed(42)
    dataset = ProteinGraphDataset(root='data')
    dataloader = DataLoader(dataset, batch_size=8)

    model = C3DPNet(**torch.load("experiments/l291rcs8/graphsage-parameters.pt", map_location=torch.device('cpu')))
    model.load_state_dict(torch.load("experiments/l291rcs8/graphsage-state-dict-val_loss=1.433738.pt", map_location=torch.device('cpu')))

    model.eval()
#    proj = torch.nn.Linear(768, dataset.num_classes)
    running_accuracy = 0
    steps = 1
    val_acc = torch.tensor(0.0)
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            output = model(data.x, data.edge_index, data.sequence_A, data.batch, return_dict=True)

            acc = compute_accuracy(output["logits"], len(data))
            print(acc)
            val_acc = compute_running_accuracy(acc, val_acc, batch_idx + 1)

            torch.cuda.empty_cache()

            # acc = accuracy(preds=y_pred, target=data.y, task='multiclass', num_classes=dataset.num_classes, average="macro")
            # running_accuracy = running_accuracy + 1 / steps * (acc - running_accuracy)
            # steps += 1

    print(val_acc)