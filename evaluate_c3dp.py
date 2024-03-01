import os
import sys
from argparse import ArgumentParser

import torch
from torch.utils.data import random_split
from torch_geometric import seed_everything

from dataset import ProteinGraphDataset
from evaluation.utils import get_model_basename, compute_metrics
from models.c3dp import C3DPNet
from training.constants import TRAINING_SPLIT_PERCENTAGE, BATCH_SIZE, VALIDATION_SPLIT_PERCENTAGE
from training.utils import get_splits, load_model_checkpoint

from torch_geometric.loader import DataLoader
from tqdm import tqdm


def evaluate_proteins(model: torch.nn.Module, device: torch.device, dtype: torch.dtype,
                      training_split_percentage: float, val_split_percentage: float, batch_size: int):
    dataset = ProteinGraphDataset(root="./data")
    train_split, val_split, test_split = get_splits(n_instances=len(dataset),
                                                    train_split_percentage=training_split_percentage,
                                                    val_split_percentage=val_split_percentage)
    train_ds, val_ds, test_ds = random_split(dataset, [train_split, val_split, test_split])
    test_dataloader = DataLoader(test_ds, batch_size=batch_size)

    progress_bar = tqdm(enumerate(test_dataloader), total=len(test_dataloader), file=sys.stdout,
                        desc=f'Testing')

    running_accuracy: float = 0
    running_f1: float = 0
    running_precision: float = 0
    running_recall: float = 0

    with torch.no_grad():
        for batch_idx, data in progress_bar:
            data = data.to(device, dtype)
            outputs = model(data.x, data.edge_index, data.sequence_A, data.batch, return_dict=True)
            n_classes = len(outputs["logits"])
            ground_truth = torch.arange(n_classes).to(device)

            acc, prec, rec, f1 = compute_metrics(logits=outputs["logits"], ground_truth=ground_truth,
                                                 n_classes=n_classes, batch_size=batch_size)

            running_precision = running_precision + 1 / (batch_idx + 1) * (prec - running_precision)
            running_recall = running_recall + 1 / (batch_idx + 1) * (rec - running_recall)
            running_accuracy = running_accuracy + 1 / (batch_idx + 1) * (acc - running_accuracy)
            running_f1 = running_f1 + 1 / (batch_idx + 1) * (f1 - running_f1)

            torch.cuda.empty_cache()

    return {
        "accuracy": running_accuracy,
        "precision": running_precision,
        "f1": running_f1,
        "recall": running_recall
    }


def evaluate_fold(model: torch.nn.Module, device: torch.device, dtype: torch.dtype,
                      training_split_percentage: float, val_split_percentage: float, batch_size: int):
    n_classes = 10

    running_accuracy: float = 0
    running_f1: float = 0
    running_precision: float = 0
    running_recall: float = 0

    proj = torch.nn.Linear(3, model.graph_model.in_channels)

    fold_classification_dir = os.path.join(os.getcwd(), "evaluation/fold_classification/dataset")
    data_list = []

    # Iterate over files in the directory
    for filename in os.listdir(fold_classification_dir):
        file_path = os.path.join(fold_classification_dir, filename)

        # Check if it is a file (not a subdirectory)
        if os.path.isfile(file_path) and ".txt" not in file_path:
            # Load data using torch.load and append to the list
            data_list.append(torch.load(file_path))
    dataloader = DataLoader(data_list, batch_size=batch_size)

    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), file=sys.stdout,
                        desc=f'Testing')

    with torch.no_grad():
        for batch_idx, data in progress_bar:
            data = data.to(device, dtype)
            data.x = proj(data.x)
            outputs = model(data.x, data.edge_index, data.sequence_A, data.batch, return_dict=True)
            acc, prec, rec, f1 = compute_metrics(logits=outputs["logits"], ground_truth=data.y,
                                                 n_classes=n_classes, batch_size=batch_size)

            running_precision = running_precision + 1 / (batch_idx + 1) * (prec - running_precision)
            running_recall = running_recall + 1 / (batch_idx + 1) * (rec - running_recall)
            running_accuracy = running_accuracy + 1 / (batch_idx + 1) * (acc - running_accuracy)
            running_f1 = running_f1 + 1 / (batch_idx + 1) * (f1 - running_f1)

            torch.cuda.empty_cache()

    return {
        "accuracy": running_accuracy,
        "precision": running_precision,
        "f1": running_f1,
        "recall": running_recall
    }


EVALUATION_DATASETS = {
    "proteins": evaluate_proteins,
    "fold": evaluate_fold
}


def main(args):
    if args.dataset not in EVALUATION_DATASETS:
        raise ValueError(f"{args.dataset} not present in EVALUATION_DATASETS. Available datasets: "
                         f"{EVALUATION_DATASETS.keys()}")

    print("Evaluating C3DP: Starting evaluation...")
    print(f"Seed Everything to {args.seed}")

    seed_everything(args.seed)

    print(f"Loading checkpoint: {args.model_checkpoint} and dataset: {args.dataset}")

    device = torch.device("cuda:0" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = None if device != "mps" else torch.float32
    model_state_dict, constructor_parameters = \
        load_model_checkpoint(args.model_checkpoint, device=device)
    print(f"Loading model from checkpoint with arguments: {constructor_parameters}\n")
    model = C3DPNet(**constructor_parameters)
    print(f"Loading model state_dict from checkpoint: {args.model_checkpoint}\n")
    model = model.to(device, dtype=dtype)
    model.load_state_dict(model_state_dict)

    model.eval()

    scores = EVALUATION_DATASETS[args.dataset](model=model, device=device, dtype=dtype,
                                               batch_size=args.batch_size,
                                               training_split_percentage=args.training_split_percentage,
                                               val_split_percentage=args.val_split_percentage)

    model_basename = args.model_basename if args.model_basename else get_model_basename(args.model_checkpoint)
    scores_path = os.path.join(args.scores_dir, args.scores_file)

    if not os.path.exists(args.scores_dir):
        os.makedirs(args.scores_dir)

    with open(scores_path, "a") as sf:
        sf.write("{:s}\t{:s}\n".format(model_basename,
                                       "\t".join(["{:s}: {:.3f}".format(k, s) for k, s in scores.items()])))

    print(f"Scores exported to: {scores_path}")


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scores_dir", type=str, default=os.path.join(os.getcwd(), "eval_results"))
    parser.add_argument("--scores_file", type=str, default="scores.tsv")
    parser.add_argument("--model_checkpoint", type=str, help="Path of the model to evaluate", required=True)

    parser.add_argument("--dataset", type=str, default="proteins")
    parser.add_argument("--model_basename", type=str, default=None)
    parser.add_argument("--training_split_percentage", type=float, default=TRAINING_SPLIT_PERCENTAGE)
    parser.add_argument("--val_split_percentage", type=int, default=VALIDATION_SPLIT_PERCENTAGE)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)

    main(parser.parse_args())
