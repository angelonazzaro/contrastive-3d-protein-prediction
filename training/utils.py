from typing import Tuple

import numpy as np
import torch
import wandb
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

from dataset import ProteinGraphDataset
from models.c3dp import C3DPNet


# All credits go to Bjarten and the other contributors: https://github.com/Bjarten/early-stopping-pytorch.git
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""

        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def get_splits(n_instances: int, train_split_percentage: float, val_split_percentage: float) -> Tuple[int, int, int]:
    """
    Calculate dataset splits based on specified percentages.

    Args:
        n_instances (int): Total number of instances.
        train_split_percentage (float): Percentage of instances for the training split.
        val_split_percentage (float): Percentage of instances for the validation split.

    Returns:
        Tuple[int, int, int]: Number of instances for training, validation, and test splits.
    """

    if train_split_percentage == 0 and val_split_percentage == 0:
        return 0, 0, n_instances

    train_split = int(n_instances * train_split_percentage / 100)
    remaining_split = n_instances - train_split
    val_split = int(n_instances * val_split_percentage / 100)
    test_split = remaining_split - val_split

    # If no test set is required, then test_split is just remainder, that we can add to the train
    if train_split_percentage + val_split_percentage >= 100.0:
        train_split = train_split + test_split
        test_split = 0

    return train_split, val_split, test_split


def train_model(args, config=None):
    with wandb.init(config=config):
        print("Loading data...")

        if args["tune_hyperparameters"]:
            config = wandb.config

        batch_size = args["batch_size"] if not args["tune_hyperparameters"] else config["batch_size"]
        n_epochs = args["n_epochs"] if not args["tune_hyperparameters"] else config["n_epochs"]

        dataset = ProteinGraphDataset(root=args["data_root_dir"])
        train_split, val_split, test_split = get_splits(n_instances=len(dataset),
                                                        train_split_percentage=args["training_split_percentage"],
                                                        val_split_percentage=args["val_split_percentage"])
        train_ds, val_ds, test_ds = random_split(dataset, [train_split, val_split, test_split])
        train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=args["shuffle"])
        val_dataloader = DataLoader(val_ds, batch_size=batch_size, shuffle=args["shuffle"])

        if args["in_channels"] is None:
            args["in_channels"] = dataset.num_node_features

        model = C3DPNet(graph_model=args["graph_model"], temperature=args["temperature"],
                        dna_embeddings_pool=args["dna_embeddings_pool"],
                        graph_embeddings_pool=args["graph_embeddings_pool"],
                        out_features_projection=args["out_features_projection"],
                        in_channels=args["in_channels"], hidden_channels=args["hidden_channels"],
                        num_layers=args["num_layers"])

        for epoch in range(n_epochs):
            avg_train_loss = train_epoch(model=model, train_dataloader=train_dataloader,
                                         config=args if not args["tune_hyperparameters"] else config)
            wandb.log({"loss": avg_train_loss, "epoch": epoch})

            # validation step
            model.eval()
            val_loss = 0.0

            for data in val_dataloader:
                loss = model(data.x, data.edge_index, data.sequence_A, data.batch)
                val_loss += loss.item()

            wandb.log({"val_loss": val_loss / len(val_dataloader)})


def train_epoch(model: C3DPNet, train_dataloader: DataLoader, config) -> float:
    optimizer = getattr(torch.optim, config["optimizer"])(model.parameters(), lr=config["learning_rate"],
                                                          weight_decay=config["weight_decay"])
    cum_loss = 0.0
    for data in train_dataloader:
        model.train()
        optimizer.zero_grad()  # Clear gradients

        loss = model(data.x, data.edge_index, data.sequence_A, data.batch)  # forward pass + compute loss
        cum_loss += loss.item()

        loss.backward()  # Derive gradients
        optimizer.step()  # Update parameters based on gradients

        wandb.log({"batch_loss": loss.item()})

    return cum_loss / len(train_dataloader)
