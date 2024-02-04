import os
import re
import sys
import time
from typing import Tuple

import numpy as np
import torch
import wandb
from torch.utils.data import random_split
from torch_geometric import seed_everything
from torch_geometric.loader import DataLoader

from dataset import ProteinGraphDataset
from models.c3dp import C3DPNet
from training.logger import Logger
from tqdm import tqdm


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
            self.trace_func(f'Val loss did not improve. EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""

        p = os.path.splitext(self.path)
        path = p[0] + f"-val_loss={val_loss:.6f}" + p[1]

        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  '
                f'Saving model to {path}')

        torch.save(model.state_dict(), path)
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
    with wandb.init(name=args["run_name"] if args["run_name"] is not None else args['graph_model'].lower(),
                    config=config):

        experiment_dir = os.path.join(args["experiment_dir"], wandb.run.id)
        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir)

        if args["tune_hyperparameters"]:
            config = wandb.config

        logger = Logger(filepath=os.path.join(experiment_dir, "trainlog.txt"), mode="a")

        seed_everything(seed=args["seed"])

        logger.log(f"Seed everything to: {args['seed']}\n"
                   f"Launching training for experiment {wandb.run.id}: \n"
                   f"Experiment dir: {experiment_dir} \n"
                   f"Config: {config}\n"
                   f"Args: {args}\n")

        batch_size = args["batch_size"] if not args["tune_hyperparameters"] else config["batch_size"]
        n_epochs = args["n_epochs"] if not args["tune_hyperparameters"] else config["n_epochs"]

        logger.log(f"Loading dataset....\n")

        dataset = ProteinGraphDataset(root=args["data_root_dir"])
        logger.log(f"Loaded dataset: {dataset}\n"
                   f"==================\n"
                   f"Batch size: {batch_size}\n"
                   f"Dataset size: {len(dataset)} \n"
                   f"Number of graphs size: {len(dataset)}\n"
                   f"Number of edges features: {dataset.num_edge_features}\n"
                   f"Number of node features: {dataset.num_node_features}\n")
        train_split, val_split, test_split = get_splits(n_instances=len(dataset),
                                                        train_split_percentage=args["training_split_percentage"],
                                                        val_split_percentage=args["val_split_percentage"])
        train_ds, val_ds, test_ds = random_split(dataset, [train_split, val_split, test_split])
        train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=args["shuffle"])
        val_dataloader = DataLoader(val_ds, batch_size=batch_size, shuffle=args["shuffle"])

        if args["in_channels"] is None:
            args["in_channels"] = dataset.num_node_features

        logger.log(f"Loading model\n"
                   f"==================\n"
                   f"Graph Model: {args['graph_model']}\n"
                   f"In channels: {args['in_channels']} \n"
                   f"Hidden channels: {args['hidden_channels']}\n"
                   f"Num layers: {args['num_layers']}\n")

        model = C3DPNet(graph_model=args["graph_model"], temperature=args["temperature"],
                        dna_embeddings_pool=args["dna_embeddings_pool"],
                        graph_embeddings_pool=args["graph_embeddings_pool"],
                        out_features_projection=args["out_features_projection"],
                        use_sigmoid=args["use_sigmoid"],
                        in_channels=args["in_channels"], hidden_channels=args["hidden_channels"],
                        num_layers=args["num_layers"])

        logger.log(f"Loaded model: {model}\n")

        if args["checkpoint_path"] is not None:
            logger.log(f"Loading checkpoint: {args['checkpoint_path']}\n")
            model.load_state_dict(torch.load(args["checkpoint_path"]))

        checkpoint_saving_path = os.path.join(experiment_dir, f"{args['graph_model'].lower()}.pt")

        early_stopping_monitor = EarlyStopping(patience=args["early_stopping_patience"], verbose=True,
                                               delta=args["early_stopping_delta"], path=checkpoint_saving_path,
                                               trace_func=logger.log)

        learning_rate = args["learning_rate"] if not args["tune_hyperparameters"] else config["learning_rate"]
        weight_decay = args["weight_decay"] if not args["tune_hyperparameters"] else config["weight_decay"]
        optimizer = getattr(torch.optim,
                            args["optimizer"] if not args["tune_hyperparameters"]
                            else config["optimizer"])(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        logger.log("Starting training...\n")

        for epoch in range(n_epochs):
            avg_train_loss = train_epoch(model=model, train_dataloader=train_dataloader, optimizer=optimizer,
                                         epoch=epoch, n_epochs=n_epochs)

            # validation step
            model.eval()
            val_loss = 0.0

            progress_bar = tqdm(enumerate(val_dataloader), total=len(val_dataloader), file=sys.stdout,
                                desc=f'Validation')

            with torch.no_grad():
                for batch_idx, data in progress_bar:
                    output = model(data.x, data.edge_index, data.sequence_A, data.batch, return_dict=True)
                    val_loss += output["loss"].item()

                    acc = compute_accuracy(output["logits"], len(data))
                    progress_bar.set_postfix({"val_loss_step": output["loss"].item(), "val_acc": acc.item()})
                    wandb.log({"val_loss_step": output["loss"].item(), "val_acc": acc.item()})

            progress_bar.close()

            val_loss = val_loss / len(val_dataloader)

            logger.log(f"Epoch {epoch + 1} out of {n_epochs} - train_loss: {avg_train_loss:.6f} - "
                       f"val_loss: {val_loss:.6f}")
            wandb.log({"train_loss": avg_train_loss, "val_loss": val_loss, "epoch": epoch + 1})

            early_stopping_monitor(model=model, val_loss=val_loss)

            if early_stopping_monitor.early_stop:
                logger.log(f"Stopping training at Epoch: {epoch}. Val_loss did not improve in "
                           f"{early_stopping_monitor.patience} epochs. "
                           f"Best score: {early_stopping_monitor.best_score:.6f}")
                break

            torch.cuda.empty_cache()

    wandb.finish()


def train_epoch(model: C3DPNet, train_dataloader: DataLoader, optimizer: torch.optim.Optimizer, epoch: int,
                n_epochs: int) -> float:
    cum_loss = 0.0
    num_batches = len(train_dataloader)

    progress_bar = tqdm(enumerate(train_dataloader), total=num_batches, desc=f'Epoch {epoch + 1}/{n_epochs}',
                        file=sys.stdout)

    for batch_idx, data in progress_bar:
        model.train()
        optimizer.zero_grad()  # Clear gradients

        output = model(data.x, data.edge_index, data.sequence_A, data.batch, return_dict=True)  # forward pass + compute loss
        cum_loss += output["loss"].item()

        output["loss"].backward()  # Derive gradients
        optimizer.step()  # Update parameters based on gradients
        acc = compute_accuracy(output["logits"], len(data))

        progress_bar.set_postfix({'train_step_loss': output["loss"].item(), 'acc': acc.item()})
        wandb.log({"train_step_loss": output["loss"].item(), 'acc': acc.item()})

    progress_bar.close()

    # Returning the average batch loss
    return cum_loss / num_batches


def compute_accuracy(graph_logits: torch.Tensor, batch_size: int):
    ground_truth = torch.arange(len(graph_logits)).to(graph_logits.device)

    acc_g = (torch.argmax(graph_logits, 1) == ground_truth).sum()
    acc_d = (torch.argmax(graph_logits, 0) == ground_truth).sum()

    return (acc_g + acc_d) / 2 / batch_size
