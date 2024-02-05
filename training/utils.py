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
        self.dir_path = os.path.dirname(path)
        self.trace_func = trace_func

    def __call__(self, val_loss, model, optimizer):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'Val loss did not improve. EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, optimizer):
        """Saves model when validation loss decrease."""

        p = os.path.splitext(self.path)
        model_path = p[0] + f"-state-dict-val_loss={val_loss:.6f}" + p[1]
        constructor_parameters_path = p[0] + "-parameters" + p[1]
        optimizer_path = "optimizer-state-dict" + p[1]

        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  '
                f'Saving model to {model_path}')

        # remove all previous checkpoints 
        for filename in os.listdir(self.dir_path):
            if filename.endswith(p[1]):
                os.remove(os.path.join(self.dir_path, filename))
        
        torch.save(model.state_dict(), model_path)
        torch.save(model.constructor_serializable_parameters(), constructor_parameters_path)
        torch.save(optimizer.state_dict(), os.path.join(self.dir_path, optimizer_path))
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
            args["hidden_channels"] = config["hidden_channels"]
            args["num_layers"] = config["num_layers"]
            args["out_features_projection"] = config["out_features_projection"]
            args["batch_size"] = config["batch_size"]
            args["learning_rate"] = config["learning_rate"]
            args["weight_decay"] = config["weight_decay"]
            args["optimizer"] = config["optimizer"]
            args["lr_scheduler"] = config["lr_scheduler"]
            args["n_epochs"] = config["n_epochs"]

        logger = Logger(filepath=os.path.join(experiment_dir, "trainlog.txt"), mode="a")

        seed_everything(seed=args["seed"])

        logger.log(f"Seed everything to: {args['seed']}\n"
                   f"Launching training for experiment {wandb.run.id}: \n"
                   f"Experiment dir: {experiment_dir} \n"
                   f"Config: {config}\n"
                   f"Args: {args}\n")

        logger.log(f"Loading dataset....\n")

        dataset = ProteinGraphDataset(root=args["data_root_dir"])
        logger.log(f"Loaded dataset: {dataset}\n"
                   f"==================\n"
                   f"Batch size: {args['batch_size']}\n"
                   f"Dataset size: {len(dataset)} \n"
                   f"Number of graphs size: {len(dataset)}\n"
                   f"Number of edges features: {dataset.num_edge_features}\n"
                   f"Number of node features: {dataset.num_node_features}\n")
        train_split, val_split, test_split = get_splits(n_instances=len(dataset),
                                                        train_split_percentage=args["training_split_percentage"],
                                                        val_split_percentage=args["val_split_percentage"])
        train_ds, val_ds, test_ds = random_split(dataset, [train_split, val_split, test_split])
        train_dataloader = DataLoader(train_ds, batch_size=args["batch_size"], shuffle=args["shuffle"])
        val_dataloader = DataLoader(val_ds, batch_size=args["batch_size"], shuffle=args["shuffle"])

        if args["in_channels"] is None:
            args["in_channels"] = dataset.num_node_features

        logger.log(f"Loading model\n"
                   f"==================\n"
                   f"Graph Model: {args['graph_model']}\n"
                   f"In channels: {args['in_channels']} \n"
                   f"Hidden channels: {args['hidden_channels']}\n"
                   f"Num layers: {args['num_layers']}\n")

        
        logger.log(f"Loaded model: {model}\n")

        checkpoint_saving_path = os.path.join(experiment_dir, f"{args['graph_model'].lower()}.pt")

        early_stopping_monitor = EarlyStopping(patience=args["early_stopping_patience"], verbose=True,
                                               delta=args["early_stopping_delta"], path=checkpoint_saving_path,
                                               trace_func=logger.log)
        optimizer = getattr(torch.optim, args["optimizer"])(model.parameters(), lr=args["learning_rate"], weight_decay=args["weight_decay"])
       
        if args["checkpoint_path"] is not None:
            checkpoint_dir = os.path.dirname(args["checkpoint_path"])
            # get graph model name
            graph_model_name = os.path.basename(args["checkpoint_path"]).split("-")[0]
            ext = os.path.splitext(args["checkpoint_path"])[1]
            # get constructor parameters 
            constructor_parameters = torch.load(os.path.join(checkpoint_dir, f"{graph_model_name}-parameters{ext}"))
            # get optimizer checkpoint
            optimizer_state_dict = torch.load(os.path.join(checkpoint_dir, f"optimizer-state-dict{ext}"))

            logger.log(f"Loading model from checkpoint with arguments: {constructor_parameters}\n")
            model = C3DPNet(**constructor_parameters)
            logger.log(f"Loading model state_dict from checkpoint: {args['checkpoint_path']}\n")
            model.load_state_dict(torch.load(args["checkpoint_path"]))
            logger.log(f"Loading optimizer state_dict from checkpoint: {args['checkpoint_path']}\n")
            optimizer.load_state_dict(optimizer_state_dict))
        else:
            model = C3DPNet(graph_model=args["graph_model"], temperature=args["temperature"],
                            dna_embeddings_pool=args["dna_embeddings_pool"],
                            graph_embeddings_pool=args["graph_embeddings_pool"],
                            out_features_projection=args["out_features_projection"],
                            use_sigmoid=args["use_sigmoid"],
                            in_channels=args["in_channels"], hidden_channels=args["hidden_channels"],
                            num_layers=args["num_layers"])

        lr_scheduler = getattr(torch.optim.lr_scheduler, args["lr_scheduler"])(optimizer)
        
        logger.log("Starting training...\n")

        for epoch in range(args["n_epochs"]):
            avg_train_loss, train_acc = train_epoch(model=model, train_dataloader=train_dataloader, optimizer=optimizer,
                                         epoch=epoch, n_epochs=args["n_epochs"])

            # validation step
            model.eval()
            val_loss = 0.0

            progress_bar = tqdm(enumerate(val_dataloader), total=len(val_dataloader), file=sys.stdout,
                                desc=f'Validation')

            val_acc = 0
            with torch.no_grad():
                for batch_idx, data in progress_bar:
                    output = model(data.x, data.edge_index, data.sequence_A, data.batch, return_dict=True)
                    val_loss += output["loss"].item()

                    acc = compute_accuracy(output["logits"], len(data))
                    val_acc = compute_running_accuracy(acc, val_acc, batch_idx + 1)

                    progress_bar.set_postfix({"val_loss_step": output["loss"].item(), "val_acc_step": acc.item()})
                    wandb.log({"val_loss_step": output["loss"].item(), "val_acc_step": acc.item()})

            progress_bar.close()

            val_loss = val_loss / len(val_dataloader)

            logger.log(f"Epoch {epoch + 1} out of {args['n_epochs']} - train_loss: {avg_train_loss:.6f} - train_acc: {train_acc:.6f} - "
                       f"val_loss: {val_loss:.6f} - val_acc: {val_acc:.6f}")
            wandb.log({"train_loss": avg_train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc, "epoch": epoch + 1})

            early_stopping_monitor(model=model, val_loss=val_loss, optimizer=optimizer)

            if early_stopping_monitor.early_stop:
                logger.log(f"Stopping training at Epoch: {epoch}. Val_loss did not improve in "
                           f"{early_stopping_monitor.patience} epochs. "
                           f"Best score: {early_stopping_monitor.best_score:.6f}")
                break

            torch.cuda.empty_cache()
            lr_scheduler.step()

    wandb.finish()


def train_epoch(model: C3DPNet, train_dataloader: DataLoader, optimizer: torch.optim.Optimizer, epoch: int,
                n_epochs: int) -> Tuple[float, float]:
    cum_loss = 0.0
    num_batches = len(train_dataloader)

    progress_bar = tqdm(enumerate(train_dataloader), total=num_batches, desc=f'Epoch {epoch + 1}/{n_epochs}',
                        file=sys.stdout)

    train_acc = 0
    model.train()
    for batch_idx, data in progress_bar:
        optimizer.zero_grad() # # Clear gradients
        output = model(data.x, data.edge_index, data.sequence_A, data.batch, return_dict=True)  # forward pass + compute loss
        cum_loss += output["loss"].item()

        output["loss"].backward()  # Derive gradients
        optimizer.step()  # Update parameters based on gradients 

        acc = compute_accuracy(output["logits"], len(data))
        train_acc = compute_running_accuracy(acc, train_acc, batch_idx + 1)

        progress_bar.set_postfix({'train_step_loss': output["loss"].item(), 'acc_step': acc.item()})
        wandb.log({"train_step_loss": output["loss"].item(), 'acc_step': acc.item()})

    progress_bar.close()

    # Returning the average batch loss and accuracy
    return cum_loss / num_batches, train_acc


def compute_running_accuracy(curr_acc: torch.Tensor, prev_acc: torch.Tensor, step: int) -> float:
    return prev_acc + 1 / (step) * (curr_acc - prev_acc)


def compute_accuracy(graph_logits: torch.Tensor, batch_size: int):
    ground_truth = torch.arange(len(graph_logits)).to(graph_logits.device)

    acc_g = (torch.argmax(graph_logits, 1) == ground_truth).sum()
    acc_d = (torch.argmax(graph_logits, 0) == ground_truth).sum()

    return (acc_g + acc_d) / 2 / batch_size
