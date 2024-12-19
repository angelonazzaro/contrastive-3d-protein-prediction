
import os
import os.path as osp
from argparse import ArgumentParser
from functools import partial

import yaml
import torch
import wandb

from training.constants import *
from training.utils import train_model


def main(args):

    if not args.tune_hyperparameters and args.learning_rate is None:
        raise Exception("You must set the learning rate (`learning_rate`) when training")

    if not args.tune_hyperparameters and args.weight_decay is None:
        raise Exception("You must set the weight decay (`weight_decay`) when training")

    torch.set_float32_matmul_precision("medium")

    # run wandb sweep to tune hyperparameters
    if args.tune_hyperparameters:
        with open(args.sweep_config, "r") as f:
            sweep_config = yaml.load(f, Loader=yaml.FullLoader)
        sweep_id = wandb.sweep(sweep=sweep_config, project=args.project_name)
        sweep_config.update(vars(args))
        wandb.agent(sweep_id, partial(train_model, vars(args), sweep_config), count=args.sweep_count)
    else:
        train_model(args=vars(args))


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_name", type=str, default="proteins")
    parser.add_argument("--data_root_dir", type=str, default=osp.join(os.getcwd(), "data"))
    parser.add_argument("--experiment_dir", type=str, default=osp.join(os.getcwd(), "experiments"))
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--graph_model", type=str, default="GraphSAGE")
    parser.add_argument("--dna_embeddings_pool", type=str, default="mean")
    parser.add_argument("--graph_embeddings_pool", type=str, default="mean")
    parser.add_argument("--out_features_projection", type=int, default=768)
    parser.add_argument("--use_sigmoid", action="store_true", default=False)
    parser.add_argument("--in_channels", type=int, default=None)
    parser.add_argument("--hidden_channels", type=int, default=10)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dim_embedding", type=int, default=128,
                        help="Dimensionality of the embeddings for DiffPool")
    parser.add_argument("--gnn_dim_hidden", type=int, default=64,
                        help="Dimensionality of the GNN embeddings for DiffPool")
    parser.add_argument("--dim_embedding_MLP", type=int, default=50,
                        help="Dimensionality of the MLP embeddings for DiffPool")
    parser.add_argument("--max_num_nodes", type=int, default=2699,
                        help="Max number of nodes to consider for DiffPool")
    parser.add_argument("--depth", type=int, default=None,
                        help="Number of nodes to consider when pooling for GraphUNet")
    parser.add_argument("--out_channels", type=int, default=None,
                        help="Size of each output sample for GraphUNet")
    parser.add_argument("--training_split_percentage", type=float, default=TRAINING_SPLIT_PERCENTAGE)
    parser.add_argument("--val_split_percentage", type=int, default=VALIDATION_SPLIT_PERCENTAGE)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--shuffle", action="store_false", default=True)
    parser.add_argument("--n_epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--project_name", type=str, default="c3dp")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--sweep_config", type=str, default=osp.join(os.getcwd(), "c3dp_sweep.yaml"))
    parser.add_argument("--sweep_count", type=int, default=10)
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument("--early_stopping_delta", type=float, default=0.0)
    parser.add_argument("--optimizer", type=str, default="AdamW")
    parser.add_argument("--lr_scheduler", type=str, default="LinearLR")
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--tune_hyperparameters", action="store_true", default=False)

    main(parser.parse_args())
