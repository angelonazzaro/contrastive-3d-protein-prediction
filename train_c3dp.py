import os
import os.path as osp
from argparse import ArgumentParser
from functools import partial

import yaml

import wandb
from torch_geometric import seed_everything

from training.constants import *
from training.utils import train_model


def main(args):

    if not args.tune_hyperparameters and args.lr is None:
        raise Exception("You must set the learning rate (`learning_rate`) when training")

    if not args.tune_hyperparameters and args.weight_decay is None:
        raise Exception("You must set the weight decay (`weight_decay`) when training")

    print("Seed Everything to " + str(args.seed))
    seed_everything(args.seed)

    # run wandb sweep to tune hyperparameters
    if args.tune_hyperparameters:
        with open(args.sweep_config, "r") as f:
            sweep_config = yaml.load(f, Loader=yaml.FullLoader)
        sweep_id = wandb.sweep(sweep=sweep_config, project=args.project_name)
        sweep_config.update(vars(args))
        wandb.agent(sweep_id, partial(train_model, vars(args)), count=args.sweep_count)
    else:
        train_model(args=vars(args))

    wandb.finish()


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_root_dir", type=str, default=osp.join(os.getcwd(), "data"))
    parser.add_argument("--graph_model", type=str, default="GraphSAGE")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--dna_embeddings_pool", type=str, default="mean")
    parser.add_argument("--graph_embeddings_pool", type=str, default="mean")
    parser.add_argument("--out_features_projection", type=int, default=768)
    parser.add_argument("--in_channels", type=int, default=None)
    parser.add_argument("--hidden_channels", type=int, default=10)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--training_split_percentage", type=float, default=TRAINING_SPLIT_PERCENTAGE)
    parser.add_argument("--val_split_percentage", type=int, default=VALIDATION_SPLIT_PERCENTAGE)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--shuffle", action="store_false", default=True)
    parser.add_argument("--n_epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--metrics", nargs='*', default=['loss', 'val_loss'])
    parser.add_argument("--project_name", type=str, default="c3dp")
    parser.add_argument("--sweep_config", type=str, default=osp.join(os.getcwd(), "c3dp_sweep.yaml"))
    parser.add_argument("--sweep_count", type=int, default=10)
    parser.add_argument("--early_stopping_patience", type=int, default=7)
    parser.add_argument("--early_stopping_delta", type=float, default=0.0)
    parser.add_argument("--optimizer", type=str, default="AdamW")
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--tune_hyperparameters", action="store_true", default=True)

    main(parser.parse_args())