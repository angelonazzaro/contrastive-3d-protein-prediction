import os
import os.path as osp
from argparse import ArgumentParser
from typing import Callable

from torch.utils.data import random_split
from torch_geometric import seed_everything
from torch_geometric.nn import global_mean_pool

from dataset import ProteinGraphDataset
from models.c3dp import C3DPNet
from training.constants import *
from training.utils import get_splits, train_model

from torch_geometric.loader import DataLoader


def main(args):
    print("Seed Everything to " + str(args.seed))
    seed_everything(args.seed)
    print("Loading model...")

    model = C3DPNet(graph_model=args.graph_model, temperature=args.temperature,
                    dna_embeddings_pool=args.dna_embeddings_pool,
                    graph_embeddings_pool=args.graph_embeddings_pool,
                    out_features_projection=args.out_features_projection)

    print("Loading data...")

    dataset = ProteinGraphDataset(root=args.data_root_dir)
    train_split, val_split, test_split = get_splits(n_instances=len(dataset),
                                                    train_split_percentage=args.training_split_percentage,
                                                    val_split_percentage=args.val_split_percentage)
    train_ds, val_ds, test_ds = random_split(dataset, [train_split, val_split, test_split])
    train_dataloader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=args.shuffle)
    val_dataloader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=args.shuffle)
    test_dataloader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=args.shuffle)

    train_model(model=model, train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                epochs=args.epochs)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_root_dir", type=str, default=osp.join(os.getcwd(), "data"))
    parser.add_argument("--graph_model", type=str, default="GraphSAGE")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--dna_embeddings_pool", type=str, default="mean")
    parser.add_argument("--graph_embeddings_pool", type=Callable, default=global_mean_pool)
    parser.add_argument("--out_features_projection", type=int, default=768)
    parser.add_argument("--training_split_percentage", type=float, default=TRAINING_SPLIT_PERCENTAGE)
    parser.add_argument("--val_split_percentage", type=int, default=VALIDATION_SPLIT_PERCENTAGE)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--shuffle", action="store_false", default=True)
    parser.add_argument("--n_epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--metrics", nargs='*', default=['loss', 'val_loss'])
    parser.add_argument("--project_name", type=str, default="c3dp")
    parser.add_argument("--early_stopping_patience", type=int, default=7)
    parser.add_argument("--early_stopping_delta", type=float, default=0.0)

    main(parser.parse_args())
