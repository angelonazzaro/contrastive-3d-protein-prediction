from typing import Callable

import torch

from torch import nn, device as torch_device
from torch_geometric.nn import global_mean_pool

from models.loss import ContrastiveLoss
from transformers import AutoTokenizer, AutoModel, BertConfig

from models.constants import GRAPH_MODELS, DNA_MODEL, DNA_TOKENIZER


class C3DPNet(nn.Module):
    def __init__(self,
                 graph_model: str = "GraphSAGE",
                 temperature: float = 0.5,
                 dna_embeddings_pool: str = "mean",
                 graph_embeddings_pool: Callable = global_mean_pool,
                 **kwargs):
        super().__init__()

        if dna_embeddings_pool not in ["mean", "max"]:
            raise Exception("Mode not supported. Supported dna embeddings pooling: [mean, max]")

        if graph_embeddings_pool is None:
            raise Exception("graph_embeddings_pool cannot be None")

        if graph_model not in GRAPH_MODELS.keys():
            raise Exception(f"graph model {graph_model} not supported. Supported models: {list(GRAPH_MODELS.keys())}")

        config = BertConfig.from_pretrained(DNA_MODEL)
        self.dna_model = AutoModel.from_config(config)
        self.dna_tokenizer = AutoTokenizer.from_pretrained(DNA_TOKENIZER)
        self.graph_model = GRAPH_MODELS[graph_model](**kwargs)
        self.__graph_model_name = graph_model
        self.dna_embeddings_pool = dna_embeddings_pool
        self.graph_embeddings_pool = graph_embeddings_pool
        self.loss = ContrastiveLoss(temperature=temperature)
        self.device = torch_device('cpu', 0)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, primary_sequence: str, batch: torch.Tensor,
                return_dict: bool = False):
        dna_inputs = self.dna_tokenizer(primary_sequence, return_tensors="pt")["input_ids"]
        dna_hidden_states = self.dna_model(dna_inputs)[0]  # [1, sequence_length, 768]
        dna_embeddings = None  # expected shape 768

        if self.dna_embeddings_pool == "mean":
            dna_embeddings = torch.mean(dna_hidden_states[0], dim=0)
        elif self.dna_embeddings_pool == "max":
            dna_embeddings = torch.max(dna_hidden_states[0], dim=0)[0]

        graph_embeddings = self.graph_model(x=x, edge_index=edge_index, batch=batch)

        if self.__graph_model_name != "DiffPool":
            graph_embeddings = self.graph_embeddings_pool(x=graph_embeddings, batch=batch)

        return self.loss(graph_embeddings, dna_embeddings.unsqueeze(0), return_dict=return_dict)

    def to(self, device: str):
        m = super().to(device)
        self.device = next(m.parameters()).device
        return m
