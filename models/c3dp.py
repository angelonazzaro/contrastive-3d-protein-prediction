from typing import Callable

import torch

from torch import nn, device as torch_device
from torch_geometric.nn import global_mean_pool

from models.loss import ContrastiveLoss
from transformers import AutoTokenizer, AutoModel, BertConfig

from models.constants import GRAPH_MODELS, DNA_MODEL, DNA_TOKENIZER, DNA_MAX_SEQUENCE_LENGTH, DNA_SEQUENCE_FEATURES, \
    SUPPORTED_DNA_POOLING


class C3DPNet(nn.Module):
    def __init__(self,
                 graph_model: str = "GraphSAGE",
                 temperature: float = 0.5,
                 dna_embeddings_pool: str = "mean",
                 graph_embeddings_pool: Callable = global_mean_pool,
                 out_features_projection: int = 768,
                 **kwargs):
        super().__init__()

        if dna_embeddings_pool not in SUPPORTED_DNA_POOLING:
            raise Exception(f"Mode not supported. Supported dna embeddings pooling: {SUPPORTED_DNA_POOLING}")

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
        self.dna_projection = nn.Linear(DNA_SEQUENCE_FEATURES, out_features_projection)
        self.graph_projection = nn.Linear(kwargs["hidden_channels"] if graph_model != "DiffPool"
                                          else kwargs["dim_target"], out_features_projection)
        self.device = torch_device('cpu', 0)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, sequences_A: list[str], batch: torch.Tensor,
                return_dict: bool = False):

        dna_inputs_list = []
        for primary_sequence in sequences_A:
            dna_inputs_list.append(self.dna_tokenizer(primary_sequence, return_tensors="pt", padding='max_length',
                                                      max_length=DNA_MAX_SEQUENCE_LENGTH, truncation=True)["input_ids"])
        dna_inputs = torch.cat(dna_inputs_list).to(self.device)
        dna_hidden_states = self.dna_model(dna_inputs)[0]  # [batch_size, sequence_length, 768]

        dna_embeddings = getattr(dna_hidden_states, self.dna_embeddings_pool)(dim=1)  # expected shape [batch_size, 768]

        graph_embeddings = self.graph_model(x=x, edge_index=edge_index, batch=batch)

        if self.__graph_model_name != "DiffPool":
            graph_embeddings = global_mean_pool(x=graph_embeddings, batch=batch)

        dna_embeddings = self.dna_projection(dna_embeddings)
        graph_embeddings = self.graph_projection(graph_embeddings)

        return self.loss(graph_embeddings, dna_embeddings, return_dict=return_dict)

    def to(self, device: str):
        m = super().to(device)
        self.device = next(m.parameters()).device
        return m
