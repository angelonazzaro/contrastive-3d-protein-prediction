from typing import Optional, Union

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel, BertConfig

from models.constants import DNA_MODEL, DNA_TOKENIZER, DNA_MAX_SEQUENCE_LENGTH, DNA_SEQUENCE_FEATURES, \
    SUPPORTED_DNA_POOLING, GRAPH_EMBEDDING_POOLS, GRAPH_MODELS
from models.loss import ContrastiveLoss


class C3DPNet(torch.nn.Module):
    def __init__(self,
                 graph_model: str = "GraphSAGE",
                 dna_embeddings_pool: str = "mean",
                 graph_embeddings_pool: str = "mean",
                 out_features_projection: int = 768,
                 use_sigmoid: bool = False,
                 **kwargs):
        super().__init__()

        if dna_embeddings_pool not in SUPPORTED_DNA_POOLING:
            raise Exception(f"Mode not supported. Supported dna embeddings pooling: {SUPPORTED_DNA_POOLING}")

        if graph_embeddings_pool not in GRAPH_EMBEDDING_POOLS.keys():
            raise Exception(f"Mode not supported. Supported graph embeddings pooling: "
                            f"{list(GRAPH_EMBEDDING_POOLS.keys())}")

        if graph_embeddings_pool is None:
            raise Exception("graph_embeddings_pool cannot be None")

        if graph_model not in GRAPH_MODELS.keys():
            raise Exception(f"graph model {graph_model} not supported. Supported models: {list(GRAPH_MODELS.keys())}")

        config = BertConfig.from_pretrained(DNA_MODEL)
        self.dna_model = AutoModel.from_config(config)
        self.dna_tokenizer = AutoTokenizer.from_pretrained(DNA_TOKENIZER)
        self.graph_model = GRAPH_MODELS[graph_model](**kwargs)
        self.dna_embeddings_pool = dna_embeddings_pool
        self.graph_embeddings_pool = GRAPH_EMBEDDING_POOLS[graph_embeddings_pool]
        self.loss = ContrastiveLoss(use_sigmoid=use_sigmoid)
        self.dna_projection = nn.Linear(DNA_SEQUENCE_FEATURES, out_features_projection)
        self.graph_projection = nn.Linear(kwargs["hidden_channels"] if graph_model != "DiffPool"
                                          else kwargs["dim_target"], out_features_projection)
        self.__graph_model_name = graph_model
        self.__out_features_projection = out_features_projection
        self.__graph_model_name = graph_model
        self.__graph_embeddings_pool = graph_embeddings_pool

    def encode_dna_sequence(self, sequences: Union[str, list[str]]):
        dna_inputs_list = []

        if isinstance(sequences, str):
            sequences = [sequences]

        for primary_sequence in sequences:
            dna_inputs_list.append(self.dna_tokenizer(primary_sequence, return_tensors="pt", padding='max_length',
                                                      max_length=DNA_MAX_SEQUENCE_LENGTH, truncation=True)["input_ids"])
        dna_inputs = torch.cat(dna_inputs_list).to(self.dna_model.device)
        dna_hidden_states = self.dna_model(dna_inputs)[0]  # [batch_size, sequence_length, 768]

        dna_embeddings = getattr(dna_hidden_states, self.dna_embeddings_pool)(dim=1)  # expected shape [batch_size, 768]

        if self.dna_embeddings_pool == "max":
            dna_embeddings = dna_embeddings[0]

        return dna_embeddings

    def encode_graph(self, x: torch.Tensor, edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None):
        graph_embeddings = self.graph_model(x=x, edge_index=edge_index, batch=batch)

        if self.__graph_model_name != "DiffPool":
            graph_embeddings = self.graph_embeddings_pool(x=graph_embeddings, batch=batch)

        return graph_embeddings

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, sequences: Union[str, list[str]],
                batch: Optional[torch.Tensor] = None, return_dict: bool = False):
        dna_embeddings = self.encode_dna_sequence(sequences)

        # for some reason x, edge_index and batch tensors are moved back to cpu 
        if x.device != dna_embeddings.device:
            x = x.to(dna_embeddings.device)
            edge_index = edge_index.to(dna_embeddings.device)
            
        if batch is not None:
            batch = batch.to(dna_embeddings.device)

        if x.shape[1] != self.graph_model.in_channels:
            x = torch.nn.Linear(x.shape[1], self.graph_model.in_channels)(x)

        graph_embeddings = self.encode_graph(x=x, edge_index=edge_index, batch=batch)

        dna_embeddings = self.dna_projection(dna_embeddings)
        graph_embeddings = self.graph_projection(graph_embeddings)

        output = self.loss(graph_embeddings, dna_embeddings)

        if return_dict:
            return {"loss": output["loss"], "logits": output["logits"]}
        return output["loss"]

    def constructor_serializable_parameters(self) -> dict:

        parameters = {
            "graph_model": self.__graph_model_name,
            "dna_embeddings_pool": self.dna_embeddings_pool,
            "graph_embeddings_pool": self.__graph_embeddings_pool,
            "use_sigmoid": self.loss.use_sigmoid,
            "out_features_projection": self.__out_features_projection,
        }

        if self.__graph_model_name != "DiffPool" and self.__graph_model_name != "GraphUNet":
            parameters.update({
                "in_channels": self.graph_model.in_channels,
                "hidden_channels": self.graph_model.hidden_channels,
                "num_layers": self.graph_model.num_layers,
            })
        elif self.__graph_model_name == "DiffPool":
            parameters.update({
                   "config": self.graph_model.config
            })
        else:
            parameters.update({
                "in_channels": self.graph_model.in_channels,
                "hidden_channels": self.graph_model.hidden_channels,
                "out_channels": self.graph_model.out_channels,
                "depth": self.graph_model.depth
            })

        return parameters
