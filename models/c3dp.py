import torch

from torch import nn, device as torch_device

from models.loss import ContrastiveLoss
from transformers import AutoTokenizer, AutoModel, BertConfig

from models.constants import GRAPH_MODELS


class C3DPNet(nn.Module):
    def __init__(self,
                 bio_model: str = "zhihan1996/DNABERT-2-117M",
                 bio_tokenizer: str = "zhihan1996/DNABERT-2-117M",
                 graph_model: str = "GraphSAGE",
                 **kwargs):
        super().__init__()

        config = BertConfig.from_pretrained(bio_model)
        self.bio_tokenizer = AutoTokenizer.from_pretrained(bio_tokenizer)
        self.bio_model = AutoModel.from_config(config)
        self.graph_model = GRAPH_MODELS[graph_model](**kwargs)
        self.loss = ContrastiveLoss()
        self.device = torch_device('cpu', 0)

    def forward(self):
        pass

    def to(self, device):
        m = super().to(device)
        self.device = next(m.parameters()).device
        return m


