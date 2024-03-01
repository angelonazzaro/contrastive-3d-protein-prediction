from torch_geometric.nn.models import GraphSAGE, GAT, GIN, GCN, GraphUNet
from torch_geometric.nn.pool import global_mean_pool, global_max_pool
from models.diffpool import DiffPool

GRAPH_MODELS = {
    "GraphSAGE": GraphSAGE,
    "GIN": GIN,
    "GAT": GAT,
    "GCN": GCN,
    "GraphUNet": GraphUNet,
    "DiffPool": DiffPool
}
GRAPH_EMBEDDING_POOLS = {
    "mean": global_mean_pool,
    "max": global_max_pool,
}

DNA_MODEL = "zhihan1996/DNABERT-2-117M"
DNA_TOKENIZER = "zhihan1996/DNABERT-2-117M"
DNA_MAX_SEQUENCE_LENGTH = 512
DNA_SEQUENCE_FEATURES = 768
SUPPORTED_DNA_POOLING = ["sum", "mean", "max"]
