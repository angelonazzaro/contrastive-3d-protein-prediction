from torch_geometric.nn import GraphSAGE, GAT, GIN, GCN, GraphUNet
from models.diffpool import DiffPool

GRAPH_MODELS = {
    "GraphSAGE": GraphSAGE,
    "GIN": GIN,
    "GAT": GAT,
    "GCN": GCN,
    "GraphUNet": GraphUNet,
    "DiffPool": DiffPool
}
DNA_MODEL = "zhihan1996/DNABERT-2-117M"
DNA_TOKENIZER = "zhihan1996/DNABERT-2-117M"
