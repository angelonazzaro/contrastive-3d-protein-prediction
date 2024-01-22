from torch_geometric.nn import GraphSAGE, GAT, GIN, GCN, GraphUNet


GRAPH_MODELS = {
    "GraphSAGE": GraphSAGE,
    "GIN": GIN,
    "GAT": GAT,
    "GCN": GCN,
    "GraphUNet": GraphUNet
}
