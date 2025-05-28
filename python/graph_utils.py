import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data


def infection_table_to_pyg(df: pd.DataFrame) -> Data:
    # Create mapping from hosts.ID to row index
    node_ids = df["hosts.ID"].values
    id_to_index = {id_: idx for idx, id_ in enumerate(node_ids)}

    # Edge construction (parent → child)
    edges = []
    for child_idx, parent_id in zip(df.index, df["inf.by"]):
        if pd.notna(parent_id) and parent_id in id_to_index:
            parent_idx = id_to_index[int(parent_id)]
            edges.append([parent_idx, child_idx])  # directed edge

    edge_index = (
        torch.tensor(edges, dtype=torch.long).t().contiguous()
    )  # [2, num_edges]

    # Node features (example: tIncub, tRecov, fate)
    features = df[["tIncub", "tRecov", "fate"]].fillna(0.0).to_numpy(dtype=np.float32)
    x = torch.tensor(features, dtype=torch.float)

    # Optional: node labels — here, just use fate as example
    y = torch.tensor(df["fate"].fillna(0).astype(int).values, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y)

    return data
