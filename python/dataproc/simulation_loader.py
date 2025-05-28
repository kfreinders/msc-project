import networkx as nx
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from summary_stats import compute_summary_statistics


def reconstruct_hosts_ID(df: pd.DataFrame) -> pd.DataFrame:
    """
    Undo the mapping in Parquet files to reconstruct a full transmission chain.

    This function restores the `hosts.ID` column based on row indices and
    remaps the `inf.by` column (which points to row indices in compressed
    storage) back to original host IDs. It also reorders the columns to a
    standardized format.

    Parameters
    ----------
    df : pd.DataFrame
        A transmission chain DataFrame where `inf.by` represents the 1-based
        row index of the infecting host, and `hosts.ID` may be missing.

    Returns
    -------
    pd.DataFrame
        A DataFrame with `hosts.ID` reconstructed, `inf.by` mapped to host IDs,
        and columns reordered to a consistent layout.
    """
    # Create a new 'hosts.ID' column based on row index (starting from 1)
    df["hosts.ID"] = np.arange(1, len(df) + 1)

    # In the Parquet files, we've dropped the hosts.ID colum and instead let
    # inf.by point to 1-based row index to save space. I.e., each row
    # represents an individual and its inf.by column containts the row number
    # its infector.
    if "inf.by" in df.columns:
        valid_mask = (
            df["inf.by"].notna() & (df["inf.by"] > 0) & (df["inf.by"] <= len(df))
        )
        df.loc[valid_mask, "inf.by"] = df.loc[
            df.loc[valid_mask, "inf.by"].astype(int) - 1, "hosts.ID"
        ].values

    # Ensure correct column order
    correct_order = [
        "hosts.ID",
        "inf.by",
        "inf.time",
        "out.time",
        "tIncub",
        "tRecov",
        "fate",
    ]
    existing_columns = [col for col in correct_order if col in df.columns]
    df = df[existing_columns]

    return df


def df_to_nx_graph(df: pd.DataFrame) -> nx.DiGraph:
    """
    Convert a reconstructed transmission chain to a directed NetworkX graph.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame with columns `hosts.ID`, `inf.by`, and metadata columns.

    Returns
    -------
    nx.DiGraph
        A directed graph where nodes represent hosts and edges represent
        infection events.
    """
    graph = nx.DiGraph()

    for _, row in df.iterrows():
        node_id = row["hosts.ID"]
        graph.add_node(node_id, **row.to_dict())
        if pd.notna(row["inf.by"]):
            graph.add_edge(int(row["inf.by"]), node_id)

    return graph


def load_simulation(
    parquet_path: str, as_graph: bool = False
) -> tuple[pd.DataFrame | nx.DiGraph, dict[str, str]]:
    """
    Load a nosoi simulation from a Parquet file.

    Reads a compressed Parquet file containing a transmission chain,
    reconstructs host and infection identifiers, and extracts any embedded
    file-level metadata (e.g., simulation parameters) from the Arrow schema.

    Parameters
    ----------
    parquet_path : str
        Path to the Parquet file containing the simulation data.
    as_graph : bool
        Optional flag to return the transmission chain as a NetworkX directed
        graph instad of a pandas DataFrame.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, str]]
        A tuple where the first element is the reconstructed transmission chain
        as a pandas DataFrame or NetworkX graph, and the second is a dictionary
        of decoded metadata stored in the file schema.
    """
    # Convert the parquet table to pandas df
    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    reconstructed = reconstruct_hosts_ID(df)

    # Also extract metadata
    schema = table.schema
    metadata = schema.metadata
    decoded = {k.decode(): v.decode() for k, v in metadata.items()}

    # Return reconstructed transmission chain
    return (reconstructed if not as_graph else df_to_nx_graph(df)), decoded


if __name__ == "__main__":
    f = "data/nosoi/inftable_1902751282_mapped.parquet"
    # graph, metadata = load_simulation(f, as_graph=True)
    df, metadata = load_simulation(f)
    compute_summary_statistics(df)
    print(df)
