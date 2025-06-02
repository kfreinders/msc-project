from dataclasses import dataclass, field
from functools import cached_property
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


@dataclass
class NosoiSimulation:
    """
    Container for a single nosoi simulation result, including:
    - the transmission table (as a pandas DataFrame)
    - simulation metadata (from Parquet file schema)
    - a lazily constructed NetworkX directed graph representation of the chain
    """
    df: pd.DataFrame
    metadata: dict[str, float]
    _graph: Optional[nx.DiGraph] = field(default=None, init=False, repr=False)

    @cached_property
    def simtime(self) -> float:
        """Total duration of the simulation (from metadata)."""
        return float(self.metadata.get("simtime", 0))

    @cached_property
    def n_hosts(self) -> int:
        """Total number of infected hosts."""
        return len(self.df)

    @cached_property
    def n_active(self) -> int:
        """Number of infective individuals at the end of simulation."""
        return (self.df["fate"] == 0).sum()

    @cached_property
    def n_deaths(self) -> int:
        """Number of deceased individuals at the end of simulation."""
        return (self.df["fate"] == 1).sum()

    @cached_property
    def n_recoveries(self) -> int:
        """Number of recovered individuals at the end of simulation."""
        return (self.df["fate"] == 2).sum()

    @classmethod
    def from_parquet(cls, parquet_path: str) -> "NosoiSimulation":
        """
        Load a nosoi simulation from a Parquet file.

        This method reads a compressed Parquet file containing a transmission
        chain, reconstructs host and infection identifiers, and extracts any
        embedded simulation metadata from the file's Arrow schema.

        Parameters
        ----------
        parquet_path : str
            Path to the Parquet file containing the simulation data.

        Returns
        -------
        NosoiSimulation
            An instance of NosoiSimulation with reconstructed DataFrame and
            decoded simulation metadata.
        """
        table = pq.read_table(parquet_path)
        df = cls._reconstruct_hosts_ID(table.to_pandas())

        metadata = {}
        for k, v in table.schema.metadata.items():
            metadata[k.decode()] = float(v.decode())

        return cls(df=df, metadata=metadata)

    @staticmethod
    def _reconstruct_hosts_ID(df: pd.DataFrame) -> pd.DataFrame:
        """
        Undo the row-index mapping to reconstruct a nosoi transmission chain.

        This restores the `hosts.ID` column based on row indices and remaps
        the `inf.by` column — which references row indices — back to actual
        host IDs. Also reorders the columns to a canonical layout if present.

        Parameters
        ----------
        df : pd.DataFrame
            A transmission chain DataFrame where `inf.by` uses 1-based row
            indexing instead of actual host IDs.

        Returns
        -------
        pd.DataFrame
            DataFrame with `hosts.ID` and `inf.by` properly reconstructed.
        """
        df["hosts.ID"] = np.arange(1, len(df) + 1)

        if "inf.by" in df.columns:
            valid_mask = (
                df["inf.by"].notna() & (df["inf.by"] > 0) & (df["inf.by"] <= len(df))
            )
            df.loc[valid_mask, "inf.by"] = df.loc[
                df.loc[valid_mask, "inf.by"].astype(int) - 1, "hosts.ID"
            ].values

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

    def as_graph(self) -> nx.DiGraph:
        """
        Lazily convert the transmission DataFrame to a directed NetworkX graph.

        This graph represents the infection tree: each node is a host, and
        directed edges point from infector to infectee. The graph is cached
        after first construction.

        Returns
        -------
        nx.DiGraph
            Directed graph of the transmission chain.
        """
        if self._graph is None:
            self._graph = self.df_to_nx_graph()
        return self._graph

    def df_to_nx_graph(self) -> nx.DiGraph:
        """
        Convert the transmission chain to a directed NetworkX graph.

        Each node corresponds to a host with metadata (e.g. infection time),
        and each directed edge represents an infection event from infector to
        infectee.

        Returns
        -------
        nx.DiGraph
            Directed graph where nodes represent hosts and edges represent
            infection events.
        """
        graph: nx.DiGraph = nx.DiGraph()

        for _, row in self.df.iterrows():
            node_id = row["hosts.ID"]
            graph.add_node(node_id, **row.to_dict())
            if pd.notna(row["inf.by"]):
                graph.add_edge(int(row["inf.by"]), node_id)

        return graph
