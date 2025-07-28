from dataclasses import dataclass, field
from functools import cached_property
from typing import Optional

import networkx as nx
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .parquet import extract_seed


@dataclass
class NosoiSimulation:
    """
    Container for a single nosoi simulation result, including:
    - the transmission table (as a pandas DataFrame)
    - simulation metadata (from Parquet file schema)
    - a NetworkX directed graph representation of the transmission chain
    """
    metadata: dict[str, float]
    seed: int
    _df: pd.DataFrame
    _graph: Optional[nx.DiGraph] = field(default=None, init=False, repr=False)

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    @df.setter
    def df(self, new_df: pd.DataFrame):
        self._df = new_df
        self._graph = self.df_to_graph(new_df)
        self._invalidate_caches()

    @property
    def graph(self) -> nx.DiGraph:
        if self._graph is None:
            self._graph = self.df_to_graph(self.df)
        return self._graph

    @graph.setter
    def graph(self, new_graph: nx.DiGraph):
        self._graph = new_graph
        self._df = self.graph_to_df(new_graph)
        self._invalidate_caches()

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

    def _invalidate_caches(self) -> None:
        """Invalidate cache when changing df or graph representation."""
        for attr in (
            "simtime", "n_hosts", "n_active", "n_deaths", "n_recoveries"
        ):
            self.__dict__.pop(attr, None)

    @classmethod
    def from_parquet(cls, parquet_path: Path) -> "NosoiSimulation":
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

        seed = extract_seed(Path(parquet_path))

        return cls(_df=df, metadata=metadata, seed=seed)

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

    @staticmethod
    def df_to_graph(df: pd.DataFrame) -> nx.DiGraph:
        """
        Convert the transmission chain to a directed NetworkX graph.

        Each node corresponds to a host with metadata (e.g. infection time),
        and each directed edge represents an infection event from infector to
        infectee.

        Parameters
        ----------
        df: pdf.DataFrame
            A pandas dataframe representation of the transmission chain.

        Returns
        -------
        nx.DiGraph
            Directed graph where nodes represent hosts and edges represent
            infection events.
        """
        graph: nx.DiGraph = nx.DiGraph()

        for _, row in df.iterrows():
            node_id = row["hosts.ID"]
            graph.add_node(node_id, **row.to_dict())
            if pd.notna(row["inf.by"]):
                graph.add_edge(int(row["inf.by"]), node_id)

        return graph

    @staticmethod
    def graph_to_df(graph: nx.DiGraph) -> pd.DataFrame:
        """
        Convert the transmission chain from nx.Digraph format to a pandas df.

        Each node corresponds to a host with metadata (e.g. infection time),
        and each directed edge represents an infection event from infector to
        infectee.

        Parameters
        ----------
        graph: nx.DiGraph
            A networkx digraph representation of the transmission chain.

        Returns
        -------
        pdf.DataFrame
            Dataframe containing all hosts and their respective properties,
            such as infection timing, infector and fate.
        """
        data = []
        for node_id, attr in graph.nodes(data=True):
            row = dict(attr)
            row["hosts.ID"] = node_id
            inf_by = next(
                (src for src, tgt in graph.in_edges(node_id)), np.nan
            )
            row["inf.by"] = inf_by
            data.append(row)

        df = pd.DataFrame(data)
        df = df.sort_values("hosts.ID").reset_index(drop=True)
        return df
