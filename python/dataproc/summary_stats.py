from collections import deque
import logging
from typing import Any, Callable, Union, Sequence
import numpy as np
import pandas as pd
import networkx as nx

from .simulation_loader import NosoiSimulation


def get_logger():
    return logging.getLogger(__name__)


def safe_stat(
    func: Callable,
    x: Sequence[float] | np.ndarray | pd.Series,
    default: float = 0.0
) -> float:
    """
    Compute a statistic and return a default if the input is empty or invalid.

    Parameters
    ----------
    func : callable
        A statistical function such as np.mean, np.median, etc.
    x : array-like
        Input data to apply the function to. Can be a list, numpy array, or
        pandas Series.
    default : float
        Fallback value if computation fails or result is NaN.

    Returns
    -------
    float
        Computed statistic or the default value.
    """
    series = pd.Series(x).dropna()
    if series.empty:
        return default

    try:
        val = func(series)
    except Exception as e:
        raise ValueError(
            f"Failed to compute statistic with {func.__name__}: {e}"
        )

    return float(val) if not pd.isna(val) else default


# ------------------------------------------------------------------------------
# SECTION: Infection Statistics
# ------------------------------------------------------------------------------


def compute_secondary_infections(
        simulation: NosoiSimulation
) -> dict[str, float]:
    """
    Compute summary statistics related to secondary infections.

    Parameters
    ----------
    simulation : NosoiSimulation
        A loaded NosoiSimulation instance containing the transmission chain.

    Returns
    -------
    dict[str, float]
        A dictionary of computed summary statistics:
        - SST_00: Count of non-infectors
        - SST_01: Mean number of secondary infections
        - SST_02: Median number of secondary infections
        - SST_03: Variance in number of secondary infections
        - SST_04: Fraction of infectors causing 50% of infections
        - SST_05: Fraction of hosts that caused at least one secondary
          infection
        - SST_06: Total number of hosts
    """
    hosts = simulation.df

    # Count how often each host infected others
    inf_by = hosts["inf.by"]
    valid_inf_by = inf_by.dropna().astype("Int64")
    freq = valid_inf_by.value_counts()

    if freq.empty:
        return {
            "SST_00": simulation.n_hosts,
            "SST_01": 0.0,
            "SST_02": 0.0,
            "SST_03": 0.0,
            "SST_04": 0.0,
            "SST_05": 0.0,
            "SST_06": simulation.n_hosts,
        }

    # Table of secondary infection counts
    table = freq.rename_axis("hosts.ID").reset_index(name="Frequency")

    # Fraction of infectors causing 50% of infections
    table_sorted = table.sort_values("Frequency", ascending=False)
    cumulative = table_sorted["Frequency"].cumsum()
    half = cumulative.iloc[-1] * 0.5
    n_top_50 = (cumulative >= half).values.argmax() + 1
    frac_top_50 = n_top_50 / len(table_sorted)

    return {
        "SST_00": (~hosts["hosts.ID"].isin(freq.index)).sum(),
        "SST_01": safe_stat(np.mean, table["Frequency"]),
        "SST_02": safe_stat(np.median, table["Frequency"]),
        "SST_03": safe_stat(np.var, table["Frequency"]),
        "SST_04": frac_top_50,
        "SST_05": (
            len(freq) / simulation.n_hosts if simulation.n_hosts > 0 else 0
        ),
        "SST_06": simulation.n_hosts,
    }


def compute_infection_timing(simulation: NosoiSimulation) -> dict[str, float]:
    """
    Compute infection duration and activity statistics.

    Parameters
    ----------
    simulation : NosoiSimulation
        A loaded NosoiSimulation instance containing the transmission chain.

    Returns
    -------
    dict[str, float]
        A dictionary of computed summary statistics:
        - SST_07: Average number of infections per day time step
        - SST_08: Mean duration of infection
        - SST_09: Median duration of infection
        - SST_10: Variance in infection duration
        - SST_11: Number of individuals still infectious at the end of the
          simulation.
        - SST_12: Proportion of all infected individuals who are still
          infectious at the end.
    """
    hosts = simulation.df

    delta = hosts["out.time"] - hosts["inf.time"]

    # Total host
    return {
        "SST_07": simulation.n_hosts / simulation.simtime,
        "SST_08": safe_stat(np.mean, delta),
        "SST_09": safe_stat(np.median, delta),
        "SST_10": safe_stat(np.var, delta),
        "SST_11": simulation.n_active,
        "SST_12": (
            simulation.n_active / simulation.n_hosts
            if simulation.n_hosts > 0 else 0
        ),
    }


def compute_infection_lag(simulation: NosoiSimulation) -> dict[str, float]:
    """
    Compute the time lag between infector and infectee infection times.

    Parameters
    ----------
    simulation : NosoiSimulation
        A loaded NosoiSimulation instance containing the transmission chain.

    Returns
    -------
    dict[str, float]
        A dictionary of computed summary statistics:
        - SST_13: Mean time between an infector's infection and the infection
          of their contacts.
        - SST_14: Median time lag between infector and infectee infections.
        - SST_15: Variance in infection lags.
        - SST_16: Shortest lag time per infector, averaged across all infectors
          (typical minimum delay to first transmission).
    """
    hosts = simulation.df
    simtime = simulation.simtime

    # Merge each host with their infector to calculate infection time lag
    merged = hosts.merge(
        hosts,
        left_on="inf.by",
        right_on="hosts.ID",
        suffixes=("", "_infector")
    )
    merged["inf_time_diff"] = merged["inf.time"] - merged["inf.time_infector"]

    if len(merged) < 2:
        default_lag = simtime + 1
        return {
            "SST_13": default_lag,
            "SST_14": default_lag,
            "SST_15": default_lag,
            "SST_16": default_lag,
        }

    return {
        "SST_13": safe_stat(np.mean, merged["inf_time_diff"], simtime + 1),
        "SST_14": safe_stat(np.median, merged["inf_time_diff"]),
        "SST_15": safe_stat(np.var, merged["inf_time_diff"]),
        "SST_16": (
            safe_stat(np.min, merged.groupby("inf.by")["inf_time_diff"].min())
        ),
    }


# TODO: don't hardcode the maximum simulation lenght here, but extract it from
# R/config.R
def compute_runtime_fraction(simulation: NosoiSimulation) -> dict[str, float]:
    """
    Compute the runtime fraction as a proportion of maximum simulation length.

    Parameters
    ----------
    simulation : NosoiSimulation
        A loaded NosoiSimulation instance containing the transmission chain.

    Returns
    -------
    dict[str, float]
        One summary statistic SS_18: runtime as fraction of 100 units.
    """
    return {
        "SST_17": simulation.simtime / 100
    }


# ------------------------------------------------------------------------------
# SECTION: Network Statistics
# ------------------------------------------------------------------------------


def sample_connected_subgraph(
    G: nx.Graph,
    sim_seed: int,
    min_nodes: int = 100,
    max_nodes: int = 200,
    max_attempts: int = 10,
    seed: Union[int, np.random.Generator, None] = None
) -> nx.Graph:
    """
    Sample a connected subgraph of G.

    Uses BFS from a random seed. Retries up to `max_attempts` if the sampled
    component is smaller than `min_nodes`, and traverses the full graph only as
    fallback.

    Parameters
    ----------
    G : nx.Graph
        Input graph.
    sim_seed: int
        Seed of the nosoi simulation, for logging purposes.
    min_nodes : int
        Minimum number of nodes required in the sampled subgraph. Default is
        100.
    max_nodes : int
        Maximum number of nodes allowed in the sampled subgraph. Default is
        200.
    max_attempts : int
        Number of retry attempts if the sampled subgraph is too small. Default
        is 10.
    seed : int, np.random.Generator, or None
        Random seed or generator for reproducibility. Default is None.

    Returns
    -------
    nx.Graph
        Subgraph induced by the sampled nodes.

    Raises
    ------
    RuntimeError
        If a sufficiently large subgraph cannot be found after `max_attempts`.
    """
    logger = get_logger()

    nodes = list(G.nodes)
    if len(nodes) < min_nodes:
        raise ValueError(
            f"Graph has only {len(nodes)} nodes, which is less than "
            f" min_nodes = {min_nodes}."
        )

    # Normalize the seed to a random Generator
    rng = (
        seed if isinstance(seed, np.random.Generator)
        else np.random.default_rng(seed)
    )

    for _ in range(max_attempts):
        start = rng.choice(nodes)
        visited: set[Any] = set()
        queue = deque([start])

        while queue and len(visited) < max_nodes:
            node = queue.popleft()
            if node not in visited:
                visited.add(node)
                queue.extend(n for n in G.neighbors(node) if n not in visited)

        if len(visited) >= min_nodes:
            return G.subgraph(visited).copy()

    logger.warning(
        f"Failed to sample a connected subgraph for seed {sim_seed} with at "
        f"least {min_nodes} nodes after {max_attempts} attempts."
    )

    largest_cc = max(nx.connected_components(G), key=len)
    return G.subgraph(largest_cc).copy()


def compute_network_statistics(
    simulation: NosoiSimulation,
    max_nodes: int = 200
) -> dict[str, float]:
    """
    Compute network-related summary statistics from the transmission chain.

    Parameters
    ----------
    simulation : NosoiSimulation
        A loaded NosoiSimulation instance containing the transmission chain.
    max_nodes : int, optional
        Maximum number of nodes to use for computing expensive statistics via
        subgraph sampling (default is 200).

    Returns
    -------
    dict[str, float]
        Summary statistics on degree, structure, and connectivity of the
        network:
        - SST_18: Average number of direct infection links per individual (mean
          degree).
        - SST_19: Likelihood that two contacts of the same individual also
          infected each other (global clustering coefficient).
        - SST_20: Proportion of all possible infection links that are present
          (graph density).
        - SST_21: Longest shortest path between any two individuals in the
          sampled graph (graph diameter).
        - SST_22: Average size of an individual's immediate infection
          neighborhood (ego graph size).
        - SST_23: Minimum number of steps to reach the furthest individual from
          the center of the sampled graph (graph radius).
        - SST_24: Overall ease of infection spreading across the network
          (global efficiency).
    """
    G = simulation.graph

    # Early return on single-node graph
    if G.number_of_nodes() == 1:
        return {
            "SST_18": 0.0,
            "SST_19": 0.0,
            "SST_20": 0.0,
            "SST_21": np.nan,
            "SST_22": 1.0,
            "SST_23": np.nan,
            "SST_24": np.nan,
        }

    undirected = G.to_undirected(as_view=True)

    degrees = np.array([deg for _, deg in undirected.degree])
    ego_sizes = [len(G[n]) + 1 for n in G.nodes]

    sampled = (
        sample_connected_subgraph(undirected, simulation.seed, max_nodes)
        if G.number_of_nodes() > max_nodes
        else undirected
    )

    diameter = (
        nx.diameter(sampled)
        if nx.is_connected(sampled) and len(sampled) > 1
        else np.nan
    )

    radius = (
        nx.radius(sampled)
        if nx.is_connected(sampled) and len(sampled) > 1
        else np.nan
    )

    efficiency = nx.global_efficiency(sampled) if len(sampled) > 1 else np.nan

    return {
        "SST_18": safe_stat(np.mean, degrees),
        "SST_19": nx.transitivity(G),
        "SST_20": nx.density(G),
        "SST_21": diameter,
        "SST_22": safe_stat(np.mean, pd.Series(ego_sizes)),
        "SST_23": radius,
        "SST_24": efficiency,
    }


# ------------------------------------------------------------------------------
# SECTION: Death Statistics
# ------------------------------------------------------------------------------


def compute_death_statistics(simulation: NosoiSimulation):
    """
    Compute summary statistics related to death and recovery outcomes.

    Parameters
    ----------
    simulation : NosoiSimulation
        A loaded NosoiSimulation instance containing the transmission chain.

    Returns
    -------
    dict[str, float]
        A dictionary of computed summary statistics:
        - SST_25: Total number of deaths.
        - SST_26: Fraction of individuals who died.
        - SST_27: Mean time to death.
        - SST_28: Variance in time to death.
        - SST_29: Total number of recoveries.
        - SST_30: Fraction of individuals who recovered.
        - SST_31: Mean time to recovery.
        - SST_32: Variance in time to recovery.
    """
    hosts = simulation.df
    if "fate" not in hosts.columns:
        return {
            "SST_25": np.nan,
            "SST_26": np.nan,
            "SST_27": np.nan,
            "SST_28": np.nan,
            "SST_29": np.nan,
            "SST_30": np.nan,
            "SST_31": np.nan,
            "SST_32": np.nan,
        }

    deaths = hosts["fate"] == 1
    recovered = hosts["fate"] == 2

    ttd = (hosts["out.time"] - hosts["inf.time"])[deaths]
    ttr = (hosts["out.time"] - hosts["inf.time"])[recovered]

    return {
        "SST_25": simulation.n_deaths,
        "SST_26": simulation.n_deaths / simulation.n_hosts,
        "SST_27": safe_stat(np.mean, ttd),
        "SST_28": safe_stat(np.var, ttd),
        "SST_29": simulation.n_recoveries,
        "SST_30": simulation.n_recoveries / simulation.n_hosts,
        "SST_31": safe_stat(np.mean, ttr),
        "SST_32": safe_stat(np.var, ttr),
    }


# ------------------------------------------------------------------------------
# Main function
# ------------------------------------------------------------------------------

def compute_summary_statistics(
    simulation: NosoiSimulation
) -> pd.DataFrame:
    """
    Compute summary statistics from a transmission chain.

    This function combines statistics from multiple aspects of the simulation,
    including secondary infection patterns, infection durations, timing of
    transmissions, network structure, and outcome distributions (deaths and
    recoveries).

    Parameters
    ----------
    simulation
        A loaded NosoiSimulation instance containing the transmission chain.

    Returns
    -------
    pd.DataFrame
        A single-row DataFrame with named summary statistics that characterize
        the transmission dynamics and outcomes of the simulation.
        - SST_00: Count of non-infectors
        - SST_01: Mean number of secondary infections
        - SST_02: Median number of secondary infections
        - SST_03: Variance in number of secondary infections
        - SST_04: Fraction of infectors causing 50% of infections
        - SST_05: Fraction of hosts that caused at least one secondary
          infection
        - SST_06: Total number of hosts
        - SST_07: Average number of infections per day time step
        - SST_08: Mean duration of infection
        - SST_09: Median duration of infection
        - SST_10: Variance in infection duration
        - SST_11: Number of individuals still infectious at the end of the
          simulation.
        - SST_12: Proportion of all infected individuals who are still
          infectious at the end.
        - SST_13: Mean time between an infector's infection and the infection
          of their contacts.
        - SST_14: Median time lag between infector and infectee infections.
        - SST_15: Variance in infection lags.
        - SST_16: Shortest lag time per infector, averaged across all infectors
          (typical minimum delay to first transmission).
        - SST_17: Runtime as a fraction of the maximum allowed simulation time.
        - SST_18: Average number of direct infection links per individual
          (mean degree).
        - SST_19: Likelihood that two contacts of the same individual also
          infected each other (global clustering coefficient).
        - SST_20: Proportion of all possible infection links that are
          present (graph density).
        - SST_21: Longest shortest path between any two individuals in the
          sampled graph (graph diameter).
        - SST_22: Average size of an individual's immediate infection
          neighborhood (ego graph size).
        - SST_23: Minimum number of steps to reach the furthest individual
          from the center of the sampled graph (graph radius).
        - SST_24: Overall ease of infection spreading across the network
          (global efficiency).
        - SST_25: Total number of deaths.
        - SST_26: Fraction of individuals who died.
        - SST_27: Mean time to death.
        - SST_28: Variance in time to death.
        - SST_29: Total number of recoveries.
        - SST_30: Fraction of individuals who recovered.
        - SST_31: Mean time to recovery.
        - SST_32: Variance in time to recovery.
    """
    sections = [
        compute_secondary_infections(simulation),
        compute_infection_timing(simulation),
        compute_infection_lag(simulation),
        compute_runtime_fraction(simulation),
        compute_network_statistics(simulation),
        compute_death_statistics(simulation),
    ]

    return pd.DataFrame([{
        key: value for section in sections for key, value in section.items()
    }])


def sst_to_name(sst: str) -> str:
    names = {
        "SST_00": "n_noninfectors",
        "SST_01": "mean_secinf",
        "SST_02": "med_secinf",
        "SST_03": "var_secinf",
        "SST_04": "frac_top_50",
        "SST_05": "frac_secinf",
        "SST_06": "n_hosts",
        "SST_07": "mean_infperday",
        "SST_08": "mean_inftime",
        "SST_09": "med_inftime",
        "SST_10": "var_inftime",
        "SST_11": "n_final_active",
        "SST_12": "frac_final_active",
        "SST_13": "mean_inflag",
        "SST_14": "med_inflag",
        "SST_15": "var_inflag",
        "SST_16": "mean_min_inflag",
        "SST_17": "frac_simtime",
        "SST_18": "graph_mean_degree",
        "SST_19": "graph_clustering_coeff",
        "SST_20": "graph_density",
        "SST_21": "graph_diameter",
        "SST_22": "graph_ego",
        "SST_23": "graph_radius",
        "SST_24": "graph_global_efficiency",
        "SST_25": "n_deaths",
        "SST_26": "frac_deaths",
        "SST_27": "mean_ttd",
        "SST_28": "var_ttd",
        "SST_29": "n_recov",
        "SST_30": "frac_recov",
        "SST_31": "mean_ttr",
        "SST_32": "var_ttr",
    }

    if sst in names:
        return names[sst]

    return sst
