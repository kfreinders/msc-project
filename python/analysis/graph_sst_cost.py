from dataproc.summary_stats import safe_stat
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from time import process_time
import numpy as np


def compute_sst(G):
    """Mimic graph summary stats computation without passing a NosoiSim."""
    G = G.to_undirected(as_view=True)

    degrees = np.array([deg for _, deg in G.degree])
    ego_sizes = [len(G[n]) + 1 for n in G.nodes]

    diameter = (
        nx.diameter(G)
        if nx.is_connected(G) and len(G) > 1
        else np.nan
    )

    radius = (
        nx.radius(G)
        if nx.is_connected(G) and len(G) > 1
        else np.nan
    )

    efficiency = nx.global_efficiency(G) if len(G) > 1 else np.nan

    return {
        "SST_18": safe_stat(np.mean, degrees),
        "SST_19": nx.transitivity(G),
        "SST_20": nx.density(G),
        "SST_21": diameter,
        "SST_22": safe_stat(np.mean, pd.Series(ego_sizes)),
        "SST_23": radius,
        "SST_24": efficiency,
    }


def generate_random_graph(n_nodes: int, p: float = 0.01) -> nx.DiGraph:
    """Generate a random directed Erdos–Renyi graph with n_nodes and edge probability p."""
    return nx.gnp_random_graph(n_nodes, p, directed=True)


def main() -> None:
    sizes = range(10, 260, 10)
    n_replicates = 20
    results = []

    for size in sizes:
        print(f"Computing duration for size {size}...")
        for i in range(n_replicates):
            g = generate_random_graph(size)
            start = process_time()
            _ = compute_sst(g)
            end = process_time()
            results.append({"size": size, "duration": end - start})
            print(f"Trial {i}: {end - start}")

    # Convert to DataFrame and summarize
    df = pd.DataFrame(results)
    summary = df.groupby("size")["duration"].agg(["mean", "std"])

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(summary.index, summary["mean"], c="black")
    plt.errorbar(
        summary.index,
        summary["mean"],
        yerr=summary["std"],
        fmt="o",
        capsize=4,
        c="black"
    )

    plt.xlabel("Graph size (number of nodes)", fontsize=12)
    plt.ylabel("Time to compute graph statistics (s)", fontsize=12)
    plt.title("Runtime vs. Graph Size")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("graph_stat_runtime.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
