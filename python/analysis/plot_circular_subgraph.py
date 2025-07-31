import argparse
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import networkx as nx

from dataproc.simulation_loader import NosoiSimulation


def sample_connected_subgraph(
    G: nx.DiGraph,
    root: int,
    depth: int
) -> nx.Graph:
    """
    Extract a connected subgraph from a nosoi transmission graph using BFS.

    This function performs a breadth-first search (BFS) starting from the given
    root node and collects all nodes within the specified depth. The resulting
    induced subgraph contains only these nodes and the edges among them.

    Parameters
    ----------
    G : nx.DiGraph
        The full nosoi transmission graph (directed).
    root : int
        Node ID to use as the root of the subgraph.
    depth : int
        Maximum BFS depth from the root to include nodes.

    Returns
    -------
    nx.Graph
        A subgraph of G containing the root and all reachable nodes within the
        given depth.
    """
    # BFS to get nodes within a given depth
    visited = set()
    queue = [(root, 0)]
    while queue:
        node, d = queue.pop(0)
        if d > depth or node in visited:
            continue
        visited.add(node)
        queue.extend((child, d + 1) for child in G.successors(node))

    H = G.subgraph(visited).copy()
    return H


def plot_transmission_subgraph(
    G: nx.DiGraph,
    root: int,
    depth: int,
    output_path: Path
) -> None:
    """
    Generate and save a visualization of a transmission chain.

    The subgraph is sampled from the full nosoi transmission graph by
    performing a BFS from the root node up to the given depth. Nodes are
    colored according to their distance from the root, and the layout is
    computed using Graphviz's radial 'twopi' layout.

    Parameters
    ----------
    G : nx.DiGraph
        The full nosoi transmission graph (directed).
    root : int
        Node ID to use as the root of the visualization.
    depth : int
        Maximum BFS depth from the root to include nodes in the plot.
    output_path : Path
        Path to save the generated figure (e.g., 'transmission_chain.pdf').
    """
    H = sample_connected_subgraph(G, root, depth)
    pos = nx.nx_agraph.graphviz_layout(H, prog="twopi")

    # Compute depth of each node from root
    depth_map = nx.single_source_shortest_path_length(H, root)
    max_depth = max(depth_map.values(), default=1)
    norm = mcolors.Normalize(vmin=0, vmax=max_depth)
    cmap = matplotlib.colormaps["viridis"]

    node_colors = [cmap(norm(depth_map.get(node, 0))) for node in H.nodes]

    plt.figure(figsize=(10, 10))
    nx.draw(
        H,
        pos,
        with_labels=False,
        arrows=True,
        node_size=100,
        node_color=node_colors,
        edge_color="gray",
        font_size=8,
    )

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.savefig(output_path, bbox_inches="tight")


def cli_main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize a transmission chain subgraph from a nosoi simulation. "
            "A subgraph is extracted from a specified root node up to a "
            "chosen depth, with node colors reflecting their distance from "
            "the root."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", type=Path, required=True,
        help="Path to the parquet file containing a nosoi transmission chain.",
    )
    parser.add_argument(
        "--root", type=int, default=1,
        help="Root node ID from which to sample the subgraph.",
    )
    parser.add_argument(
        "--depth", type=int, default=3,
        help="Maximum depth to explore from the root node.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("transmission_chain.pdf"),
        help="Path to save the generated plot.",
    )

    args = parser.parse_args()

    G = NosoiSimulation.from_parquet(args.input).graph
    plot_transmission_subgraph(
        G,
        root=args.root,
        depth=args.depth,
        output_path=args.output
    )


if __name__ == "__main__":
    cli_main()
