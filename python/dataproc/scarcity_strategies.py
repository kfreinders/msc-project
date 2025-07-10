from abc import ABC, abstractmethod
import random
import networkx as nx


class DataScarcityStrategy(ABC):
    @abstractmethod
    def apply(self, graph: nx.DiGraph, seed: int) -> nx.DiGraph:
        """
        Apply a specific data scarcity strategy to the input graph.

        Parameters
        ----------
        graph : nx.DiGraph
            The reconstructed nosoi transmission/contact graph.
        seed : int
            Seed to reproduce the data scarcity strategy.

        Returns
        -------
        nx.DiGraph
            A new graph with the specified data scarcity applied.
        """
        pass


class RandomNodeDrop(DataScarcityStrategy):
    def __init__(self, drop_fraction: float):
        self.drop_fraction = drop_fraction

    def apply(self, graph: nx.DiGraph, seed: int) -> nx.DiGraph:
        # If drop_fraction is 0.0, then the graph remains unchanged and we can
        # just return a copy of the original graph
        if self.drop_fraction == 0.0:
            return graph

        g = graph.copy()
        nodes = list(g.nodes)

        rng = random.Random(seed)
        to_drop = rng.sample(nodes, int(len(nodes) * self.drop_fraction))

        for node in to_drop:
            if g.has_node(node):
                parents = list(g.predecessors(node))
                children = list(g.successors(node))
                g.remove_node(node)
                for p in parents:
                    for c in children:
                        g.add_edge(p, c)

        return g


class ShuffleEdges(DataScarcityStrategy):
    """
    Randomly shuffles a fraction of the edges in a directed graph.

    Avoids self-loops, parallel edges, and overlapping nodes in the swap.
    """
    def __init__(self, shuffle_fraction: float = 1.0):
        self.shuffle_fraction = shuffle_fraction

    def apply(self, graph: nx.DiGraph, seed: int) -> nx.DiGraph:
        if self.shuffle_fraction == 0.0:
            return graph

        g = graph.copy()
        edges = list(g.edges())
        rng = random.Random(seed)

        target_swaps = int(len(edges) * self.shuffle_fraction // 2)
        swaps_made = 0
        visited = set()

        while swaps_made < target_swaps:
            e1, e2 = rng.sample(edges, 2)
            u1, v1 = e1
            u2, v2 = e2

            if len({u1, v1, u2, v2}) < 4:
                continue

            edge_key = frozenset((e1, e2))
            if edge_key in visited:
                continue
            visited.add(edge_key)

            new_e1 = (u1, v2)
            new_e2 = (u2, v1)

            if (
                g.has_edge(*new_e1)
                or g.has_edge(*new_e2)
                or u1 == v2
                or u2 == v1
            ):
                continue

            g.remove_edge(*e1)
            g.remove_edge(*e2)
            g.add_edge(*new_e1)
            g.add_edge(*new_e2)
            swaps_made += 1

        return g
