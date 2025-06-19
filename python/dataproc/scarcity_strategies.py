from abc import ABC, abstractmethod
import random
import networkx as nx


class DataScarcityStrategy(ABC):
    @abstractmethod
    def apply(self, graph: nx.DiGraph) -> nx.DiGraph:
        """
        Apply a specific data scarcity strategy to the input graph.

        Parameters
        ----------
        graph : nx.DiGraph
            The reconstructed nosoi transmission/contact graph.

        Returns
        -------
        nx.DiGraph
            A new graph with the specified data scarcity applied.
        """
        pass


class RandomNodeDrop(DataScarcityStrategy):
    def __init__(self, drop_fraction: float):
        self.drop_fraction = drop_fraction

    def apply(self, graph: nx.DiGraph) -> nx.DiGraph:
        # If drop_fraction is 0.0, then the graph remains unchanged and we can
        # just return a copy of the original graph
        if self.drop_fraction == 0.0:
            return graph.copy()

        g = graph.copy()
        nodes = list(g.nodes)
        to_drop = random.sample(nodes, int(len(nodes) * self.drop_fraction))

        for node in to_drop:
            if g.has_node(node):
                parents = list(g.predecessors(node))
                children = list(g.successors(node))
                g.remove_node(node)
                for p in parents:
                    for c in children:
                        g.add_edge(p, c)

        return g
