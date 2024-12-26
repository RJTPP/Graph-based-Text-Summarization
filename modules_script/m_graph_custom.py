import sys
import os
from typing import Dict, Tuple, List, Set, Union, Optional
import networkx as nx # type: ignore
import operator


current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)


class WeightedWordDiGraph():

    def __init__(self, edge_list: Union[List[Tuple[str, str, int]], None] = None):
        self.nodes: Set[str] = set()
        self.nodes_total_out_weight: Dict[str, int] = dict()
        self.neighbors: Dict[str, List[Tuple[str, int]]] = dict()
        self.edges: List[Tuple[str, str, int]] = []  # For easier way to reverse graph

        if edge_list is not None:
            self.add_edge_from_list(edge_list)

    @property
    def reversed_edges(self) -> List[Tuple[str, str, int]]:
        return [(node2, node1, weight) for node1, node2, weight in self.edges]
    
    
    def add_edge(self, weighted_edge: Tuple[str, str, int]) -> None: # (node1, node2, weight) -> None:
        node1, node2, weight = weighted_edge
        self.nodes.add(node1)
        self.nodes.add(node2)

        if node1 not in self.nodes_total_out_weight:
            self.nodes_total_out_weight[node1] = weight
        else:
            self.nodes_total_out_weight[node1] += weight

        # If the edge=(node1, node2, weight) is duplicate, but the weight is different, this condition will not work
        if weighted_edge not in self.edges:
            self.edges.append(weighted_edge)

        if node1 not in self.neighbors:
            self.neighbors[node1] = [(node2, weight)]
        else:
            self.neighbors[node1].append((node2, weight))
        
        if node2 not in self.neighbors:
            self.neighbors[node2] = []

    def add_edge_from_list(self, edge_list: List[Tuple[str, str, int]]) -> None:
        for edge in edge_list:
            self.add_edge(edge)
    
    def get_reversed_digraph(self) -> "WeightedWordDiGraph":
        new_graph = WeightedWordDiGraph()
        new_graph.add_edge_from_list(self.reversed_edges)
        return new_graph
    
    def markov_chain(self, alpha: float = 0.85, epsilon: float = 1e-5, max_iter: int = 200, bias_set: Optional[Set[str]] = None) -> Dict[str, float]:
        """
        Markov Chain algorithm is a base algorithm for PageRank and TrustRank algorithms.

        Parameters
        ----------
        alpha : float, optional
            The damping factor. Defaults to 0.85.
        epsilon : float, optional
            The convergence threshold. Defaults to 1e-5.
        max_iter : int, optional
            The maximum number of iterations. Defaults to 200.
        bias_set : Set[str], optional
            The set of nodes to bias the PageRank scores. If not provided, all nodes are biased equally (i.e. pagerank algorithm).

        Returns
        -------
        Dict[str, float]
            A dictionary mapping each node to its PageRank score.

        Notes
        -----
        The algorithm works by iterating over the nodes in the graph and updating their scores based on the scores of their neighbors.
        The scores are updated until the algorithm converges (i.e. the difference between the current and previous scores is less than epsilon).
        """
        """"""

        # If bias_set is not provided, assume no nodes are biased (by making every node a bias equally) -> pagerank algorithm
        if bias_set is None or len(bias_set) == 0:
            bias_set = set(self.nodes)
        
        # Normalized bias factor for biased nodes
        n = len(bias_set)
        starting_score = 1 / n
        scores = {node: starting_score if node in bias_set else 0 for node in self.nodes}


        based_score = (1 - alpha) / n

        for i in range(max_iter):
            # Initialize new scores only for nodes that are biased
            new_scores = {node: based_score if node in bias_set else 0 for node in self.nodes}
            dangling_sum = 0

            # Calculate for each round
            for node in self.nodes:
                neighbor_len = len(self.neighbors[node])
                if neighbor_len == 0:
                    dangling_sum += scores[node]
                else:
                    base_passing_score = alpha * scores[node] / self.nodes_total_out_weight[node]

                    for neighbor, weight in self.neighbors[node]:
                        passing_score = base_passing_score * weight
                        new_scores[neighbor] += passing_score
            
            if dangling_sum != 0:
                dangling_value = alpha * dangling_sum / n
                for node in self.nodes:
                    if node not in bias_set:
                        continue
                    new_scores[node] += dangling_value

            # Check for convergence
            if all(abs(new_scores[node] - scores[node]) < epsilon for node in self.nodes):
                break

            # Update scores for the next iteration
            scores = new_scores

        # return [(node, scores) for node, scores in sorted(scores.items(), key=operator.itemgetter(1), reverse=True)]
        return scores

    def get_pagerank(self, alpha: float = 0.85, epsilon: float = 1e-5, max_iter: int = 200) -> Dict[str, float]:
        return self.markov_chain(alpha, epsilon, max_iter, bias_set=None)
    
    def get_inverse_pagerank(self, alpha: float = 0.85, epsilon: float = 1e-5, max_iter: int = 200) -> Dict[str, float]:
        reversed_graph = self.get_reversed_digraph()
        return reversed_graph.get_pagerank(alpha, epsilon, max_iter)

    def get_trust_rank(self, bias_amount: int, inverse_pagerank_scores: Union[ Dict[str, float], List[Tuple[str, float]], None ] = None , alpha: float = 0.85, epsilon: float = 1e-5, max_iter: int = 200, filter_threshold: float = 0) -> Dict[str, float]:
        if bias_amount <= 0:
            raise ValueError("Bias amount must be greater than 0")

        if bias_amount > len(self.nodes):
            bias_amount = len(self.nodes)

        if inverse_pagerank_scores is None:
            inverse_pagerank_scores = self.get_inverse_pagerank(alpha, epsilon, max_iter)

        if isinstance(inverse_pagerank_scores, dict):
            inverse_pagerank_scores = get_sorted_rank_score(inverse_pagerank_scores)

        bias_set = set([sorted_score[0] for sorted_score in inverse_pagerank_scores[: bias_amount]])

        trust_rank_scores = self.markov_chain(alpha, 1e-10, max_iter, bias_set)
        if filter_threshold > 0:
            trust_rank_scores = {node: score for node, score in trust_rank_scores.items() if score > filter_threshold}
        return trust_rank_scores
 
    def __repr__(self) -> str:
        return f"WordWeightedDiGraph({self.neighbors})"


def get_sorted_rank_score(scores: Dict[str, float]) -> List[Tuple[str, float]]:
    return sorted(scores.items(), key=operator.itemgetter(1), reverse=True)

def compare_pagerank(score1: Dict[any, float], score2: Dict[any, float]) -> float:
    sum_diff = 0
    node_count = len(score1)

    for node in score1.keys():
        diff = (score1[node] - score2[node])**2
        sum_diff += diff

    return (sum_diff / node_count)**0.5

