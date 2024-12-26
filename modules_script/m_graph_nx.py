import sys
import os
from typing import Dict, Tuple, List

import networkx as nx # type: ignore
import matplotlib.pyplot as plt # type: ignore


current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)


def generate_graph(bigrams_list, weighted=False) -> nx.DiGraph:
    word_graph = nx.DiGraph()
    if weighted:
        word_graph.add_weighted_edges_from(bigrams_list)
    else:
        word_graph.add_edges_from(bigrams_list)
    return word_graph

def reverse_graph(graph: nx.DiGraph) -> nx.DiGraph:
    return nx.reverse(graph)

def get_inverse_pagerank(graph: nx.DiGraph, alpha=0.85, max_iter=200) -> Dict[any, float]:
    # Reverse the graph
    reversed_graph = reverse_graph(graph)

    # Compute PageRank on the reversed graph
    scores = nx.pagerank(reversed_graph, alpha=alpha, max_iter=max_iter)
    return dict(scores)

def plot_graph(word_graph: nx.DiGraph, node_size=1500, with_labels=True, weighted=False) -> None:
    pos = nx.random_layout(word_graph)
    # pos = nx.spiral_layout(word_graph)

    nx.draw(word_graph, pos, with_labels=with_labels, node_size=node_size, node_color='skyblue', edge_color='gray')
    if weighted:
        edge_labels = nx.get_edge_attributes(word_graph, 'weight')
        nx.draw_networkx_edge_labels(word_graph, pos, edge_labels, font_size=5)
    plt.show()

def test_graph(size=100) -> None:
    from helper_script.file_reader_helper import read_from_file
    from helper_script.json_helper import read_json

    tweet_data = read_from_file("caches/processed_output/graph_Itaewon_tragedy.json", read_json)
    size = min(len(tweet_data), size)
    tweet_data = tweet_data[:size]

    word_graph = generate_graph(tweet_data, weighted=True)
    pagerank_scores = nx.pagerank(word_graph)
    inverse_pagerank_scores = get_inverse_pagerank(word_graph)
    print(pagerank_scores)
    print(inverse_pagerank_scores)
    print(sorted(pagerank_scores, key=pagerank_scores.get, reverse=True))
    print(sorted(inverse_pagerank_scores, key=inverse_pagerank_scores.get, reverse=True))

    print("Number of nodes: ", len(word_graph.nodes))
    print("Number of edges: ", len(word_graph.edges))
    plot_graph(word_graph, node_size=1500, weighted=True)
    print(word_graph.edges)

