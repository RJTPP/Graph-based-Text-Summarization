import sys
import os
from typing import Dict, Tuple, List, Set, Union, Optional

from collections import deque

import networkx as nx
import matplotlib.pyplot as plt


from m_graph_custom import *


current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)


class BfsBigramsTree:
    def __init__(self):
        self.root: str = ""
        self.nodes: Set[str] = set()
        self.children: Dict[str, List[str]] = {}

    def clear_tree(self):
        self.nodes = set()
        self.children = {}

    def print_tree(self):
        print(self.children)

    def get_all_path(self):
        all_path = []
        stack = [(self.root, [self.root])]  # Stack stores (current_node, current_path)

        # Use DFS
        while stack:
            current_node, path_buff = stack.pop()

            children = self.children.get(current_node, [])
            # If current_node is a leaf
            if not children:
                all_path.append(path_buff)  # Add the completed path
            else:
                # Add children to the stack with updated paths
                for child in children:
                    stack.append((child, path_buff + [child]))

        return all_path

    def get_all_text(self):
        all_bigrams_list: List[List[str]] = self.get_all_path()

        all_text: List[str] = []
        for bigrams_list in all_bigrams_list:
            word_list = [bigram.split()[0] for bigram in bigrams_list]
            word_list.append(bigrams_list[-1].split()[1])
            text = " ".join(word_list)
            all_text.append(text)

        return all_text


    def visualize_tree(self, node_size: int =1500, use_tree_layout: bool = False):
        """
        Graph Visualization with `nx_pydot` and `nx_agraph`

        nx_pydot:
        * Uses the `pydot` library to provide Graphviz layouts.
        * Deprecated and not actively maintained.
        * Suitable for lightweight or temporary use.
        * Installation:
            * `pip install pydot`
            * Install Graphviz:
                * macOS: `brew install graphviz`
                * Linux: `sudo apt-get install graphviz`



        nx_agraph:
        * Uses the `pygraphviz` library for Graphviz layouts.
        * Actively maintained and recommended for long-term use.
        * Preferred for future compatibility.
        * Installation:
            * `pip install pygraphviz`
            * Install Graphviz:
                * macOS: `brew install graphviz`
                * Linux: `sudo apt-get install graphviz graphviz-dev`

        **Recommendation**:
        * Use `nx_agraph` for better support and reliability in new projects.
        * Use `nx_pydot` only if `pygraphviz` is unavailable or difficult to set up.

        
        """
    
        G = nx.DiGraph()
        for node, children in self.children.items():
            for child in children:
                G.add_edge(node, child)
        
                # Use nx_agraph's graphviz_layout

        pos = nx.planar_layout(G, scale=1)
        
        if use_tree_layout:
            try:
                from networkx.drawing.nx_agraph import graphviz_layout
                pos = graphviz_layout(G, prog='dot', root=self.root)
            except ImportError:
                print("Can not import nx_agraph, using nx_pydot")
                from networkx.drawing.nx_pydot import graphviz_layout
                pos = graphviz_layout(G, prog="dot", root=self.root)
            

        nx.draw(G, pos, with_labels=True, node_size=node_size, node_color='lightblue', edge_color='gray')
        plt.show()


    @staticmethod
    def init_tree_from_weighted_digraph(weighted_digraph: WeightedWordDiGraph, trust_rank_scores: Dict[str, float], root: Optional[str] = None) -> ['BFS_Tree']:
        tree = BfsBigramsTree()
        q = deque()
        seen: Set[str] = set()
        layer: Dict[str: int] = {}

        if root is None:
            root = max(trust_rank_scores, key=trust_rank_scores.get)
        tree.root = root
        all_neighbors = weighted_digraph.neighbors
        q.append(root)
        seen.add(root)
        layer[root] = 0

                
        # TODO
        # Run BFS, priority node with larger trustRank

        while len(q) > 0:
            current_node = q.popleft()
            current_neighbor: List[Tuple[str]] = [n for n, _ in all_neighbors[current_node]]
            current_neighbor.sort(key=lambda x: trust_rank_scores.get(x, 0), reverse=True)  # sort by trustRank
            
            for neighbor in current_neighbor:
                if neighbor not in seen:
                    tree.nodes.add(neighbor)
                    if current_node not in tree.children:
                        tree.children[current_node] = []
                    tree.children[current_node].append(neighbor)

                    q.append(neighbor)
                    seen.add(neighbor)
                    layer[neighbor] = layer[current_node] + 1

        return tree

