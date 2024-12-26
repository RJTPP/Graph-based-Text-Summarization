import sys
import os
import argparse
import pathlib

from typing import Dict, Tuple, List, Union, Optional
import orjson
from orjson import JSONDecodeError
import networkx as nx
from prettytable import PrettyTable

from setting import *

from helper_script.json_helper import *
from helper_script.file_reader_helper import *
from helper_script.func_timer import SingleTimer, MultipleTimer

from modules_script import m_preprocess_text
from modules_script import m_process_text
from modules_script import m_graph_nx
from modules_script import m_graph_custom
from modules_script import m_bfs_tree
from modules_script import m_rouge_score


# MARK: Util functions
def get_command_line_arg() -> argparse.ArgumentParser.parse_args:
    """
    Parse command line arguments using argparse
    """
    parser = argparse.ArgumentParser(description="Calculate inverse pagerank from json file")

    parser.add_argument(
        "-f", "--files", 
        nargs='*', 
        help="Calculate only specified json file(s)"
    )

    parser.add_argument(
        "-e", "--exclude",
        nargs='+',
        help="Exclude specified json file(s)" 
    )

    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress output"
    )

    args = parser.parse_args()

    return args


def print_settings() -> None:
    print(f"=== Settings === (Config in setting.py)\n")

    print(f"DATA_DIR\t\t\t: {DATA_DIR}")
    print(f"NLTK_PATH\t\t\t: {NLTK_PATH}")
    print(f"OUTPUT_DIR\t\t\t: {OUTPUT_DIR}")
    print(f"VALIDATION_FILE\t\t\t: {VALIDATION_FILE}")
    print()

    print(f"STOP_ON_ERROR\t\t\t: {STOP_ON_ERROR}")
    print(f"USE_PAGERANK_LIBRARY\t\t: {USE_PAGERANK_LIBRARY}")
    print(f"OUTPUT_GRAPH\t\t\t: {OUTPUT_GRAPH}")
    print(f"SHOW_GRAPH\t\t\t: {SHOW_GRAPH}")
    print()

    print(f"TARGET_DATA_KEY\t\t\t: {'.'.join(TARGET_DATA_KEY)}")
    print(f"MAX_CALCULATION_THRESHOLD\t: {CALCULATION_THRESHOLD}")
    print(f"MAX_CALCULATION_ITERATION\t: {MAX_CALCULATION_ITERATION}")
    print(f"MAX_TRUST_RANK_ITERATION\t: {MAX_TRUST_RANK_ITERATION}")
    print()


def print_timer(timer: SingleTimer, newline: bool = True) -> None:
    print(f"  ({timer.get_time_and_restart():.2f} ms)")
    if newline:
        print()


def get_all_files_name(dir: str, extension: Optional[List[str]] = None) -> List[str]:
    if extension is not None:
        return [file for file in os.listdir(dir) if os.path.isfile(os.path.join(dir, file)) and pathlib.Path(file).suffix in extension]

    return [file for file in os.listdir(dir) if os.path.isfile(os.path.join(dir, file))]


# MARK: Calculations Section
def calculation_processed_text(data_path: str, write_to_output: bool = True, output_path: str = OUTPUT_DIR, logging: bool = False) -> List[Tuple[str, str, int]]:
    
    """
    Preprocesses text data and writes the result to cache.

    Reads a JSON file from the given data path, preprocesses the text data by
    converting it to bigrams, merging multiple bigrams, and converting it to 
    weighted bigrams. The result is written to cache if write_to_output is True.

    Parameters
    ----------
    data_path : str
        The path to the JSON file containing the text data.
    write_to_output : bool, optional
        Whether to write the result to cache. Defaults to True.
    output_path : str, optional
        The path to write the result to. Defaults to OUTPUT_DIR.
    logging : bool, optional
        Whether to print the result. Defaults to False.

    Returns
    -------
    List[Tuple[str, str, int]]
        The preprocessed text data in the form of weighted bigrams.
    """
    print("Preprocessing data")
    # Get raw text data
    all_text_data = read_json(data_path)

    # Preprocess & process text data
    processed_text_data = m_process_text.json_to_bigrams(all_text_data, TARGET_DATA_KEY, throw_key_error=True)

    # Merge multiple text data (list of bigrams)
    processed_text_data = m_process_text.merge_multiple_bigrams_list(processed_text_data, sort=False)

    # Convert to weighted bigrams
    processed_text_data = m_process_text.bigrams_to_weighted_bigrams(processed_text_data, sort=True)

    # Write to cache
    if write_to_output:
        write_to_file(output_path, to_json(processed_text_data, indent=True), overwrite=True)

    return processed_text_data


def calculation_inverse_pagerank(word_graph: Union[nx.DiGraph, m_graph_custom.WeightedWordDiGraph], epsilon: float = CALCULATION_THRESHOLD, max_iter: int = MAX_CALCULATION_ITERATION) -> Dict[str, float]:
    """
    Calculate inverse PageRank scores on a given weighted directed graph.

    Parameters
    ----------
    word_graph : Union[nx.DiGraph, m_graph_custom.WeightedWordDiGraph]
        The weighted directed graph to calculate the scores on.
    max_iter : int, optional
        The maximum number of iterations. Defaults to MAX_CALCULATION_ITERATION.

    Returns
    -------
    Dict[str, float]
        A dictionary mapping each node to its inverse PageRank score.

    Notes
    -----
    Supports two types of weighted directed graph: nx.DiGraph and m_graph_custom.WeightedWordDiGraph.
    """
    inverse_pagerank_scores = None

    if isinstance(word_graph, nx.DiGraph):
        return m_graph_nx.get_inverse_pagerank(word_graph, max_iter=max_iter)

    elif isinstance(word_graph, m_graph_custom.WeightedWordDiGraph):
        return word_graph.get_inverse_pagerank(max_iter=max_iter, epsilon=epsilon)

    else:
        raise TypeError("word_graph must be either nx.DiGraph or m_graph_custom.WeightedWordDiGraph")


def calculation_trust_rank(word_graph: Union[nx.DiGraph, m_graph_custom.WeightedWordDiGraph], sorted_inverse_pagerank_scores: List[Tuple[str, float]], bias_amount: int, epsilon: float = CALCULATION_THRESHOLD, max_iter: int = MAX_TRUST_RANK_ITERATION, filter_threshold: float = TRUST_RANK_FILTER_THRESHOLD) -> None:
    if isinstance(word_graph, nx.DiGraph):
        graph = list(word_graph.edges(data=True))
        graph = [(n1, n2, p["weight"]) for n1, n2, p in graph]
        word_graph = m_graph_custom.WeightedWordDiGraph(graph)
    
    return word_graph.get_trust_rank(bias_amount, sorted_inverse_pagerank_scores, epsilon=epsilon, max_iter=max_iter, filter_threshold=filter_threshold)


# TODO
def calculation_bfs_tree(di_graph: m_graph_custom.WeightedWordDiGraph, trust_score: Dict[str, float], root_amount: int) -> List[str]:
    if root_amount > len(trust_score):
        root_amount = len(trust_score)
    roots = sorted(trust_score.keys(), key=trust_score.get, reverse=True)[:root_amount]

    bfs_tree_list = [m_bfs_tree.BfsBigramsTree.init_tree_from_weighted_digraph(di_graph, trust_score, root) for root in roots]


    all_summary = [m_bfs_tree.BfsBigramsTree.get_all_text(tree) for tree in bfs_tree_list]
    all_summary = sum(all_summary, [])

    return all_summary


# MARK: Main calculation function 
def calculation_main(data_dir: str, data_name: str, validation_text: Optional[str]) -> None:
    """
    Main calculation function.

    This function reads the json file from the specified directory with the given name,
    preprocesses the text, generates a graph, calculates the inverse pagerank, and writes
    the result to a new json file. If SHOW_GRAPH is set to True, it will also visualize
    the graph.

    Parameters
    ----------
    data_dir : str
        The directory of the json file
    data_name : str
        The name of the json file

    Returns
    -------
    None
    """

    global all_best_rouge_score

    data_path = f"{data_dir}/{data_name}"

    # Time function runtime
    running_timer = MultipleTimer(["func"])
    
    print(f"=== Calculating {data_name} ===\n")
    
    print("* ", end="")
    running_timer.timer["func"].start()
    bigrams_list = calculation_processed_text(
        data_path,
        output_path=os.path.join(OUTPUT_DIR, "graph", f"graph_{data_name}.json"),
        write_to_output=OUTPUT_GRAPH,
        logging=True
    )
    edge_list = bigrams_list
    print_timer(running_timer.timer["func"])


    # Generate graph
    print("* Creating graph")
    word_graph: Union[nx.DiGraph, m_graph_custom.WeightedWordDiGraph, None] = None
    if USE_PAGERANK_LIBRARY:
        word_graph = m_graph_nx.generate_graph(bigrams_list, weighted=True)
    else:
        word_graph = m_graph_custom.WeightedWordDiGraph(bigrams_list)
    print(f"  nodes: {len(word_graph.nodes)}, edges: {len(word_graph.edges)}")
    print_timer(running_timer.timer["func"])


    # Inverse-PageRank
    print("* Calculating inverse pagerank")
    inverse_pagerank_scores = calculation_inverse_pagerank(word_graph)
    sorted_inverse_pagerank_scores = m_graph_custom.get_sorted_rank_score(inverse_pagerank_scores)
    print(f"  Sum: {sum(inverse_pagerank_scores.values()): .4f}")  # Verifying
    print_timer(running_timer.timer["func"])
    

    # TrustRank (filtered out the nodes that have score less than TRUST_RANK_FILTER_THRESHOLD)
    print("* Calculating trustrank")
    trust_rank_scores = calculation_trust_rank(word_graph, sorted_inverse_pagerank_scores, bias_amount=TRUST_RANK_BIAS_AMOUNT, max_iter=MAX_TRUST_RANK_ITERATION, filter_threshold=TRUST_RANK_FILTER_THRESHOLD)
    sorted_trust_rank_scores = m_graph_custom.get_sorted_rank_score(trust_rank_scores)
    # print(f"  Sum: {sum(trust_rank_scores.values()): .4f}")  # Verifying
    print_timer(running_timer.timer["func"])


    # Filtering graph
    filtered_edge_list = [(n1, n2, score) for n1, n2, score in edge_list if trust_rank_scores.get(n1, 0) > TRUST_RANK_FILTER_THRESHOLD and trust_rank_scores.get(n2, 0) > TRUST_RANK_FILTER_THRESHOLD]

    # Graph to BFS tree (from filtered edge list)
    print("* Converting graph to BFS tree")
    filtered_word_graph = m_graph_custom.WeightedWordDiGraph(filtered_edge_list)
    all_summary = calculation_bfs_tree(filtered_word_graph, trust_rank_scores, TRUST_RANK_BIAS_AMOUNT)
    print_timer(running_timer.timer["func"])

    # TODO
    # Validation
    if VALIDATION_FILE is not None and validation_text is not None:
        # if validation_text is None:
            # raise ValueError("Validation file is specified but validation argument is not provided.")
        print("* Validating")
        validation_result = [
            {
                "text": summary,
                "rouge1": m_rouge_score.rouge1(summary, validation_text),
                "rouge2": m_rouge_score.rouge2(summary, validation_text),
                "rougeL": m_rouge_score.rouge_l(summary, validation_text)
            }
            for summary in all_summary
        ]

        validation_result.sort(key=lambda x: x["rougeL"].get("f-measure", 0), reverse=True)
        all_best_rouge_score[data_name] = {
            "summary" : validation_result[0].get("text", None),
            "score": (
                validation_result[0].get("rouge1", None),
                validation_result[0].get("rouge2", None),
                validation_result[0].get("rougeL", None)
            )
        }

    else:
        validation_result = None
        print("* Skipping validation")
    
    print_timer(running_timer.timer["func"])

    # ==== Write to files ====

    # Write inverse-PageRank score to file
    print("* Writing to output")
    write_to_file(
        os.path.join(OUTPUT_DIR, "inverse_pagerank" , f"inverse_pagerank_{data_name}"),
        to_json(sorted_inverse_pagerank_scores, indent=True), 
        overwrite=True
    )

    # Write TrustRank score to file
    write_to_file(
        os.path.join(OUTPUT_DIR, "trustrank" , f"trustrank_{data_name}"),
        to_json(sorted_trust_rank_scores, indent=True), 
        overwrite=True
    )

    # Write filtered graph to file
    write_to_file(
        os.path.join(OUTPUT_DIR,"graph" , f"filtered_graph_{data_name}"),
        to_json(filtered_edge_list, indent=True), 
        overwrite=True
    )

    # Write summary to file
    write_to_file(
        os.path.join(OUTPUT_DIR, "summary" , f"summary_{data_name}"),
        to_json(all_summary, indent=True), 
        overwrite=True
    )

    if validation_result is not None:
        write_to_file(
            os.path.join(OUTPUT_DIR, "validation" , f"validation_{data_name}"),
            to_json(validation_result, indent=True), 
            overwrite=True
        )
    print_timer(running_timer.timer["func"])


    running_timer.main.stop()

    # Visualize graph (if SHOW_GRAPH is set to True)
    if SHOW_GRAPH:
        print("* Visualizing graph")
        m_graph_nx.plot_graph(word_graph, node_size=100, weighted=True, with_labels=False)


# MARK: Main
def main() -> None:
    print("==================================")
    print()
    
    global cmd_arg, all_best_rouge_score
    
    data_file_name = None

    if cmd_arg.files:
        data_file_name = cmd_arg.files
    elif cmd_arg.exclude:
        data_file_name = get_all_files_name(DATA_DIR, [".json"])
        data_file_name = [file for file in data_file_name if file not in cmd_arg.exclude]
    else:
        data_file_name = get_all_files_name(DATA_DIR, [".json"])

    # Print calculating file(s)
    print("=== Running ===\n")
    print("Using networkx library" if USE_PAGERANK_LIBRARY else "Using custom graph", "\n")

    print("Data to calculate:")
    for i, data in enumerate(data_file_name):
        print(f"  {i+1}.) {data}")
    print()

    # Print settings
    print_settings()

    # Start timer
    main_timer = MultipleTimer()

    print("Loading validation file...")
    print()
    try:
        all_validation_text: dict = read_json(VALIDATION_FILE)
    except Exception as e:
        print(f"Error loading validation file: {e}")
        print()
        all_validation_text = {}


    # MARK: Calculation part in main
    # Calculate all file(s)
    for i, data in enumerate(data_file_name):
        all_best_rouge_score[data] = {}

        print(f"({i+1}/{len(data_file_name)}) ", end="")

        # Time each file runtime
        main_timer.newTimer(data)

        try:
            calculation_main(DATA_DIR, data, all_validation_text.get(data, None))
        except Exception as e:
            if STOP_ON_ERROR: raise(e)
            print(f"\nError calculating {data} ({type(e)}): {e}\n")

        main_timer.timer[data].stop()
        file_runtime = main_timer.timer[data].get_start_to_stop()
        print(f"Calculation runtime: {file_runtime:.2f} ms\n")

    print("\n=== Summary ===\n")


    # Print each runtime & total runtime
    runtime = main_timer.main.get_time_and_restart()


    if runtime < 1e4:
        print(f"Total runtime: {runtime:.2f} ms")
    else:
        print(f"Total runtime: {runtime/1e3:.3f} s")
    print()


    # Print best ROUGE scores and each file runtime, 
    summary_table = PrettyTable(["No.", "File", "Best ROUGE (P, R, F)", "Calculation Time (ms)"])
    summary_table.align = "l"
    for i, data in enumerate(data_file_name):
        scores = all_best_rouge_score[data].get("score", None)
        if scores is None:
            scores = "-, -, -"
        else:
            scores = ", ".join([f"{s.get('f-measure', None):.3f}" if s.get('f-measure', None) is not None else "-" for s in scores])

        summary_table.add_row([
            i+1,
            data,
            scores,
            f"{main_timer.timer[data].get_start_to_stop():.2f}"
        ])
    print(summary_table)


    print()

    # Print best summarization
    print("Best summarization:\n")
    for i, (file_name, file_summary) in enumerate(all_best_rouge_score.items()):
        file_summary = file_summary.get("summary", "-")
        print(f"{i+1}.) {file_name: <30}: {file_summary.capitalize()}{'.' if file_summary != '-' else ''}")
        print()
    print()

    return


if __name__ == "__main__":
    cmd_arg = get_command_line_arg()

    # {filename: {score1: {text: "", score: (...) }, ...}}
    all_best_rouge_score: Dict[str, Dict[str, Union[str, tuple]]] = {}

    if cmd_arg.quiet:
        import contextlib
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            main()
    else:
        main()
