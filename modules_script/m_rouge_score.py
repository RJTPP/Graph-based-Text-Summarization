import sys
import os
from typing import Union, Tuple, List, Dict, Iterable


current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)


# MARK: helper functions

def words_to_bigrams(text: List[str]) -> List[str]:
    return [f"{text[i]} {text[i+1]}" for i in range(len(text) - 1)]

def get_dp_matrix(text1: str, text2: str) -> List[List[int]]:
    words1 = text1.split()
    words2 = text2.split()
    m, n = len(words1), len(words2)

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # word matches
            if words1[i - 1] == words2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            # word doesn't match
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp

def find_longest_common_subsequence(text1: str, text2: str) -> List[str]:
    word1 = text1.split()
    word2 = text2.split()

    dp = get_dp_matrix(text1, text2)

    lcs = []
    i = len(word1)
    j = len(word2)

    while i > 0 and j > 0:
        # Word matches
        if word1[i-1] == word2[j-1]:
            lcs.append(word1[i-1])
            i -= 1
            j -= 1
        # Word doesn't match & left cell > top cell
        elif dp[i-1][j] > dp[i][j-1]:
                i -= 1
        # Word doesn't match & top cell > left cell
        else:
            j -= 1

    lcs.reverse()
    return lcs


def get_overlap(candidate_list: Iterable, reference_list: Iterable) -> int:
    candidate_set = set(candidate_list)
    reference_set = set(reference_list)

    overlap_list = None
    if len(candidate_list) < len(reference_list):
        overlap_list = [word for word in candidate_list if word in reference_set]
    else:
        overlap_list = [word for word in reference_list if word in candidate_set]

    return overlap_list


# MARK: Score Calculation

def precision_calculation(overlap_count: int, candidate_count: int) -> Union[int, float]:
    return overlap_count / candidate_count


def recall_calculation(overlap_count: int, reference_count: int) -> Union[int, float]:
    return overlap_count / reference_count


def f_measure_calculation(precision: float, recall: float) -> Union[int, float]:
    if precision == 0 or recall == 0:
        return 0
    return (2 * precision * recall) / (precision + recall)


def rouge_calculation(overlap_count: int, candidate_count: int, reference_count: int) -> Dict[str, Union[int, float]]:
    """
    Calculate the ROUGE score for a given candidate text and a reference text.

    Args:
    overlap_count (int): The number of overlapping words between the candidate text and the reference text.
    candidate_count (int): The number of words in the candidate text.
    reference_count (int): The number of words in the reference text.

    Returns:
    A tuple of three values: precision, recall, and f-measure. The values are of type float.
    """
    precision = precision_calculation(overlap_count, candidate_count)
    recall = recall_calculation(overlap_count, reference_count)
    f_measure = f_measure_calculation(precision, recall)

    scores = {
        "precision": precision,
        "recall": recall,
        "f-measure": f_measure
    }

    return scores


# MARK: ROUGE Score Calculation

def rouge1(candidate_text: str, reference_text: str) -> Dict[str, Union[int, float]]:
    candidate_text = candidate_text.split()
    reference_text = reference_text.split()

    len_cndt = len(candidate_text)
    len_ref = len(reference_text)
    
    overlap_count = len(get_overlap(candidate_text, reference_text))

    return rouge_calculation(overlap_count, len_cndt, len_ref)


def rouge2(candidate_text: str, reference_text: str) -> Dict[str, Union[int, float]]:
    candidate_bigrams = words_to_bigrams(candidate_text.split())
    reference_bigrams = words_to_bigrams(reference_text.split())

    len_cndt = len(candidate_bigrams)
    len_ref = len(reference_bigrams)
    
    overlap_count = len(get_overlap(candidate_bigrams, reference_bigrams))

    return rouge_calculation(overlap_count, len_cndt, len_ref)


def rouge_l(candidate_text: str, reference_text: str) -> Dict[str, Union[int, float]]: 
    len_cndt = len(candidate_text.split())
    len_ref = len(reference_text.split())
    len_lcs = len(find_longest_common_subsequence(candidate_text, reference_text))

    return rouge_calculation(len_lcs, len_cndt, len_ref)

