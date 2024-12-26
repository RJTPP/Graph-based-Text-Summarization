import sys
import os
from typing import Dict, Tuple, List
import operator

from helper_script.json_helper import *
from m_preprocess_text import *


current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)


# Create bigram
def pair_word_to_bigram(preprocessed_words: List[str]) -> List[Tuple[str, str]]:
    """Create bigram out of a list of preprocessed list of words

    Args:
        preprocessed_words (list): List of preprocessed words

    Returns:
        list: List of bigrams
    """
    if len(preprocessed_words) < 2:
        return []
    return [(f"{preprocessed_words[i]} {preprocessed_words[i + 1]}", f"{preprocessed_words[i+1]} {preprocessed_words[i + 2]}") for i in range(len(preprocessed_words)-2) ]
    # return  [f"{preprocessed_words[i]} {preprocessed_words[i + 1]}" for i in range(len(preprocessed_words)-1) ]

# TODO
def json_to_bigrams(all_data: List[Dict[str, any]], target_key: List[str], throw_key_error: bool = False) -> List[List[Tuple[str, str]]]:

    all_bigrams = []

    # loop through each tweet in the list
    for data in all_data:
        text_part = get_from_nested_key(data, target_key, throw_key_error=throw_key_error) if target_key is not None else data

        if text_part is None:
            continue

        # preprocess the text to get a list of words
        words = preprocess_text(text_part)

        # create bigrams from the preprocessed words
        bigram_words = pair_word_to_bigram(words)
        
        # add the bigrams to the result list
        all_bigrams.append(bigram_words)

    return all_bigrams

# TODO
def tweet_json_file_to_bigrams(json_file_path: str) -> None:
    """
    Reads a JSON file and processes each tweet from the list of tweet data objects
    and prints the list of bigrams

    Args:
        json_file_path (str): Path to JSON file containing list of tweet data objects

    Returns:
        None
    """
    all_data = read_json(json_file_path)
    all_bigrams = json_to_bigrams(all_data)

    print_as_json(all_bigrams)
    print()


def merge_multiple_bigrams_list(
        list_of_bigrams_list: List[List[Tuple[str, str]]], 
        sort: bool = False, 
        remove_duplicates: bool = False
    ) -> List[Tuple[str, str]]:
    """
    Merge a list of bigrams list into one list of bigrams.

    Args:
        list_of_bigrams_list (list): A list of lists of bigrams

    Returns:
        list: A single list of bigrams
    """
    merged_bigrams = []
    # loop through each list of bigrams
    for bigram_list in list_of_bigrams_list:
        if remove_duplicates:
            for bigram in bigram_list:
                if bigram not in merged_bigrams:
                    merged_bigrams.append(bigram)  
        else:
            merged_bigrams.extend(bigram_list)

    return sorted(merged_bigrams) if sort else merged_bigrams


def bigrams_to_weighted_bigrams(bigrams_list: List[Tuple[str, str]], sort: bool = False) -> List[Tuple[str, str, int]]:
    bigrams_count = { bigram : bigrams_list.count(bigram) for bigram in set(bigrams_list) }
    weighted_bigrams = [(bigram[0], bigram[1], count) for bigram, count in bigrams_count.items()]
    return sorted(weighted_bigrams, key=operator.itemgetter(2, 0, 1), reverse=True) if sort else weighted_bigrams


def get_all_words(bigrams_list: List[Tuple[str, str]]) -> List[str]:
    collection = set()
    for word1, word2 in bigrams_list:
        collection.add(word1)
        collection.add(word2)
    return list(collection)
