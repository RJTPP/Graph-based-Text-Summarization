import sys
import os
from typing import Dict, Tuple, List
import string 
import re
import nltk # type: ignore

from pathlib import Path


current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)


from setting import NLTK_PATH

print("Downloading NLTK data...")

nltk.download('stopwords', download_dir=NLTK_PATH)
nltk.download('punkt', download_dir=NLTK_PATH)
nltk.data.path.append(NLTK_PATH)
from nltk.corpus import stopwords # type: ignore

CONTRACTION_WORD = {
    "gonna": "going to",
    "wanna": "want to",
    "gotta": "got to",
    "ain't": "is not",
    "can't": "cannot",
    "won't": "will not",
    "n't": " not",
    "'ll": " will",
    "'re": " are",
    "'ve": " have",
    "'d": " would",
    "'s": " is",
}

ABBREVIATIONS = {
    "acc": "actually",
    "b": "be",
    "u": "you",
    "ur": "your",
    "lol": "laughing out loud",
    "bc": " because",
    "2nd": "second",
    "ops": "operations",
    "govt": "government",
    "rs": "rupees",
    "cr-worth": "crore worth",
    "ryt": "right"
}

ADDITIONAL_STOPWORDS = set([
    "amp"
])


def expand_contractions_custom(text: str) -> str:
    for contraction, expansion in CONTRACTION_WORD.items():
        text = re.sub(r'\b' + contraction + r'\b', expansion, text)
    return text


def replace_abbreviations_str(text: str, replacements: Dict[str, str]) -> str:
    words = text.split()
    replaced_words = [replacements[word.lower()] if word.lower() in replacements else word for word in words]
    return " ".join(replaced_words)


def replace_abbreviations_list(text: List[str], replacements: Dict[str, str]) -> List[str]:
    return [replacements[word.lower()] if word.lower() in replacements else word for word in text]

def remove_punctuations(text: str) -> str:
    return text.translate(str.maketrans('', '', string.punctuation))


def remove_url(text: str) -> str:
    text = re.sub(r'http\S+','',text)
    text = re.sub(r'https\S+','',text)
    return text


def remove_stopwords(words: List[str]) -> List[str]:
    stop_words = set(stopwords.words("english"))
    stop_words = stop_words.union(ADDITIONAL_STOPWORDS)
    return [word for word in words if word.lower() not in stop_words]


def tokenize_split(text: str) -> List[str]:
    return text.split()

def tokenize_nltk(text: str) -> List[str]:
    return nltk.word_tokenize(text)


def preprocess_text(text: str, tokenizer=tokenize_split) -> List[str]:
    # Convert text to lowercase
    text = text.lower()

    # Replace abbreviations
    text = replace_abbreviations_str(text, ABBREVIATIONS)

    # Expand contractions
    text = expand_contractions_custom(text)

    # Remove non-letter characters except spaces
    text = re.sub(r'[^A-Za-z\s]', '', text)


    # Remove punctuations
    text = remove_punctuations(text)

    # Remove URLs
    text = remove_url(text)

    # Split text into words (Tokenize)
    words = tokenizer(text)

    # Remove stopwords
    words = remove_stopwords(words)

    return words
