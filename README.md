# Graph-based Text Summarization

[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python-based text summarization tool that leverages graph-based techniques for generating concise and meaningful summaries. This project combines PageRank and TrustRank algorithms with ROUGE validation to produce high-quality text summaries.


## Table of Contents

- [Quick Start](#quick-start)
- [Requirements and Dependencies](#requirements-and-dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Dataset Structure](#dataset-structure)
- [Validation File Structure](#validation-file-structure)
- [Output](#output)
- [Workflow Overview](#workflow-overview)
- [Project Directory Structure](#project-directory-structure)
- [License](#license)
- [Contributors](#contributors)


## Quick Start

1. Ensure that Python (version 3.6 or later) and pip are installed on your system.

2. Clone the repository and navigate to the directory:
```bash
git clone https://github.com/RJTPP/Graph-based-Text-Summarization.git &&
cd Graph-based-Text-Summarization
```

3. Run the setup script. This will create a virtual environment and install the required dependencies.
```bash
python3 setup.py
```

4. Activate the virtual environment.
 - For Linux/macOS
```bash
source venv/bin/activate
```
- For Windows
```bash
venv\Scripts\activate
```

5. Configure the project by editing `config.json`
   - Set paths for datasets and outputs.
   - Define algorithm parameters and workflow options.
   - For more details, see [Configuration](#configuration) section.

6. Add your dataset to the `dataset/` directory. 
   - Ensure the dataset is in JSON format and contains the structure specified in the [Dataset Structure](#dataset-structure) section.

7. Run the main script. Optional arguments can be provided. See [Options](#options) section for details.
```bash
python3 main.py [OPTIONAL_OPTIONS]
```

8. View the results in the `output/` directory. See [Output](#output) section for an example.


> [!NOTE]
> For further details, please refer to the [Installation](#installation) and [Usage](#usage) sections.


## Requirements and Dependencies

This project was developed using Python 3.6 and is tested to be compatible with Python 3.6 through 3.12. It is expected to work with newer versions of Python. The project requires the following dependencies:
- `networkx`
- `matplotlib`
- `scipy`
- `regex`
- `tqdm`
- `nltk`
- `pathlib`
- `numpy`
- `orjson`
- `prettytable`

> [!NOTE]
> For a complete list of dependencies, see [`requirements.txt`](requirements.txt)

## Installation

1. Clone the repository.
```bash
git clone https://github.com/RJTPP/Graph-based-Text-Summarization.git &&
cd Graph-based-Text-Summarization
```

2. Run `setup.py`,This will create a virtual environment and install the required dependencies.
```bash
python3 setup.py
```


### Manual Installation (Optional)  
If you prefer to install the dependencies manually, follow these steps:

1. Create a virtual environment.
```bash
python3 -m venv venv
```

2. Activate the virtual environment.
 - For Linux/macOS
```bash
source venv/bin/activate
```
- For Windows
```bash
venv\Scripts\activate
```

3. Upgrade `pip`.
```bash
pip install --upgrade pip
```

4. Install the required dependencies.
```bash
pip install -r requirements.txt
```

## Usage

To execute the project, follow these steps:

### 1.	Add Dataset:
- Ensure your dataset files are in the directory specified by the `dataset_dir` field in `config.json`. The default directory is `dataset/`.

### 2. Activate the virtual environment:

 - For Linux/macOS
```bash
source venv/bin/activate
```
- For Windows
```bash
venv\Scripts\activate
```

### 3.	Run the Main Script:

```bash
python3 main.py [OPTIONAL_OPTIONS]
```

### Options:

- `-f`, `--files`: Specify one or more JSON files to process.

```bash
python3 main.py -f file1.json file2.json
```


- `-e`, `--exclude`: Exclude one or more JSON files from processing.

```bash
python3 main.py -e file1.json
```

- `-q`, `--quiet`: Suppress output.

> [!NOTE]
> If no options are provided, the script processes all JSON files in the dataset directory by default.


## Configuration
The project can be configured through `config.json`, which contains:

### Path Configuration
  - `cached_dir`: Directory for storing cache data.
  - `dataset_dir`: Directory containing input JSON datasets.
  - `output_dir`: Directory where processed results and outputs will be stored.
  - `validation_file`: File for storing validation results. If set to an empty string (`""`), validation will not be performed.

### Algorithm Parameters
  - `calculation_threshold`: Convergence threshold for iterative calculations like PageRank and TrustRank.
  - `max_calculation_iteration`: Maximum number of iterations for the scoring algorithms.
  - `trustrank_bias_amount`: Number of nodes or elements to bias in TrustRank, chosen from most scored from inverse PageRank.
  - `max_summarize_length`:  Maximum number of iterations for TrustRank algorithm. This will also be the maximum number of nodes or elements to summarize.
  - `trustrank_filter_threshold`: Threshold for filtering nodes or elements in TrustRank algorithm.

### Workflow Options
  - `stop_on_error`: If true, stops the execution if an error occurs.
  - `use_pagerank_library`: Set to true to use a library-based PageRank implementation (`networkx`) or  false for the custom implementation.
  - `output_graph`: If true, saves the generated graphs as files in the output directory.
  - `show_graph`: If true, displays graphs during execution (requires GUI).

### Target Data Keys
  - `target_data_key`: Specifies which keys from the JSON dataset to process. See [**Dataset Structure**](#dataset-structure) for details.


### Example Configuration
```json
{
    "path": {
        "cached_dir": "caches",
        "dataset_dir": "dataset",
        "output_dir": "output", 
        "validation_file": "validation/validation.json"
    },
    "parameters": {
        "calculation_threshold": 1e-5,
        "max_calculation_iteration": 200,
        "trustrank_bias_amount": 5,
        "max_summarize_length": 20,
        "trustrank_filter_threshold": 1e-3
    },
    "options": {
        "stop_on_error": false,
        "use_networkx_library": false,
        "output_graph": true,
        "show_graph": false
    },
    "target_data_key": [
        "full_text"
    ]
}
```

> [!TIP]
> To print the currently configured paths in `config.json`, you can run the `verify_path.py` script.

## Dataset Structure

Input datasets must be JSON files structured as **an array of dictionaries or an array of strings** to ensure compatibility with the workflow. Below is an example of the expected dataset format:

### Example 1: Array of Dictionaries
```json
[
    {
        "id"  : "001",
        "data": {
            "full_text" : "This is a sample text.",
            "author"    : "author_name",
            "date"      : "2024-01-01"
        }
    },
    {
        "id"  : "002",
        "data": {
            "full_text" : "This is another sample text.",
            "author"    : "another_author_name",
            "date"      : "2024-01-02"
        }
    }
]
```

- In this example, the `data.full_text` field contains the text to be processed.
- target_data_key : `["data", "full_text"]`

### Example 2: Array of Strings
```json
[
    "This is a sample text.",
    "This is another sample text."
]
```
- target_data_key: Leave as an empty array `[ ]`.


## Validation File Structure

Validation files are JSON files located in the `validation/` directory (or configured path), containing reference summaries for comparison. The structure is as follows:

```json
  {
      "data1.json": "This is a reference summary for the first text.",
      "data2.json": "This is a reference summary for the second text."
  }
```

> [!NOTE]
> If the `validation_file` field in `config.json` is set to an empty string (`""`), validation will not be performed.


## Output

The project generates several output files in the `output/` directory (or configured path):

- `graph_{name}.json`: Represents the bigram graph edges, including relationships and weights.
- `filtered_graph_{name}.json`: A refined version of the bigram graph with nodes filtered based on TrustRank scores.
- `inverse_pagerank_{name}.json`: Contains the calculated inverse PageRank scores for graph nodes.
- `trust_rank_{name}.json`: Contains the TrustRank scores for graph nodes.
- `summary_{name}.json`: Generated text summaries derived from BFS paths in the graph.
- `validation_{name}.json`: ROUGE-1, ROUGE-2, and ROUGE-L scores comparing generated summaries to reference summaries.

### Example Outputs

`graph_sample.json` and `filtered_graph_sample.json`:
```js
[
  [
    "word1 word2",  // Source Bigram Node
    "word2 word3",  // Target Bigram Node
    2               // Weight (Frequency)
  ],
  ...
]
```
<br>


`inverse_pagerank_sample.json` and `trust_rank_sample.json`:
```js
[
  ["word1 word2", 0.70], // [Bigram Node, Score]
  ["word2 word3", 0.15],
  ...
]
```

`summary_{name}.json`
```js
[
  "Summary of the text.",
  "Another summary of the text.",
  ...
]
```
<br>

`validation_{name}.json`

```js
[
  {
    "text": "Summary of the text.",
    "rouge1": {
      "precision": 1.0,
      "recall"   : 1.0,
      "f-measure": 1.0
    },
    "rouge2": {
      "precision": 1.0,
      "recall"   : 1.0,
      "f-measure": 1.0
    },
    "rougeL": {
      "precision": 1.0,
      "recall"   : 1.0,
      "f-measure": 1.0
    }
  },
  ...
]
```


## Workflow Overview


This project processes textual data through six key stages, culminating in the generation of summarized texts:

### 1. Text Preprocessing

The text preprocessing module applies several cleaning techniques:
  - Converts text to lowercase
  - Expands contractions and replaces slang
  - Removes non-alphabetic characters, punctuation, and URLs
  - Removes stopwords

### 2. Bigram Graph Generation
  - Converts processed text into bigrams
  - Generates weighted bigrams and graphs (library-based or custom implementation depending on the configuration)
  - Optionally visualizes graphs using matplotlib

### 3. Score Calculation
  - Calculates PageRank using either library-based or customized algorithm depending on configuration
  - Calculates TrustRank using customized algorithm based on seeded bias.

### 4. BFS Tree Generation
  - Converts the bigram graph to a BFS tree based on TrustRank-filtered nodes.
  - Generates all possible paths from the root to leaf nodes, forming candidate summaries.

### 5. Validation
  - Uses reference summaries from the `validation_file` to evaluate generated summaries.
  - Calculates ROUGE scores to measure the quality of summarization.

### 6. Results Output

  - Processed results are saved to the output directory as defined in the configuration file.
  - Following calculations will be saved:
    - Bigram Graph
    - Filtered Bigram Graph
    - Inverse PageRank Scores
    - TrustRank Scores
    - Summarized Texts
    - ROUGE scores

## Project Directory Structure

```
work/
├── caches/
│   └── ... (Cache files)
│   
├── dataset/
│   └── ... (Put your dataset here)
│
├── helper_script/              # Utility scripts
│   ├── __init__.py
│   ├── file_reader_helper.py   # File related helper functions
│   ├── func_timer.py           # Timer for monitoring function runtime
│   └── json_helper.py          # JSON helper functions
│
├── modules_script/             # Core processing modules
│   ├── __init__.py  
│   ├── m_bfs_tree.py           # Convert graph to BFS tree
│   ├── m_graph_custom.py       # Graph generation for calculating inverse pagerank (custom implementation)
│   ├── m_graph_nx.py           # Graph generation for calculating inverse pagerank (networkx library)
│   ├── m_preprocess_text.py    # Text preprocessing logic
│   ├── m_process_text.py       # Text to bigrams logic
│   └── m_rouge_score.py        # ROUGE score calculation
│
├── output/                     # Output directory  
│   └── ...
│
├── validation/                 
│   └── validation.json         # JSON file for validating output
│
├── config.json                 # Configuration
├── main.py                     # Main script
├── requirements.txt            # Dependencies
├── settings.py                 # Handle settings from config.json 
├── setup.py                    # Setup script
├── verify_path.py              # Verify path settings
├── venv                        # Virtual environment
└── LICENSE                     # License file
```

## License

This project is released under the [MIT License](LICENSE).

You are free to use, modify, and distribute this software under the terms of the MIT License. See the LICENSE file for detailed terms and conditions.

## Contributors

Rajata Thamcharoensatit ([@RJTPP](https://github.com/RJTPP))
