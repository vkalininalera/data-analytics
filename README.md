# Complaint Topic Extraction

This project identifies prevalent themes in consumer complaint narratives from the Consumer Financial Protection
Bureau (CFPB). It compares two text representations—Bag-of-Words (BoW) and TF-IDF—and two topic-modelling
techniques—Latent Semantic Analysis (LSA) and Latent Dirichlet Allocation (LDA).

## Requirements

- Python 3.11
- Internet access for the initial data and NLTK resource downloads
- At least 11 GB of free disk space when downloading the complete CFPB dataset

The CFPB archive was approximately 1.39 GB and expanded to about 8.8 GB when the reference analysis was conducted.
Dataset size and runtime may change as new complaints are added.

## Installation

From the repository directory, create a virtual environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Windows, activate the environment with:

```text
.venv\Scripts\activate
```

Missing NLTK language resources are downloaded automatically. Installed resources are reused on subsequent runs.

## Usage

Run the complete analysis with the default settings:

```bash
python analysis.py
```

If the CFPB data are not available locally, the script downloads and extracts them into `data/raw/`. The default
analysis uses a sample of 30,000 documents, seed 42, and topic counts of 5, 8, 10, 12, and 15.

The source is the [CFPB complaint archive](https://files.consumerfinance.gov/ccdb/complaints.csv.zip).
Sampling approximately preserves the proportions of product categories. Retain the downloaded CSV to repeat an
analysis on the same data: the public database changes, so the same seed alone does not guarantee identical results.

To analyse an existing CFPB-format CSV or perform a shorter run:

```bash
python analysis.py --csv path/to/complaints.csv --sample-size 1000 --k-values 5 8
```

The CSV must contain `Complaint ID`, `Product`, and `Consumer complaint narrative`. Lowercase headers with
underscores are also accepted. Rows without narratives are excluded; missing product labels are grouped as `Unknown`.
A smaller sample reduces modelling time, but the script still reads the input CSV before sampling.

View all command-line options with:

```bash
python analysis.py --help
```

## Analysis workflow

1. Load complaint narratives and create a deterministic sample stratified by product.
2. Remove URLs, email addresses, numbers, punctuation, redaction marks, and stopwords.
3. Tokenise the text and lemmatise each token using its part-of-speech tag.
4. Create BoW and sublinear TF-IDF matrices containing unigrams and bigrams.
5. Fit LSA and LDA models for each requested number of topics.
6. Calculate `c_v` coherence and LDA perplexity, and export topic words for manual interpretation.
7. Use the most coherent LDA–BoW model to calculate topic prevalence and topic distribution by product.

## Outputs

Results are written to `reports/`:

| File                   | Contents                                                       |
|------------------------|----------------------------------------------------------------|
| `model_scores.csv`     | Coherence and perplexity results for all model configurations  |
| `topic_words.csv`      | Top words for every topic in every model configuration         |
| `topics.txt`           | Top words from the selected LDA–BoW model                      |
| `topic_prevalence.csv` | Document count and share for each topic                        |
| `topic_by_product.csv` | Number of documents assigned to each topic by product category |

Downloaded data and generated outputs are excluded from version control.

Topic numbers start at zero. Prevalence assigns each document to its highest-probability topic in the selected
LDA–BoW model; it does not count every topic mentioned in a document. Topics with no assigned documents remain
in the tables with a count of zero. Reusing an output directory replaces its generated result files.
