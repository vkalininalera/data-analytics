"""Compare BoW/TF-IDF and LSA/LDA on CFPB complaint narratives."""

import argparse
import re
import shutil
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import nltk
import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

SEED = 42
CFPB_URL = "https://files.consumerfinance.gov/ccdb/complaints.csv.zip"
CSV_COLUMNS = {
    "Complaint ID": "complaint_id",
    "Product": "product",
    "Consumer complaint narrative": "raw_text",
    "complaint_id": "complaint_id",
    "product": "product",
    "consumer_complaint_narrative": "raw_text",
}
DOMAIN_STOPWORDS = {"complaint", "company", "account", "told", "said", "received", "xxxx"}
WORDNET_POS = {"J": "a", "V": "v", "N": "n", "R": "r"}
NLTK_RESOURCES = {
    "stopwords": "corpora/stopwords",
    "punkt_tab": "tokenizers/punkt_tab",
    "wordnet": "corpora/wordnet",
    "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--csv", type=Path, help="Existing CFPB CSV; omit to download")
    parser.add_argument(
        "--sample-size", type=int, default=30_000, help="Narratives to sample"
    )
    parser.add_argument(
        "--k-values", type=int, nargs="+", default=[5, 8, 10, 12, 15], help="Topic counts"
    )
    parser.add_argument("--min-df", type=int, default=5, help="Minimum documents per term")
    parser.add_argument(
        "--max-df", type=float, default=0.5, help="Maximum document fraction per term"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("reports"), help="Results directory"
    )
    args = parser.parse_args()
    if args.sample_size < 1:
        parser.error("--sample-size must be positive")
    if min(args.k_values) < 2:
        parser.error("--k-values must contain topic counts of at least 2")
    if args.min_df < 1 or not 0 < args.max_df <= 1:
        parser.error("--min-df must be positive and --max-df must be in (0, 1]")
    if args.csv is not None and not args.csv.is_file():
        parser.error(f"CSV file not found: {args.csv}")
    args.k_values = sorted(set(args.k_values))
    return args


def prepare_nltk() -> None:
    for package, resource_path in NLTK_RESOURCES.items():
        try:
            nltk.data.find(resource_path)
            continue
        except LookupError:
            pass
        try:
            nltk.data.find(f"{resource_path}.zip")
            continue
        except LookupError:
            pass
        print(f"Downloading NLTK resource: {package}...")
        if not nltk.download(package, quiet=True):
            raise SystemExit(f"Could not download {package}. Check your internet connection.")


def download_complaints() -> Path:
    csv_path = Path("data/raw/complaints.csv")
    if csv_path.is_file():
        return csv_path

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    zip_path = csv_path.with_suffix(".csv.zip")
    if not zip_path.is_file():
        print("Downloading the CFPB archive...")
        partial_zip = zip_path.with_suffix(".zip.part")
        urlretrieve(CFPB_URL, partial_zip)
        partial_zip.replace(zip_path)

    print("Extracting complaint data...")
    partial_csv = csv_path.with_suffix(".csv.part")
    with zipfile.ZipFile(zip_path) as archive:
        csv_members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(csv_members) != 1:
            raise ValueError("The CFPB archive must contain exactly one CSV file.")
        with archive.open(csv_members[0]) as source, partial_csv.open("wb") as target:
            shutil.copyfileobj(source, target)
    partial_csv.replace(csv_path)
    return csv_path


def load_sample(csv_path: Path, sample_size: int) -> pd.DataFrame:
    headers = pd.read_csv(csv_path, nrows=0).columns
    selected_columns = [column for column in headers if column in CSV_COLUMNS]
    complaints = pd.read_csv(csv_path, usecols=selected_columns, dtype="string")
    complaints = complaints.rename(columns=CSV_COLUMNS)
    required_columns = ["complaint_id", "product", "raw_text"]
    if not set(required_columns).issubset(complaints.columns):
        raise ValueError("CSV must contain complaint ID, product, and narrative columns.")
    if not complaints.columns.is_unique:
        raise ValueError("CSV contains duplicate complaint columns in different formats.")

    complaints = complaints[required_columns].dropna(subset=["raw_text"])
    complaints = complaints.loc[complaints["raw_text"].str.strip().ne("")]
    complaints["product"] = complaints["product"].fillna("Unknown")
    if complaints.empty:
        raise ValueError("The CSV contains no non-empty complaint narratives.")
    if len(complaints) <= sample_size:
        return complaints.reset_index(drop=True)

    product_shares = complaints["product"].value_counts(normalize=True)
    samples = []
    for product, group in complaints.groupby("product", sort=False):
        group_size = max(1, round(sample_size * product_shares[product]))
        samples.append(group.sample(n=min(group_size, len(group)), random_state=SEED))
    sample = pd.concat(samples)
    if len(sample) < sample_size:
        remaining = complaints.drop(index=sample.index)
        extra = remaining.sample(n=sample_size - len(sample), random_state=SEED)
        sample = pd.concat([sample, extra])
    sample = sample.sample(frac=1, random_state=SEED).head(sample_size)
    return sample.reset_index(drop=True)


def preprocess_text(
    text: str, stop_words: set[str], lemmatizer: WordNetLemmatizer
) -> list[str]:
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+|\S+@\S+", " ", text)
    text = re.sub(r"\bx{2,}(?:/x{2,}/\d{2,4})?\b", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [
        token for token in word_tokenize(text)
        if len(token) >= 3 and token not in stop_words
    ]
    lemmas = []
    for token, tag in nltk.pos_tag(tokens):
        word_type = WORDNET_POS.get(tag[0], "n")
        lemma = lemmatizer.lemmatize(token, word_type)
        if len(lemma) >= 3 and lemma not in stop_words:
            lemmas.append(lemma)
    return lemmas


def preprocess_complaints(complaints: pd.DataFrame) -> pd.DataFrame:
    stop_words = set(stopwords.words("english")) | DOMAIN_STOPWORDS
    lemmatizer = WordNetLemmatizer()
    complaints = complaints.copy()
    complaints["tokens"] = [
        preprocess_text(text, stop_words, lemmatizer) for text in complaints["raw_text"]
    ]
    return complaints.loc[complaints["tokens"].str.len() > 0].reset_index(drop=True)


def get_topic_words(
    model: TruncatedSVD | LatentDirichletAllocation,
    feature_names: np.ndarray,
    number_of_words: int = 10,
) -> list[list[str]]:
    return [
        feature_names[weights.argsort()[::-1][:number_of_words]].tolist()
        for weights in model.components_
    ]


def calculate_coherence(
    topics: list[list[str]], tokens: list[list[str]], dictionary: Dictionary
) -> float:
    unigram_topics = []
    for topic in topics:
        words = []
        for term in topic:
            for word in term.split():
                if word not in words:
                    words.append(word)
        unigram_topics.append(words)
    evaluator = CoherenceModel(
        topics=unigram_topics,
        texts=tokens,
        dictionary=dictionary,
        coherence="c_v",
        processes=1,
    )
    return float(evaluator.get_coherence())


def compare_models(
    tokens: list[list[str]], topic_counts: list[int], min_df: int, max_df: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    if len(tokens) < max(topic_counts):
        raise ValueError("Too few cleaned narratives for the requested topic counts.")
    texts = [" ".join(document) for document in tokens]
    dictionary = Dictionary(tokens)
    vectorizers = {
        "BOW": CountVectorizer(min_df=min_df, max_df=max_df, ngram_range=(1, 2)),
        "TFIDF": TfidfVectorizer(
            min_df=min_df, max_df=max_df, ngram_range=(1, 2), sublinear_tf=True
        ),
    }
    scores = []
    topic_rows = []
    best_lda_coherence = float("-inf")
    primary_topics = None

    for representation, vectorizer in vectorizers.items():
        matrix = vectorizer.fit_transform(texts)
        if matrix.shape[1] <= max(topic_counts):
            raise ValueError("Too few vocabulary terms for the requested topic counts.")
        feature_names = vectorizer.get_feature_names_out()
        for topic_count in topic_counts:
            models = {
                "LSA": TruncatedSVD(
                    n_components=topic_count, n_iter=10, random_state=SEED
                ),
                "LDA": LatentDirichletAllocation(
                    n_components=topic_count,
                    learning_method="online",
                    max_iter=20,
                    random_state=SEED,
                ),
            }
            for method, model in models.items():
                print(f"Fitting {method} with {representation}, {topic_count} topics...")
                model.fit(matrix)
                topic_words = get_topic_words(model, feature_names)
                coherence = calculate_coherence(topic_words, tokens, dictionary)
                if not np.isfinite(coherence):
                    raise ValueError(
                        "Coherence is undefined. Try a larger or more varied sample."
                    )
                configuration = {
                    "model": method, "vectorizer": representation, "topics": topic_count
                }
                perplexity = np.nan
                if isinstance(model, LatentDirichletAllocation):
                    perplexity = model.perplexity(matrix)
                scores.append({
                    **configuration, "coherence_cv": coherence, "perplexity": perplexity
                })
                for topic_number, words in enumerate(topic_words):
                    topic_rows.append({
                        **configuration, "topic": topic_number, "words": ", ".join(words)
                    })

                if (
                    method == "LDA"
                    and representation == "BOW"
                    and coherence > best_lda_coherence
                ):
                    best_lda_coherence = coherence
                    assignments = model.transform(matrix).argmax(axis=1)
                    primary_topics = pd.Series(assignments, name="topic")

    if primary_topics is None:
        raise ValueError("No LDA–BoW model was fitted.")
    score_table = pd.DataFrame(scores).sort_values(
        "coherence_cv", ascending=False, kind="stable"
    )
    return score_table.reset_index(drop=True), pd.DataFrame(topic_rows), primary_topics


def save_results(
    scores: pd.DataFrame,
    topics: pd.DataFrame,
    primary_topics: pd.Series,
    products: pd.Series,
    output_dir: Path,
) -> None:
    lda_bow_scores = scores.loc[
        (scores["model"] == "LDA") & (scores["vectorizer"] == "BOW")
    ]
    selected = lda_bow_scores.iloc[0]
    topic_count = int(selected["topics"])
    topic_numbers = pd.Index(range(topic_count), name="topic")
    counts = primary_topics.value_counts().reindex(topic_numbers, fill_value=0)
    prevalence = counts.to_frame("documents")
    prevalence["share"] = prevalence["documents"] / len(primary_topics)
    by_product = pd.crosstab(products.reset_index(drop=True), primary_topics)
    by_product = by_product.reindex(columns=topic_numbers, fill_value=0)

    output_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output_dir / "model_scores.csv", index=False)
    topics.to_csv(output_dir / "topic_words.csv", index=False)
    prevalence.to_csv(output_dir / "topic_prevalence.csv")
    by_product.to_csv(output_dir / "topic_by_product.csv")

    selected_topics = topics.loc[
        (topics["model"] == "LDA")
        & (topics["vectorizer"] == "BOW")
        & (topics["topics"] == topic_count)
    ]
    with (output_dir / "topics.txt").open("w", encoding="utf-8") as output:
        output.write(
            f"Selected model: LDA with BoW, k={topic_count}, "
            f"c_v={selected['coherence_cv']:.4f}\n\n"
        )
        for topic in selected_topics.itertuples(index=False):
            output.write(f"Topic {topic.topic}: {topic.words}\n")


def main() -> None:
    args = parse_arguments()
    prepare_nltk()
    csv_path = args.csv if args.csv is not None else download_complaints()
    print(f"Loading narratives from {csv_path}...")
    complaints = load_sample(csv_path, args.sample_size)
    print(f"Preprocessing {len(complaints):,} narratives...")
    complaints = preprocess_complaints(complaints)
    scores, topics, primary_topics = compare_models(
        complaints["tokens"].tolist(), args.k_values, args.min_df, args.max_df
    )
    save_results(scores, topics, primary_topics, complaints["product"], args.output_dir)
    best = scores.iloc[0]
    print(f"Finished: {len(complaints):,} cleaned narratives.")
    print(
        f"Highest coherence: {best['model']}-{best['vectorizer']}, "
        f"k={int(best['topics'])}, c_v={best['coherence_cv']:.4f}."
    )
    print(f"Results: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
