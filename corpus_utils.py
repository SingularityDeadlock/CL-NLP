from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
DATASET_NAME = "sentence-transformers/parallel-sentences-tatoeba"

CORPUS_PAIRS = 1000
CORPUS_MIN_WORDS = 4
DEFAULT_N_EVAL = 1000
DEFAULT_SCAN = 8000
MIN_EVAL_WORDS = 3

LANGUAGE_NAMES = {
    "en": "English",
    "de": "German",
    "es": "Spanish",
    "fr": "French",
    "hi": "Hindi",
}
TARGET_LANGUAGE_CODES = ("de", "es", "fr", "hi")


def normalize_sentence(text: Any) -> str:
    text = unicodedata.normalize("NFC", str(text))
    return " ".join(text.strip().split())


def load_indexed_corpus(path: str | Path = "corpus.csv") -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    if "sentence" not in df.columns:
        raise ValueError(f"{path} needs a 'sentence' column.")
    if "language" not in df.columns:
        df["language"] = "?"
    df = df[["language", "sentence"]].dropna().copy()
    df["sentence"] = df["sentence"].map(normalize_sentence)
    df = df[df["sentence"] != ""].reset_index(drop=True)
    return df


def indexed_sentence_set(df: pd.DataFrame) -> set[str]:
    return set(df["sentence"].tolist())


@dataclass(frozen=True)
class EvalPair:
    config: str
    dataset_index: int
    source_lang: str
    target_lang: str
    source: str
    target: str

    @property
    def pair_id(self) -> str:
        return f"{self.config}:{self.dataset_index}"


@dataclass
class HeldoutStats:
    loaded: int = 0
    skipped_empty: int = 0
    skipped_short: int = 0
    skipped_corpus_overlap: int = 0
    skipped_duplicate: int = 0
    kept: int = 0


def load_heldout_pairs(
    target_lang: str,
    *,
    n: int = DEFAULT_N_EVAL,
    start: int = CORPUS_PAIRS,
    scan: int = DEFAULT_SCAN,
    exclude: set[str] | None = None,
    min_words: int = MIN_EVAL_WORDS,
) -> tuple[list[EvalPair], HeldoutStats]:
    config = f"en-{target_lang}"
    excluded = exclude or set()
    stats = HeldoutStats()
    pairs: list[EvalPair] = []
    seen_sources: set[str] = set()
    seen_targets: set[str] = set()
    offset = start

    while len(pairs) < n:
        split = f"train[{offset}:{offset + scan}]"
        dataset = load_dataset(DATASET_NAME, config, split=split)
        if len(dataset) == 0:
            break

        for i, item in enumerate(dataset):
            stats.loaded += 1
            source = normalize_sentence(item.get("english") or "")
            target = normalize_sentence(item.get("non_english") or "")
            dataset_index = offset + i

            if not source or not target:
                stats.skipped_empty += 1
                continue
            if len(source.split()) < min_words or len(target.split()) < min_words:
                stats.skipped_short += 1
                continue
            if source in excluded or target in excluded:
                stats.skipped_corpus_overlap += 1
                continue
            if source in seen_sources or target in seen_targets:
                stats.skipped_duplicate += 1
                continue

            seen_sources.add(source)
            seen_targets.add(target)
            pairs.append(
                EvalPair(
                    config=config,
                    dataset_index=dataset_index,
                    source_lang="en",
                    target_lang=target_lang,
                    source=source,
                    target=target,
                )
            )
            stats.kept += 1
            if len(pairs) == n:
                break

        if len(dataset) < scan:
            break
        offset += scan

    return pairs, stats


def encode_texts(
    model,
    texts: Sequence[str],
    *,
    cache: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    if cache is None:
        return model.encode(
            list(texts),
            normalize_embeddings=True,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    missing = [text for text in dict.fromkeys(texts) if text not in cache]
    if missing:
        new_embeddings = model.encode(
            missing,
            normalize_embeddings=True,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        for text, embedding in zip(missing, new_embeddings):
            cache[text] = embedding

    return np.vstack([cache[text] for text in texts])


def retrieval_ranks(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError("Expected a square query-document score matrix.")
    correct = np.diag(scores)
    greater = (scores > correct[:, None]).sum(axis=1)
    equal = (scores == correct[:, None]).sum(axis=1)
    return greater.astype(np.float64) + (equal + 1) / 2.0


def retrieval_metrics(ranks: Sequence[float]) -> dict[str, float]:
    ranks = np.asarray(ranks, dtype=np.float64)
    return {
        "Recall@1": float(np.mean(ranks <= 1)),
        "Recall@5": float(np.mean(ranks <= 5)),
        "Recall@10": float(np.mean(ranks <= 10)),
        "MRR": float(np.mean(1.0 / ranks)),
        "nDCG@10": float(
            np.mean(np.where(ranks <= 10, 1.0 / np.log2(ranks + 1.0), 0.0))
        ),
        "Top-1": float(np.mean(ranks <= 1)),
        "Top-5": float(np.mean(ranks <= 5)),
        "Top-10": float(np.mean(ranks <= 10)),
    }


def tfidf_scores(queries: Sequence[str], documents: Sequence[str]) -> np.ndarray:
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        sublinear_tf=True,
    )
    document_matrix = vectorizer.fit_transform(documents)
    query_matrix = vectorizer.transform(queries)
    return (query_matrix @ document_matrix.T).toarray()
