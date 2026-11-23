from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from corpus_utils import MODEL_NAME, encode_texts, normalize_sentence

USER_AGENT = "cross-lingual-search-phase2/1.0 (local educational project)"
API_URL = "https://en.wikipedia.org/w/api.php"
MIN_WORDS = 8
MAX_WORDS = 40
TARGET_TOTAL_MIN = 100
TARGET_TOTAL_MAX = 200

TOPIC_ARTICLES = {
    "government": [
        "Government",
        "Democracy",
        "Parliament",
        "United Nations",
        "Constitution",
        "Election",
        "Public policy",
        "Supreme court",
        "Legislature",
        "Civil service",
        "Tax",
        "Diplomacy",
        "Separation of powers",
        "Federal government of the United States",
        "Local government",
        "Monarchy",
        "Political party",
        "Human rights",
        "International law",
        "Public administration",
    ],
    "technology": [
        "Technology",
        "Artificial intelligence",
        "Computer",
        "Internet",
        "Software",
        "Robotics",
        "Semiconductor",
        "Telecommunication",
        "Cloud computing",
        "Machine learning",
        "Electric vehicle",
        "Renewable energy",
        "Computer security",
        "Smartphone",
        "World Wide Web",
        "Information technology",
        "Automation",
        "Data center",
        "Quantum computing",
        "Digital electronics",
    ],
    "science": [
        "Science",
        "Physics",
        "Chemistry",
        "Biology",
        "Astronomy",
        "Climate change",
        "Genetics",
        "Quantum mechanics",
        "Evolution",
        "Scientific method",
        "Ecology",
        "Neuroscience",
        "Geology",
        "Particle physics",
        "Microbiology",
        "Scientific theory",
        "Experiment",
        "Researcher",
        "Natural science",
        "Earth science",
    ],
    "health": [
        "Health",
        "Medicine",
        "Public health",
        "Vaccine",
        "Hospital",
        "Cancer",
        "Diabetes",
        "Mental health",
        "Nutrition",
        "Surgery",
        "Antibiotic",
        "Cardiovascular disease",
        "Epidemiology",
        "Primary care",
        "World Health Organization",
        "Disease",
        "Immunization",
        "Medical diagnosis",
        "Physical therapy",
        "Health care",
    ],
}

TOPIC_KEYWORDS = {
    "government": [
        "government",
        "law",
        "vote",
        "election",
        "parliament",
        "minister",
        "president",
        "policy",
        "court",
        "senate",
        "congress",
        "tax",
        "diplomat",
        "treaty",
        "military",
        "war",
        "peace",
        "citizen",
        "democracy",
        "republic",
        "political",
        "politics",
        "official",
        "authority",
        "constitution",
        "legislature",
        "human rights",
        "state",
        "public",
    ],
    "technology": [
        "technology",
        "computer",
        "software",
        "internet",
        "digital",
        "data",
        "robot",
        "artificial intelligence",
        "ai",
        "algorithm",
        "network",
        "device",
        "phone",
        "machine",
        "engineer",
        "program",
        "cyber",
        "satellite",
        "electric",
        "energy",
        "online",
        "website",
        "system",
        "semiconductor",
        "telecommunication",
        "automation",
        "computing",
        "electronics",
    ],
    "science": [
        "science",
        "scientist",
        "research",
        "experiment",
        "physics",
        "chemistry",
        "biology",
        "astronomy",
        "space",
        "planet",
        "molecule",
        "atom",
        "gene",
        "evolution",
        "climate",
        "environment",
        "species",
        "theory",
        "discovery",
        "laboratory",
        "study",
        "scientific",
        "ecology",
        "neuroscience",
        "geology",
        "microbiology",
    ],
    "health": [
        "health",
        "doctor",
        "hospital",
        "medicine",
        "disease",
        "patient",
        "treatment",
        "vaccine",
        "virus",
        "cancer",
        "surgery",
        "nurse",
        "medical",
        "symptom",
        "therapy",
        "mental",
        "nutrition",
        "exercise",
        "diet",
        "clinic",
        "drug",
        "blood",
        "infection",
        "care",
        "diagnosis",
        "antibiotic",
        "epidemiology",
        "world health organization",
    ],
}

ABBREVIATIONS = [
    "U.S.",
    "U.K.",
    "U.N.",
    "E.U.",
    "e.g.",
    "i.e.",
    "Dr.",
    "Mr.",
    "Mrs.",
    "Ms.",
    "Prof.",
    "St.",
    "No.",
    "vs.",
    "Inc.",
    "Ltd.",
    "Co.",
    "Jr.",
    "Sr.",
    "Gen.",
    "Sen.",
    "Rep.",
    "Gov.",
    "Ave.",
    "Blvd.",
    "Est.",
    "approx.",
]

BAD_PATTERNS = re.compile(
    r"(may refer to|see also|further information|this article|this section|"
    r"citation needed|wikipedia|wikimedia|for other uses|is a list of|"
    r"the following|the word|the term|etymology)",
    re.IGNORECASE,
)


def configure_stdout() -> None:
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass


def fetch_extracts(titles: list[str]) -> list[tuple[str, str]]:
    params = urllib.parse.urlencode(
        {
            "action": "query",
            "prop": "extracts",
            "explaintext": 1,
            "exintro": 1,
            "exlimit": "max",
            "redirects": 1,
            "titles": "|".join(titles),
            "format": "json",
            "formatversion": 2,
        }
    )
    request = urllib.request.Request(
        f"{API_URL}?{params}", headers={"User-Agent": USER_AGENT}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        data = json.load(response)

    pages = data.get("query", {}).get("pages", [])
    if isinstance(pages, dict):
        pages = list(pages.values())

    extracts = []
    for page in pages:
        if page.get("missing"):
            continue
        extract = (page.get("extract") or "").strip()
        if extract:
            extracts.append((page.get("title", ""), extract))
    return extracts


def split_sentences(text: str) -> list[str]:
    protected = text.replace("\n", " ")
    for index, abbreviation in enumerate(ABBREVIATIONS):
        protected = protected.replace(abbreviation, f"ABBR{index}§")

    parts = re.split(r"(?<=[.!?])\s+", protected)
    restored = []
    for part in parts:
        for index, abbreviation in enumerate(ABBREVIATIONS):
            part = part.replace(f"ABBR{index}§", abbreviation)
        restored.append(part.strip())
    return restored


def has_keyword(sentence: str, keywords: list[str]) -> bool:
    lower = sentence.lower()
    return any(
        re.search(rf"(?<!\w){re.escape(keyword)}(?!\w)", lower)
        for keyword in keywords
    )


def is_good_sentence(sentence: str, keywords: list[str]) -> bool:
    if not sentence or not sentence[0].isupper():
        return False
    if not re.search(r"[.!?]$", sentence):
        return False
    if not (MIN_WORDS <= len(sentence.split()) <= MAX_WORDS):
        return False
    if re.search(r"[\[\]{}<>|]", sentence):
        return False
    if BAD_PATTERNS.search(sentence):
        return False
    lower = sentence.lower()
    if "http://" in lower or "https://" in lower or "www." in lower:
        return False
    return has_keyword(sentence, keywords)


def collect_topic_candidates(
    topic: str,
) -> list[tuple[str, list[str]]]:
    extracts = fetch_extracts(TOPIC_ARTICLES[topic])
    per_article = []
    for title, text in extracts:
        sentences = []
        for raw in split_sentences(text):
            sentence = normalize_sentence(raw).strip(" \"'“”")
            if is_good_sentence(sentence, TOPIC_KEYWORDS[topic]):
                sentences.append(sentence)
        if sentences:
            per_article.append((title, sentences))
    return per_article


def select_round_robin(
    per_article: list[tuple[str, list[str]]],
    seen: set[str],
    limit: int,
) -> list[tuple[str, str]]:
    selected = []
    depth = 0
    while len(selected) < limit:
        added = False
        for title, sentences in per_article:
            if depth >= len(sentences):
                continue
            sentence = sentences[depth]
            key = sentence.lower()
            if key in seen:
                continue
            seen.add(key)
            selected.append((title, sentence))
            added = True
            if len(selected) >= limit:
                break
        if not added:
            break
        depth += 1
    return selected


def build_additions(
    per_topic: int, existing_sentences: set[str]
) -> tuple[pd.DataFrame, dict[str, int]]:
    seen = {sentence.lower() for sentence in existing_sentences}
    rows = []
    counts = {}
    for topic in TOPIC_ARTICLES:
        candidates = collect_topic_candidates(topic)
        selected = select_round_robin(candidates, seen, per_topic)
        counts[topic] = len(selected)
        for title, sentence in selected:
            rows.append(
                {
                    "language": "English",
                    "sentence": sentence,
                    "topic": topic,
                    "source_title": title,
                }
            )
    return pd.DataFrame(rows), counts


def back_up(path: Path, backup_dir: Path, label: str, stamp: str) -> Path:
    backup_dir.mkdir(parents=True, exist_ok=True)
    destination = backup_dir / f"{path.name}.pre_wikipedia_{stamp}.bak"
    shutil.copy2(path, destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Append topic-rich English Wikipedia sentences to the corpus."
    )
    parser.add_argument("--corpus", default="corpus.csv")
    parser.add_argument("--embeddings", default="embeddings.npy")
    parser.add_argument("--backup-dir", default="backups")
    parser.add_argument(
        "--per-topic",
        type=int,
        default=40,
        help="Maximum number of sentences to add per topic.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and report candidates without modifying corpus or embeddings.",
    )
    args = parser.parse_args()

    configure_stdout()
    corpus_path = Path(args.corpus)
    embeddings_path = Path(args.embeddings)

    corpus = pd.read_csv(corpus_path)
    if "sentence" not in corpus.columns:
        raise ValueError(f"{corpus_path} needs a 'sentence' column.")
    if "language" not in corpus.columns:
        corpus["language"] = "English"
    if corpus["sentence"].duplicated().any():
        raise ValueError("Existing corpus already contains duplicate sentences.")

    additions, counts = build_additions(
        args.per_topic, set(corpus["sentence"].tolist())
    )
    total = len(additions)

    print("Wikipedia topic enrichment")
    print(f"Existing corpus: {len(corpus):,} sentences from {corpus_path}")
    print(f"Target per topic: {args.per_topic}")
    for topic, count in counts.items():
        print(f"  {topic:<11} {count:>3} sentences")
    print(f"Candidate additions: {total}")

    if total == 0:
        print("No new sentences to add; existing corpus already covers these candidates.")
        return

    if total < TARGET_TOTAL_MIN or total > TARGET_TOTAL_MAX:
        raise RuntimeError(
            f"Expected {TARGET_TOTAL_MIN}-{TARGET_TOTAL_MAX} additions, got {total}. "
            "Adjust --per-topic or the article list."
        )

    if args.dry_run:
        print("\nDry run only; no files modified.")
        for topic in TOPIC_ARTICLES:
            print(f"\n[{topic}]")
            for sentence in additions.loc[
                additions["topic"] == topic, "sentence"
            ].head(3):
                print(f"  - {sentence}")
        return

    embeddings = np.load(embeddings_path)
    if embeddings.shape[0] != len(corpus):
        raise ValueError(
            f"Corpus/embeddings mismatch: {len(corpus)} rows vs "
            f"{embeddings.shape[0]} embeddings."
        )

    model = SentenceTransformer(MODEL_NAME)
    new_embeddings = encode_texts(
        model, additions["sentence"].tolist()
    )

    new_corpus = pd.concat(
        [corpus[["language", "sentence"]], additions[["language", "sentence"]]],
        ignore_index=True,
    )
    if new_corpus["sentence"].duplicated().any():
        raise ValueError("Duplicate sentences detected after adding Wikipedia rows.")

    new_matrix = np.vstack([embeddings, new_embeddings])
    if new_matrix.shape[0] != len(new_corpus):
        raise ValueError(
            f"New corpus/embeddings mismatch: {len(new_corpus)} rows vs "
            f"{new_matrix.shape[0]} embeddings."
        )
    if not np.array_equal(embeddings, new_matrix[: len(embeddings)]):
        raise ValueError("Existing embedding rows changed unexpectedly.")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_dir = Path(args.backup_dir)
    corpus_backup = back_up(corpus_path, backup_dir, "corpus", stamp)
    embeddings_backup = back_up(embeddings_path, backup_dir, "embeddings", stamp)

    temp_corpus = corpus_path.with_suffix(corpus_path.suffix + ".tmp")
    temp_embeddings = embeddings_path.with_suffix(embeddings_path.suffix + ".tmp.npy")
    new_corpus.to_csv(temp_corpus, index=False, encoding="utf-8")
    np.save(temp_embeddings, new_matrix)
    os.replace(temp_corpus, corpus_path)
    os.replace(temp_embeddings, embeddings_path)

    additions.to_csv(
        "wikipedia_additions.csv", index=False, encoding="utf-8-sig"
    )

    print("\nFiles updated")
    print(f"  {corpus_path}: {len(corpus):,} -> {len(new_corpus):,} rows")
    print(
        f"  {embeddings_path}: {embeddings.shape} -> {new_matrix.shape}"
    )
    print(f"  wikipedia_additions.csv: {total} topic-labelled rows")
    print("\nBackups")
    print(f"  {corpus_backup}")
    print(f"  {embeddings_backup}")
    print("\nVerified")
    print(f"  corpus rows == embedding rows: {len(new_corpus)}")
    print(f"  duplicate sentences: {int(new_corpus['sentence'].duplicated().sum())}")
    print(f"  original rows preserved: {new_corpus.iloc[: len(corpus)].equals(corpus[['language', 'sentence']].reset_index(drop=True))}")


if __name__ == "__main__":
    main()
