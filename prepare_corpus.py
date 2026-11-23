import numpy as np
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from corpus_utils import (
    CORPUS_MIN_WORDS,
    CORPUS_PAIRS,
    DATASET_NAME,
    LANGUAGE_NAMES,
    MODEL_NAME,
    TARGET_LANGUAGE_CODES,
    normalize_sentence,
)

rows = []

for target_lang in TARGET_LANGUAGE_CODES:
    config = f"en-{target_lang}"
    print(f"Loading Tatoeba: {config}")

    dataset = load_dataset(
        DATASET_NAME,
        config,
        split=f"train[:{CORPUS_PAIRS}]",
    )

    print("Columns:", dataset.column_names)

    for item in dataset:
        english = normalize_sentence(item.get("english") or "")
        translated = normalize_sentence(item.get("non_english") or "")

        if english and translated:
            rows.append({"language": LANGUAGE_NAMES["en"], "sentence": english})
            rows.append(
                {
                    "language": LANGUAGE_NAMES[target_lang],
                    "sentence": translated,
                }
            )

df = pd.DataFrame(rows)
df["sentence"] = df["sentence"].fillna("").str.strip()
df = df[df["sentence"].str.split().str.len() >= CORPUS_MIN_WORDS]
df = df.drop_duplicates(subset=["sentence"]).reset_index(drop=True)

if df.empty:
    raise ValueError("No sentences loaded. Check the dataset column names.")

df.to_csv("corpus.csv", index=False)

print(f"Saving embeddings for {len(df)} sentences...")
model = SentenceTransformer(MODEL_NAME)
embeddings = model.encode(
    df["sentence"].tolist(),
    normalize_embeddings=True,
    batch_size=64,
    show_progress_bar=True,
)
np.save("embeddings.npy", embeddings)

print("Done! corpus.csv and embeddings.npy updated.")
