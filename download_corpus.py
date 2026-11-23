from datasets import load_dataset
import pandas as pd

# Load 1,000 English-German sentence pairs
dataset = load_dataset(
    "Helsinki-NLP/opus-100",
    "de-en",
    split="train[:1000]"
)

# Convert into a CSV file
df = pd.DataFrame({
    "english": [row["en"] for row in dataset["translation"]],
    "german": [row["de"] for row in dataset["translation"]]
})

df.to_csv("opus100_sample.csv", index=False)

print(df.head())
print("Total sentence pairs:", len(df))
