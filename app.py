import os
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

LANGUAGE_COLORS = {
    "English": "blue",
    "Spanish": "orange",
    "French": "violet",
    "German": "green",
    "Hindi": "red",
}

SAMPLE_CORPUS = [
    ("English", "The stock market fell sharply today."),
    ("English", "I enjoy long walks on the beach with my dog."),
    ("English", "Scientists discovered a new species of frog."),
    ("English", "The government will vote on the new law tomorrow."),
    ("English", "Climate change is raising sea levels around the world."),
    ("English", "She is learning to cook traditional Italian pasta."),
    ("Spanish", "La bolsa de valores cayó bruscamente hoy."),
    ("Spanish", "Disfruto de largos paseos por la playa con mi perro."),
    ("Spanish", "Los científicos descubrieron una nueva especie de rana."),
    ("Spanish", "El cambio climático está elevando el nivel del mar."),
    ("French", "Le marché boursier a fortement chuté aujourd'hui."),
    ("French", "J'aime faire de longues promenades sur la plage avec mon chien."),
    ("French", "Le gouvernement votera la nouvelle loi demain."),
    ("French", "Elle apprend à cuisiner des pâtes italiennes traditionnelles."),
    ("German", "Der Aktienmarkt ist heute stark gefallen."),
    ("German", "Ich genieße lange Spaziergänge am Strand mit meinem Hund."),
    ("German", "Forscher haben eine neue Froschart entdeckt."),
    ("German", "Der Klimawandel lässt den Meeresspiegel weltweit steigen."),
    ("Hindi", "आज शेयर बाज़ार में भारी गिरावट आई।"),
    ("Hindi", "मुझे अपने कुत्ते के साथ समुद्र तट पर लंबी सैर करना पसंद है।"),
    ("Hindi", "वैज्ञानिकों ने मेंढक की एक नई प्रजाति की खोज की।"),
    ("Hindi", "जलवायु परिवर्तन से दुनिया भर में समुद्र का स्तर बढ़ रहा है।"),
    ("Hindi", "सरकार कल नए कानून पर मतदान करेगी।"),
]


@st.cache_resource(show_spinner="Loading model (first run downloads ~470 MB)...")
def load_model():
    return SentenceTransformer(MODEL_NAME)


@st.cache_data(show_spinner="Embedding corpus...")
def embed_corpus(sentences: tuple):
    model = load_model()
    # normalized embeddings
    return model.encode(list(sentences), normalize_embeddings=True, batch_size=64)


def load_user_corpus(file) -> pd.DataFrame:
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
        if "sentence" not in df.columns:
            st.error("CSV needs a 'sentence' column.")
            st.stop()
        if "language" not in df.columns:
            df["language"] = "?"
        return df[["language", "sentence"]].dropna()
    lines = [l.strip() for l in file.read().decode("utf-8").splitlines() if l.strip()]
    return pd.DataFrame({"language": "?", "sentence": lines})


st.set_page_config(
    page_title="Cross-lingual semantic search",
    page_icon=":material/travel_explore:",
)

st.title("Cross-lingual semantic search", icon=":material/travel_explore:")
st.caption(
    ":material/translate: Search in any language. Get results in every language, ranked by meaning."
)

with st.sidebar:
    st.header("Corpus", icon=":material/database:")
    upload = st.file_uploader(
        "Upload your own corpus",
        type=["txt", "csv"],
        help="Use a plain .txt file (one sentence per line) or a .csv with a 'sentence' column. A 'language' column is optional.",
    )
    top_k = st.slider("Number of results", 1, 20, 5)

precomputed = None
if upload:
    df = load_user_corpus(upload)
elif os.path.exists("corpus.csv") and os.path.exists("embeddings.npy"):
    df = pd.read_csv("corpus.csv")
    precomputed = np.load("embeddings.npy")
else:
    df = pd.DataFrame(SAMPLE_CORPUS, columns=["language", "sentence"])
    st.caption("Using the tiny sample corpus. Run `python build_corpus.py` to build the real one.")

with st.sidebar:
    langs = sorted(df["language"].unique())
    lang_filter = st.multiselect("Show results in", langs, default=langs)
    st.caption(
        f":material/dataset: {len(df):,} sentences · {len(langs)} languages indexed"
    )

query = st.text_input(
    "Query",
    placeholder="e.g. el mercado bursátil cayó  /  शेयर बाज़ार गिरा",
    icon=":material/search:",
    label_visibility="collapsed",
)

if query:
    with st.spinner("Searching across languages..."):
        model = load_model()
        corpus_emb = (
            precomputed
            if precomputed is not None
            else embed_corpus(tuple(df["sentence"]))
        )
        q_emb = model.encode([query], normalize_embeddings=True)[0]
        scores = corpus_emb @ q_emb

    out = df.assign(score=scores)
    out = out[out["language"].isin(lang_filter)]
    out = out.sort_values("score", ascending=False).head(top_k)

    if out.empty:
        st.warning(
            "No results to show. Select at least one language in the sidebar.",
            icon=":material/filter_list_off:",
        )
    else:
        st.caption(
            f":material/format_list_numbered: {len(out)} results · ranked by similarity"
        )
        for rank, (_, row) in enumerate(out.iterrows(), start=1):
            score = float(row["score"])
            lang = row["language"]
            color = LANGUAGE_COLORS.get(lang, "gray")
            with st.container(border=True):
                st.markdown(
                    f":gray-badge[#{rank}] :{color}-badge[{lang}]",
                )
                st.markdown(f"**{row['sentence']}**")
                st.progress(
                    max(0.0, min(1.0, score)),
                    text=f"{score:.1%} similarity",
                )
else:
    with st.container(border=True):
        st.markdown(":material/travel_explore: **Search across languages**")
        st.caption(
            "Type a query above in any language. Results are matched by meaning, "
            "not by exact words."
        )
