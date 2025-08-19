
import os, io, re, html, joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

# ------------------- Config -------------------
st.set_page_config(page_title="Fake News Detection", layout="wide")
st.title("📰 Fake News Detection Using Machine Learning")
st.markdown("By Prateek Negi and Rishabh Bhagat")

DATA_DIR = "data"
LABEL = "label"  # 1=real, 0=fake

# ------------------- Helpers -------------------
URL_RE = re.compile(r"https?://\S+|www\.\S+")
HTML_TAG_RE = re.compile(r"<.*?>")
NON_ALPHA_RE = re.compile(r"[^a-zA-Z\s]")
MULTISPACE_RE = re.compile(r"\s+")

def basic_clean(text):
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = URL_RE.sub(" ", text)
    text = HTML_TAG_RE.sub(" ", text)
    text = text.lower()
    text = NON_ALPHA_RE.sub(" ", text)
    text = MULTISPACE_RE.sub(" ", text).strip()
    return text

def combine_columns(row, cols):
    parts = [str(row[c]) for c in cols if c in row and isinstance(row[c], str)]
    return " ".join(parts).strip()

def load_from_folder():
    p_true = Path(DATA_DIR) / "True.csv"
    p_fake = Path(DATA_DIR) / "Fake.csv"
    p_single = Path(DATA_DIR) / "news.csv"
    if p_true.exists() and p_fake.exists():
        df_true = pd.read_csv(p_true, encoding="utf-8")
        df_fake = pd.read_csv(p_fake, encoding="utf-8")
        df_true[LABEL] = 1
        df_fake[LABEL] = 0
        df = pd.concat([df_true, df_fake], ignore_index=True)
        # choose text columns
        text_cols = [c for c in ["title","text","content","article","body"] if c in df.columns]
        if not text_cols:
            text_cols = [c for c in df.columns if c not in [LABEL, "subject", "date"]]
    elif p_single.exists():
        df = pd.read_csv(p_single, encoding="utf-8")
        if LABEL not in df.columns:
            st.error("news.csv must contain a 'label' column (1=real, 0=fake).")
            return None
        text_cols = [c for c in ["title","text","content","article","body"] if c in df.columns]
        if not text_cols:
            text_cols = [c for c in df.columns if c != LABEL]
    else:
        return None
    df["full_text"] = df.apply(lambda r: combine_columns(r, text_cols), axis=1).map(basic_clean)
    df = df[df[LABEL].isin([0,1])].dropna(subset=["full_text"]).copy()
    df = df.drop_duplicates(subset=["full_text"]).reset_index(drop=True)
    return df

def make_vectorizer():
    return TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_df=0.9, min_df=3, sublinear_tf=True)

def train_pipeline(X_train, y_train):
    vec = make_vectorizer()
    pipe = Pipeline([("tfidf", vec), ("clf", LogisticRegression(max_iter=2000))])
    pipe.fit(X_train, y_train)
    return pipe

# ------------------- UI Layout -------------------
menu = ["Dataset", "EDA", "Train & Evaluate", "Predict"]
choice = st.sidebar.selectbox("Choose Page", menu)

# ------------------- Dataset Page -------------------
if choice == "Dataset":
    st.header("📥 Dataset")
    st.write("Place `True.csv` and `Fake.csv` inside the `data/` folder (or upload below). These are the Kaggle files.")
    uploaded = st.file_uploader("Upload a CSV (must contain a label column 1=real,0=fake)", type=["csv"])
    if uploaded is not None:
        try:
            df_up = pd.read_csv(uploaded, encoding="utf-8")
        except Exception:
            uploaded.seek(0)
            df_up = pd.read_csv(uploaded, encoding="ISO-8859-1")
        if LABEL not in df_up.columns:
            st.error("Uploaded CSV must include a 'label' column (1=real,0=fake).")
        else:
            text_cols = [c for c in ["title","text","content","article","body"] if c in df_up.columns]
            if not text_cols:
                text_cols = [c for c in df_up.columns if c != LABEL]
            df_up["full_text"] = df_up.apply(lambda r: combine_columns(r, text_cols), axis=1).map(basic_clean)
            st.session_state["df"] = df_up[df_up[LABEL].isin([0,1])].dropna(subset=["full_text"]).reset_index(drop=True)
            st.success(f"Uploaded dataset with {len(st.session_state['df'])} rows.")
    else:
        st.write("Or load from `data/` folder below:")
        if st.button("Load from data/ folder"):
            df0 = load_from_folder()
            if df0 is None:
                st.warning("No dataset found in data/. Please put Kaggle files or upload a CSV.")
            else:
                st.session_state["df"] = df0
                st.success(f"Loaded dataset with {len(df0)} rows.")

    if "df" in st.session_state and st.session_state["df"] is not None:
        st.write("### Sample rows")
        st.dataframe(st.session_state["df"].head(10))
        st.write("Columns:", list(st.session_state["df"].columns))

# ------------------- EDA Page -------------------
elif choice == "EDA":
    st.header("🔎 Exploratory Data Analysis (EDA)")
    if "df" not in st.session_state or st.session_state["df"] is None:
        st.warning("Load dataset first (Dataset page).")
    else:
        df = st.session_state["df"]
        st.write("Total rows:", len(df))
        st.write("Class distribution:")
        counts = df[LABEL].value_counts().sort_index()
        fig = px.bar(x=["Fake News","Real News"], y=[counts.get(0,0), counts.get(1,0)], labels={"x":"Label","y":"Count"})
        st.plotly_chart(fig, use_container_width=True)

        st.write("### Wordclouds (top terms)")
        fake_text = " ".join(df[df[LABEL]==0]["full_text"].astype(str).values)
        real_text = " ".join(df[df[LABEL]==1]["full_text"].astype(str).values)
        if len(fake_text.strip())>0:
            wc_fake = WordCloud(width=600, height=300, background_color="white").generate(fake_text)
            fig1, ax1 = plt.subplots(figsize=(8,4))
            ax1.imshow(wc_fake, interpolation="bilinear"); ax1.axis("off"); st.pyplot(fig1)
        if len(real_text.strip())>0:
            wc_real = WordCloud(width=600, height=300, background_color="white").generate(real_text)
            fig2, ax2 = plt.subplots(figsize=(8,4))
            ax2.imshow(wc_real, interpolation="bilinear"); ax2.axis("off"); st.pyplot(fig2)

# ------------------- Train & Evaluate Page -------------------
elif choice == "Train & Evaluate":
    st.header("🧠 Train & Evaluate (models train every run)")
    if "df" not in st.session_state or st.session_state["df"] is None:
        st.warning("Load dataset first (Dataset page).")
    else:
        df = st.session_state["df"]
        X = df["full_text"].values
        y = df[LABEL].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        st.write("Training Logistic Regression (TF-IDF) now...")
        pipe = train_pipeline(X_train, y_train)
        st.session_state["best_pipe"] = pipe
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.write(f"Accuracy: {acc:.4f}")

        st.write("Classification report:")
        st.text(classification_report(y_test, y_pred, digits=4))

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion Matrix")
        st.pyplot(fig)

        # Save model
        joblib.dump(pipe, os.path.join("models", "best_pipeline.pkl"))
        st.success("Saved trained model to models/best_pipeline.pkl")

# ------------------- Predict Page -------------------
elif choice == "Predict":
    st.header("🧪 Predict (use trained model)")
    pipe = None
    model_path = os.path.join("models", "best_pipeline.pkl")
    if os.path.exists(model_path):
        try:
            pipe = joblib.load(model_path)
            st.caption("Loaded saved model from models/best_pipeline.pkl")
        except Exception as e:
            st.error(f"Failed to load saved model: {e}")
    elif "best_pipe" in st.session_state and st.session_state["best_pipe"] is not None:
        pipe = st.session_state["best_pipe"]
        st.caption("Using model from this session (recently trained)")

    if pipe is None:
        st.warning("No trained model found. Train a model in 'Train & Evaluate' first.")
    else:
        text = st.text_area("Enter news text (headline or paragraph):", height=200)
        if st.button("Predict"):
            if text.strip() == "":
                st.warning("Please enter some text.")
            else:
                pred = pipe.predict([basic_clean(text)])[0]
                label = "Real (1) ✅" if pred==1 else "Fake (0) ❌"
                st.subheader("Prediction: " + label)
                if hasattr(pipe.named_steps["clf"], "predict_proba"):
                    proba = pipe.predict_proba([basic_clean(text)])[0,1]
                    st.write(f"Probability of Real: {proba:.3f}")
