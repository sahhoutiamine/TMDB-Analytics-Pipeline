import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy import sparse

from src.config import DATA_PROCESSED, FIGURES_DIR, MODELS_DIR
from src.models.clustering import STRUCTURED_FEATURES, top_terms_per_cluster

st.title("Clusters")


@st.cache_data
def load_structured():
    return pd.read_pickle(DATA_PROCESSED / "movies_clustered_structured.pkl")


@st.cache_data
def load_tfidf_clusters():
    movies = pd.read_pickle(DATA_PROCESSED / "movies_features.pkl")
    movie_ids = np.load(DATA_PROCESSED / "tfidf_movie_ids.npy")
    labels = np.load(DATA_PROCESSED / "tfidf_cluster_labels.npy")
    clusters = pd.DataFrame({"movie_id": movie_ids, "cluster": labels})
    return movies.merge(clusters, on="movie_id")


@st.cache_resource
def load_tfidf_artifacts():
    matrix = sparse.load_npz(DATA_PROCESSED / "tfidf_matrix.npz")
    vectorizer = joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib")
    labels = np.load(DATA_PROCESSED / "tfidf_cluster_labels.npy")
    return matrix, vectorizer, labels


view = st.radio("Clustering view", ["Structured variables", "TF-IDF (overview content)"])

if view == "Structured variables":
    df = load_structured()

    st.subheader("Average feature values per cluster")
    st.dataframe(df.groupby("cluster")[STRUCTURED_FEATURES].mean().round(1))

    st.subheader("Most common genre per cluster")
    exploded = df.explode("genres").dropna(subset=["genres"])
    st.dataframe(exploded.groupby("cluster")["genres"].agg(lambda g: g.value_counts().idxmax()))

    cluster_choice = st.selectbox("Browse movies in cluster", sorted(df["cluster"].unique()))
    st.dataframe(df[df["cluster"] == cluster_choice][["title", "release_year", "vote_average"]].head(20))

else:
    df = load_tfidf_clusters()
    matrix, vectorizer, labels = load_tfidf_artifacts()

    st.subheader("Top terms per cluster")
    for cluster, terms in top_terms_per_cluster(vectorizer, matrix, labels).items():
        st.write(f"**Cluster {cluster}**: {', '.join(terms)}")

    st.subheader("2D projection")
    image_path = FIGURES_DIR / "10_tfidf_clusters_2d.png"
    if image_path.exists():
        st.image(str(image_path), use_column_width=True)
    else:
        st.info("No figure found. Run: python -m src.models.clustering")

    cluster_choice = st.selectbox("Browse movies in cluster", sorted(df["cluster"].unique()))
    st.dataframe(df[df["cluster"] == cluster_choice][["title", "release_year"]].head(20))