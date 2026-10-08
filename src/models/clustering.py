import logging

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.config import DATA_PROCESSED, FIGURES_DIR, MODELS_DIR, RANDOM_STATE

logger = logging.getLogger(__name__)

STRUCTURED_FEATURES = ["runtime", "budget", "revenue", "popularity",
                       "vote_average", "n_genres", "overview_word_count",
                       "release_year"]
K_VALUES = range(2, 11)


def load_features():
    return pd.read_pickle(DATA_PROCESSED / "movies_features.pkl")


def load_tfidf():
    matrix = sparse.load_npz(DATA_PROCESSED / "tfidf_matrix.npz")
    movie_ids = np.load(DATA_PROCESSED / "tfidf_movie_ids.npy")
    return matrix, movie_ids


def build_structured_matrix(df):
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X = imputer.fit_transform(df[STRUCTURED_FEATURES])
    X = scaler.fit_transform(X)
    return X


def find_best_k(X, k_values=K_VALUES, label=""):
    scores = {}
    for k in k_values:
        labels = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit_predict(X)
        score = silhouette_score(X, labels)
        scores[k] = score
        print(f"-> {label} k={k}: silhouette score = {score:.3f}")

    best_k = max(scores, key=scores.get)
    print(f"-> {label} best k = {best_k} (silhouette score = {scores[best_k]:.3f})")
    return best_k, scores


def interpret_structured_clusters(df, labels):
    df = df.copy()
    df["cluster"] = labels

    print("-> Structured clusters: average feature values per cluster")
    print(df.groupby("cluster")[STRUCTURED_FEATURES].mean().round(1))

    exploded = df.explode("genres").dropna(subset=["genres"])
    if not exploded.empty:
        top_genres = exploded.groupby("cluster")["genres"].agg(lambda g: g.value_counts().idxmax())
        print("-> Structured clusters: most common genre per cluster")
        print(top_genres)

    return df


def run_structured_clustering():
    df = load_features()
    X = build_structured_matrix(df)

    best_k, scores = find_best_k(X, k_values=K_VALUES, label="structured")

    kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=RANDOM_STATE)
    labels = kmeans.fit_predict(X)
    df = interpret_structured_clusters(df, labels)

    joblib.dump(kmeans, MODELS_DIR / "kmeans_structured.joblib")
    df.to_pickle(DATA_PROCESSED / "movies_clustered_structured.pkl")
    logger.info("Saved structured clustering model and results")

    return df, kmeans, scores


def top_terms_per_cluster(vectorizer, matrix, labels, n_terms=10):
    terms = vectorizer.get_feature_names_out()
    result = {}
    for cluster in sorted(set(labels)):
        mask = labels == cluster
        mean_scores = np.asarray(matrix[mask].mean(axis=0)).ravel()
        top_idx = np.argsort(mean_scores)[::-1][:n_terms]
        result[cluster] = [terms[i] for i in top_idx]
    return result


def plot_tfidf_clusters(matrix, labels):
    svd = TruncatedSVD(n_components=2, random_state=RANDOM_STATE)
    coords = svd.fit_transform(matrix)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", alpha=0.5, s=15)
    ax.set_title("TF-IDF clusters (TruncatedSVD 2D projection)")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    fig.savefig(FIGURES_DIR / "10_tfidf_clusters_2d.png", bbox_inches="tight", dpi=120)
    logger.info("Saved %s", FIGURES_DIR / "10_tfidf_clusters_2d.png")
    return fig


def run_tfidf_clustering():
    matrix, movie_ids = load_tfidf()
    vectorizer = joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib")

    best_k, scores = find_best_k(matrix, k_values=K_VALUES, label="tfidf")

    kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=RANDOM_STATE)
    labels = kmeans.fit_predict(matrix)

    top_terms = top_terms_per_cluster(vectorizer, matrix, labels)
    print("-> TF-IDF clusters: top terms per cluster")
    for cluster, terms in top_terms.items():
        print(f"   cluster {cluster}: {terms}")

    plot_tfidf_clusters(matrix, labels)

    joblib.dump(kmeans, MODELS_DIR / "kmeans_tfidf.joblib")
    np.save(DATA_PROCESSED / "tfidf_cluster_labels.npy", labels)
    logger.info("Saved TF-IDF clustering model and results")

    return labels, kmeans, scores, top_terms


def run_clustering():
    print("-> Clustering on structured variables (runtime, budget, revenue, "
          "popularity, vote_average, n_genres, overview_word_count, release_year)")
    structured_df, structured_kmeans, structured_scores = run_structured_clustering()

    print("-> Clustering on TF-IDF (movie overview content)")
    tfidf_labels, tfidf_kmeans, tfidf_scores, top_terms = run_tfidf_clustering()

    return {
        "structured": (structured_df, structured_kmeans, structured_scores),
        "tfidf": (tfidf_labels, tfidf_kmeans, tfidf_scores, top_terms),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    run_clustering()