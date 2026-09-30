from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score

from src.config import RANDOM_STATE


def best_k(X, k_range=range(2, 11)):
    scores = {}
    for k in k_range:
        labels = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit_predict(X)
        scores[k] = silhouette_score(X, labels)
    return scores


def project_2d(X):
    return TruncatedSVD(n_components=2, random_state=RANDOM_STATE).fit_transform(X)

