"""Step 6-7: classification, cross-validation, GridSearchCV."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

from src.config import RANDOM_STATE

NUM = ["runtime", "budget", "year", "month", "n_genres", "n_keywords", "overview_len"]
CAT = ["original_language", "runtime_cat"]
# WARNING: vote_count creates the target -> never a feature. Popularity / vote_average: check leakage!


def make_target(df: pd.DataFrame, quantile: float = 0.75) -> pd.Series:
    threshold = df["vote_count"].quantile(quantile)
    return (df["vote_count"] >= threshold).astype(int)


def build_models() -> dict[str, Pipeline]:
    pre = ColumnTransformer([("num", StandardScaler(), NUM),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])
    return {
        "logreg": Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))]),
        "rf": Pipeline([("pre", pre), ("clf", RandomForestClassifier(random_state=RANDOM_STATE))]),
        "svm": Pipeline([("pre", pre), ("clf", LinearSVC(random_state=RANDOM_STATE))]),
    }

# TODO: train/test split (stratified), StratifiedKFold + cross_validate,
#       metrics (acc, precision, recall, F1, ROC-AUC, confusion matrix), GridSearchCV, joblib.dump
