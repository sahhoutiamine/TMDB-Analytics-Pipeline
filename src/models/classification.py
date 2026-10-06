import logging

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                              precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

from src.config import DATA_PROCESSED, MODELS_DIR, RANDOM_STATE

logger = logging.getLogger(__name__)

NUMERIC_FEATURES = ["runtime", "budget", "revenue", "popularity", "vote_average",
                    "release_year", "release_month", "release_decade",
                    "n_genres", "overview_word_count", "budget_known", "revenue_known"]
CATEGORICAL_FEATURES = ["original_language", "runtime_category"]
TEXT_FEATURE = "keywords_text"
TARGET_QUANTILE = 0.75


def load_features():
    return pd.read_pickle(DATA_PROCESSED / "movies_features.pkl")


def build_target(df, quantile=TARGET_QUANTILE):
    df = df.copy()
    threshold = df["vote_count"].quantile(quantile)
    df["high_engagement"] = (df["vote_count"] >= threshold).astype(int)
    return df, threshold


def prepare_features(df):
    df = df.copy()
    df["keywords_text"] = df["keywords"].apply(lambda words: " ".join(words))
    columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TEXT_FEATURE]
    X = df[columns]
    y = df["high_engagement"]
    return X, y


def make_preprocessor():
    return ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), NUMERIC_FEATURES),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL_FEATURES),
        ("text", TfidfVectorizer(max_features=300), TEXT_FEATURE),
    ])


def build_pipelines():
    return {
        "logistic_regression": Pipeline([
            ("preprocess", make_preprocessor()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        "random_forest": Pipeline([
            ("preprocess", make_preprocessor()),
            ("classifier", RandomForestClassifier(random_state=RANDOM_STATE)),
        ]),
        "linear_svm": Pipeline([
            ("preprocess", make_preprocessor()),
            ("classifier", LinearSVC(random_state=RANDOM_STATE)),
        ]),
    }


def get_scores(pipeline, X_test):
    classifier = pipeline.named_steps["classifier"]
    if hasattr(classifier, "predict_proba"):
        return pipeline.predict_proba(X_test)[:, 1]
    return pipeline.decision_function(X_test)


def evaluate_model(name, pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    scores = get_scores(pipeline, X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, scores),
    }
    cm = confusion_matrix(y_test, y_pred)

    print(f"-> {name}")
    for metric, value in metrics.items():
        print(f"   {metric}: {value:.3f}")
    print(f"   confusion matrix (rows=actual, cols=predicted):\n{cm}")

    return metrics, cm


def run_classification(test_size=0.2):
    df = load_features()
    df, threshold = build_target(df)

    positive_rate = df["high_engagement"].mean() * 100
    print(f"-> high_engagement = 1 when vote_count >= {threshold:.0f} "
          f"(the top {int((1 - TARGET_QUANTILE) * 100)}% most-voted movies). "
          f"{positive_rate:.1f}% of movies are labeled high engagement.")
    print("-> vote_count is only used to build this target, never as a "
          "feature below - using it as a feature would be data leakage.")

    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE)

    pipelines = build_pipelines()
    results = {}
    for name, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)
        metrics, _ = evaluate_model(name, pipeline, X_test, y_test)
        results[name] = metrics
        joblib.dump(pipeline, MODELS_DIR / f"classification_{name}.joblib")
        logger.info("Saved %s", MODELS_DIR / f"classification_{name}.joblib")

    best_name = max(results, key=lambda n: results[n]["f1"])
    print(f"-> Best model by F1-score: {best_name}")

    return results, pipelines


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    run_classification()