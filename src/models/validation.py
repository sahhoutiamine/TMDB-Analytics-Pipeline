import logging

import joblib
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate

from src.config import MODELS_DIR, RANDOM_STATE
from src.models.classification import build_pipelines, build_target, load_features, prepare_features

logger = logging.getLogger(__name__)

CV_SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]
N_SPLITS = 5

PARAM_GRID = {
    "classifier__n_estimators": [100, 300],
    "classifier__max_depth": [None, 10, 20],
    "classifier__min_samples_leaf": [1, 2, 4],
}


def run_cross_validation(X, y, pipelines, n_splits=N_SPLITS):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    results = {}

    for name, pipeline in pipelines.items():
        scores = cross_validate(pipeline, X, y, cv=cv, scoring=CV_SCORING)
        summary = {metric: scores[f"test_{metric}"].mean() for metric in CV_SCORING}
        results[name] = summary

        print(f"-> {name} ({n_splits}-fold cross-validation)")
        for metric, value in summary.items():
            std = scores[f"test_{metric}"].std()
            print(f"   {metric}: {value:.3f} (+/- {std:.3f})")

    return results


def run_grid_search(X, y, pipeline, param_grid=PARAM_GRID, n_splits=N_SPLITS):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(pipeline, param_grid, scoring="f1", cv=cv, n_jobs=-1)
    search.fit(X, y)
    return search


def run_validation():
    df = load_features()
    df, threshold = build_target(df)
    X, y = prepare_features(df)

    print(f"-> Using {N_SPLITS}-fold StratifiedKFold: each fold keeps the same "
          f"class balance as the full dataset, so no fold ends up with too "
          f"few high_engagement movies to learn from.")

    pipelines = build_pipelines()
    before = run_cross_validation(X, y, pipelines)

    print("-> Running GridSearchCV on random_forest: n_estimators, max_depth, "
          "min_samples_leaf")
    search = run_grid_search(X, y, pipelines["random_forest"])

    print(f"-> Best parameters found: {search.best_params_}")

    before_f1 = before["random_forest"]["f1"]
    after_f1 = search.best_score_
    print(f"-> Random Forest F1-score before tuning: {before_f1:.3f}")
    print(f"-> Random Forest F1-score after tuning:  {after_f1:.3f}")
    if after_f1 > before_f1:
        print("-> GridSearchCV improved the F1-score.")
    else:
        print("-> GridSearchCV did not improve on the default hyperparameters "
              "for this metric - the defaults were already a good fit.")

    joblib.dump(search.best_estimator_, MODELS_DIR / "classification_random_forest_tuned.joblib")
    logger.info("Saved %s", MODELS_DIR / "classification_random_forest_tuned.joblib")

    return before, search


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    run_validation()