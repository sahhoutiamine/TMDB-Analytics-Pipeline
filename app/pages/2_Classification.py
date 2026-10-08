import joblib
import pandas as pd
import streamlit as st

from src.config import DATA_PROCESSED, MODELS_DIR
from src.models.classification import build_target, prepare_features

st.title("Classification")


@st.cache_data
def load_data():
    df = pd.read_pickle(DATA_PROCESSED / "movies_features.pkl")
    return build_target(df)


@st.cache_resource
def load_models():
    return {
        "Logistic Regression": joblib.load(MODELS_DIR / "classification_logistic_regression.joblib"),
        "Random Forest": joblib.load(MODELS_DIR / "classification_random_forest.joblib"),
        "Linear SVM": joblib.load(MODELS_DIR / "classification_linear_svm.joblib"),
        "Random Forest (tuned)": joblib.load(MODELS_DIR / "classification_random_forest_tuned.joblib"),
    }


df, threshold = load_data()
models = load_models()

st.write(f"A movie is labeled **high engagement** when vote_count >= {threshold:.0f} "
         f"(the top 25% most-voted movies).")

st.subheader("Model comparison (5-fold cross-validation)")
comparison = pd.DataFrame({
    "accuracy": [0.863, 0.887, 0.845],
    "precision": [0.803, 0.881, 0.723],
    "recall": [0.598, 0.638, 0.618],
    "f1": [0.685, 0.739, 0.666],
    "roc_auc": [0.908, 0.941, 0.884],
}, index=["Logistic Regression", "Random Forest", "Linear SVM"])
st.dataframe(comparison)
st.write("Random Forest after GridSearchCV: F1 improved from 0.739 to 0.764.")

st.subheader("Try a prediction")
title = st.selectbox("Pick a movie", sorted(df["title"].unique()))
model_name = st.selectbox("Pick a model", list(models.keys()))

row = df[df["title"] == title].iloc[[0]]
X_row, y_row = prepare_features(row)

model = models[model_name]
prediction = model.predict(X_row)[0]
true_label = y_row.iloc[0]

labels = {1: "high engagement", 0: "not high engagement"}
st.write(f"Real label: **{labels[true_label]}**")
st.write(f"{model_name} prediction: **{labels[prediction]}**")

if hasattr(model, "predict_proba"):
    probability = model.predict_proba(X_row)[0][1]
    st.write(f"Predicted probability of high engagement: {probability:.2f}")

if prediction == true_label:
    st.success("The model got this movie right.")
else:
    st.error("The model got this movie wrong.")