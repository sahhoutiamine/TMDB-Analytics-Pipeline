import pandas as pd
import streamlit as st

from src.config import DATA_PROCESSED, FIGURES_DIR

st.title("Dashboard")


@st.cache_data
def load_movies():
    return pd.read_pickle(DATA_PROCESSED / "movies_features.pkl")


df = load_movies()

col1, col2, col3 = st.columns(3)
col1.metric("Movies", len(df))
col2.metric("Avg rating", f"{df['vote_average'].mean():.1f}")
col3.metric("Avg popularity", f"{df['popularity'].mean():.1f}")

st.subheader("Movies released per year")
year_counts = df["release_year"].dropna().astype(int).value_counts().sort_index()
st.bar_chart(year_counts)

st.subheader("Movies per genre")
genre_counts = df.explode("genres").dropna(subset=["genres"])["genres"].value_counts()
st.bar_chart(genre_counts)

st.subheader("EDA figures")
figures = sorted(FIGURES_DIR.glob("*.png"))
if not figures:
    st.info("No figures found. Run: python -m src.eda.plots")
else:
    for path in figures:
        if not path.name.startswith("10_"):
            st.image(str(path), caption=path.stem, use_column_width=True)