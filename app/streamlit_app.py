import pandas as pd
import streamlit as st

from src.config import DATA_PROCESSED

st.set_page_config(page_title="Movie ML Platform", layout="wide")


@st.cache_data
def load_movies():
    return pd.read_pickle(DATA_PROCESSED / "movies_features.pkl")


st.title("Movie ML Platform")
st.write("Explore the TMDB movie dataset: dashboard, classification and clusters.")
df = load_movies()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Movies", len(df))
col2.metric("Avg rating", f"{df['vote_average'].mean():.1f}")
col3.metric("Avg popularity", f"{df['popularity'].mean():.1f}")
col4.metric("Years covered", f"{int(df['release_year'].min())} - {int(df['release_year'].max())}")

st.write("Use the sidebar to open a page.")