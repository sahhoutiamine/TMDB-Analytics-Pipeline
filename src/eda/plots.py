import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import DATA_PROCESSED, FIGURES_DIR

logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid")


def load_clean():
    return pd.read_pickle(DATA_PROCESSED / "movies_clean.pkl")


def _save(fig, name):
    path = FIGURES_DIR / name
    fig.savefig(path, bbox_inches="tight", dpi=120)
    logger.info("Saved %s", path)


def plot_rating_distribution(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["vote_average"].dropna(), bins=30, kde=True, ax=ax, color="#4C72B0")
    ax.set_title("Distribution of movie ratings")
    ax.set_xlabel("Average rating (0-10)")
    _save(fig, "01_rating_distribution.png")

    mean = df["vote_average"].mean()
    print(f"-> Ratings are centered around {mean:.1f}/10. Very low and very high "
          f"ratings are both rare, typical of averaged scores from many votes.")
    return fig


def plot_popularity_distribution(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["popularity"], bins=40, ax=ax, color="#DD8452")
    ax.set_title("Distribution of popularity score")
    ax.set_yscale("log")
    _save(fig, "02_popularity_distribution.png")

    median = df["popularity"].median()
    max_pop = df["popularity"].max()
    print(f"-> Popularity is heavily skewed: half the movies score below "
          f"{median:.1f}, while the most popular one reaches {max_pop:.0f}.")
    return fig


def explode_genres(df):
    return df.explode("genres").dropna(subset=["genres"])


def plot_movies_by_genre(df):
    exploded = explode_genres(df)
    counts = exploded["genres"].value_counts().head(15)

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(x=counts.values, y=counts.index, ax=ax, color="#55A868")
    ax.set_title("Number of movies per genre (top 15)")
    ax.set_xlabel("Number of movies")
    _save(fig, "03_movies_by_genre.png")

    top_genre = counts.index[0]
    print(f"-> '{top_genre}' is the most common genre ({counts.iloc[0]} movies). "
          f"Counts don't sum to the total because most movies have several genres.")
    return fig


def plot_releases_per_year(df):
    counts = df["release_date"].dt.year.value_counts().sort_index()
    counts = counts[counts.index >= 1960]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(counts.index, counts.values, color="#4C72B0")
    ax.fill_between(counts.index, counts.values, alpha=0.2, color="#4C72B0")
    ax.set_title("Number of movies released per year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of movies")
    _save(fig, "04_releases_per_year.png")

    peak_year = counts.idxmax()
    print(f"-> Releases climb sharply after 2000 and peak around {int(peak_year)}. "
          f"This reflects TMDB having more complete data for recent movies.")
    return fig


def plot_runtime_distribution(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["runtime"].dropna(), bins=30, ax=ax, color="#C44E52")
    ax.set_title("Distribution of runtime (minutes)")
    _save(fig, "05_runtime_distribution.png")

    median = df["runtime"].median()
    print(f"-> Half of the movies run {median:.0f} minutes or less, with a long "
          f"right tail from a small number of very long movies.")
    return fig


def plot_budget_vs_revenue(df):
    data = df.dropna(subset=["budget", "revenue"])
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=data, x="budget", y="revenue", alpha=0.4, ax=ax, color="#8172B2")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Budget vs revenue (log-log scale)")
    _save(fig, "06_budget_vs_revenue.png")

    corr = data["budget"].corr(data["revenue"])
    print(f"-> Budget and revenue are correlated ({corr:.2f}): bigger budgets "
          f"tend to bring bigger revenue, but many exceptions exist both ways.")
    return fig


def plot_votes_vs_popularity(df):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df, x="vote_count", y="popularity", alpha=0.4, ax=ax, color="#64B5CD")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Vote count vs popularity (log-log scale)")
    _save(fig, "07_votes_vs_popularity.png")

    corr = df["vote_count"].corr(df["popularity"])
    if abs(corr) < 0.2:
        strength = "barely related"
    elif abs(corr) < 0.5:
        strength = "weakly related"
    else:
        strength = "strongly related"
    print(f"-> Vote count and popularity are {strength} (correlation {corr:.2f}). "
          f"TMDB's popularity score depends on recent activity (views, searches), "
          f"not just on how many people have voted over a movie's lifetime - a "
          f"movie can be popular right now with few total votes, or vice versa.")
    return fig


def plot_boxplots(df):
    cols = ["runtime", "budget", "revenue", "popularity"]
    fig, axes = plt.subplots(1, len(cols), figsize=(16, 5))
    for ax, col in zip(axes, cols):
        sns.boxplot(y=df[col].dropna(), ax=ax, color="#4C72B0")
        ax.set_title(col)
    fig.suptitle("Boxplots: spotting outliers")
    _save(fig, "08_boxplots.png")

    print("-> Every column has outliers (blockbusters with huge budgets/revenue, "
          "unusually long or short movies). These are real values, not errors, "
          "but scale-sensitive models will need feature scaling.")
    return fig


def plot_correlation_heatmap(df):
    cols = ["runtime", "budget", "revenue", "popularity", "vote_average", "vote_count"]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation between numeric variables")
    _save(fig, "09_correlation_heatmap.png")

    mask = np.eye(len(corr), dtype=bool)
    strongest = corr.mask(mask).abs().stack().idxmax()
    value = corr.loc[strongest]
    print(f"-> The strongest relationship is between '{strongest[0]}' and "
          f"'{strongest[1]}' ({value:.2f}). vote_count will be used to build the "
          f"classification target later, so it must not be used as a feature.")
    return fig


def run_eda():
    df = load_clean()
    logger.info("Running EDA on %s movies", len(df))

    plot_rating_distribution(df)
    plot_popularity_distribution(df)
    plot_movies_by_genre(df)
    plot_releases_per_year(df)
    plot_runtime_distribution(df)
    plot_budget_vs_revenue(df)
    plot_votes_vs_popularity(df)
    plot_boxplots(df)
    plot_correlation_heatmap(df)

    plt.close("all")
    logger.info("EDA done. Figures saved in %s", FIGURES_DIR)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    run_eda()