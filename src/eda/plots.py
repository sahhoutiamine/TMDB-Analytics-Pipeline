"""Step 3: EDA plots. Each function saves a figure in reports/figures/."""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import FIGURES_DIR

sns.set_theme(style="whitegrid")


def plot_rating_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["vote_average"], bins=30, kde=True, ax=ax)
    ax.set_title("Distribution of ratings")
    fig.savefig(FIGURES_DIR / "rating_distribution.png", bbox_inches="tight")
    return fig

# TODO: popularity, genres, releases per year, runtime, budget vs revenue,
#       votes vs popularity, boxplots, correlation heatmap (+ short interpretation for each)
