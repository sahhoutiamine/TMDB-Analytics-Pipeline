import numpy as np
import pandas as pd
import pytest

from src.eda import plots


@pytest.fixture
def sample_df():
    n = 40
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "movie_id": range(n),
        "title": [f"Movie {i}" for i in range(n)],
        "overview": ["some overview text"] * n,
        "release_date": pd.to_datetime([f"{1990 + i}-01-01" for i in range(n)]),
        "runtime": rng.integers(60, 180, n).astype(float),
        "original_language": rng.choice(["en", "fr", "es"], n),
        "genres": [["Drama", "Action"] if i % 2 == 0 else ["Comedy"] for i in range(n)],
        "keywords": [["love"] for _ in range(n)],
        "budget": rng.integers(1_000, 100_000_000, n).astype(float),
        "revenue": rng.integers(1_000, 500_000_000, n).astype(float),
        "popularity": rng.uniform(1, 500, n),
        "vote_average": rng.uniform(1, 10, n),
        "vote_count": rng.integers(1, 30000, n),
    })


@pytest.fixture(autouse=True)
def use_tmp_figures_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(plots, "FIGURES_DIR", tmp_path)
    yield tmp_path


def test_plot_rating_distribution_saves_file(sample_df, use_tmp_figures_dir, capsys):
    plots.plot_rating_distribution(sample_df)
    assert (use_tmp_figures_dir / "01_rating_distribution.png").exists()
    assert "Ratings are centered" in capsys.readouterr().out


def test_plot_popularity_distribution_saves_file(sample_df, use_tmp_figures_dir):
    plots.plot_popularity_distribution(sample_df)
    assert (use_tmp_figures_dir / "02_popularity_distribution.png").exists()


def test_explode_genres_expands_list_column(sample_df):
    exploded = plots.explode_genres(sample_df)
    assert len(exploded) > len(sample_df)
    assert exploded["genres"].apply(lambda g: isinstance(g, str)).all()


def test_plot_movies_by_genre_saves_file(sample_df, use_tmp_figures_dir, capsys):
    plots.plot_movies_by_genre(sample_df)
    assert (use_tmp_figures_dir / "03_movies_by_genre.png").exists()
    assert "most common genre" in capsys.readouterr().out


def test_plot_releases_per_year_saves_file(sample_df, use_tmp_figures_dir):
    plots.plot_releases_per_year(sample_df)
    assert (use_tmp_figures_dir / "04_releases_per_year.png").exists()


def test_plot_runtime_distribution_saves_file(sample_df, use_tmp_figures_dir):
    plots.plot_runtime_distribution(sample_df)
    assert (use_tmp_figures_dir / "05_runtime_distribution.png").exists()


def test_plot_budget_vs_revenue_saves_file(sample_df, use_tmp_figures_dir):
    plots.plot_budget_vs_revenue(sample_df)
    assert (use_tmp_figures_dir / "06_budget_vs_revenue.png").exists()


def test_plot_votes_vs_popularity_saves_file(sample_df, use_tmp_figures_dir):
    plots.plot_votes_vs_popularity(sample_df)
    assert (use_tmp_figures_dir / "07_votes_vs_popularity.png").exists()


def test_plot_boxplots_saves_file(sample_df, use_tmp_figures_dir):
    plots.plot_boxplots(sample_df)
    assert (use_tmp_figures_dir / "08_boxplots.png").exists()


def test_plot_correlation_heatmap_saves_file(sample_df, use_tmp_figures_dir, capsys):
    plots.plot_correlation_heatmap(sample_df)
    assert (use_tmp_figures_dir / "09_correlation_heatmap.png").exists()
    assert "strongest relationship" in capsys.readouterr().out


def test_run_eda_end_to_end(monkeypatch, sample_df, use_tmp_figures_dir):
    monkeypatch.setattr(plots, "load_clean", lambda: sample_df)
    plots.run_eda()
    pngs = list(use_tmp_figures_dir.glob("*.png"))
    assert len(pngs) == 9