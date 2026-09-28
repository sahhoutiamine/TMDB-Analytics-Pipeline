import pandas as pd

from src.features.engineering import add_features


def test_add_features():
    df = pd.DataFrame({"release_date": pd.to_datetime(["2010-05-01"]), "genres": [["Drama"]],
                       "keywords": [["a", "b"]], "runtime": [100], "overview": ["hello world"]})
    out = add_features(df)
    assert out.loc[0, "decade"] == 2010 and out.loc[0, "n_keywords"] == 2
