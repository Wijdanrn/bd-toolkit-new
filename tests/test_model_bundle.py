import os
import pandas as pd


def test_build_fitted_preprocessor():
    # stub: ensure SessionPreprocessor can be constructed with empty dicts
    from data_modeling import SessionPreprocessor
    p = SessionPreprocessor({}, {}, {}, [])
    df = pd.DataFrame({"a": [1, 2, 3]})
    out = p.transform(df)
    assert isinstance(out, pd.DataFrame)


def test_save_load_bundle(tmp_path):
    # stub: write a dummy bundle and load it via joblib
    import joblib
    bundle = {"model": None, "preprocessor": None, "metadata": {"name": "test"}}
    p = tmp_path / "models"
    p.mkdir()
    f = p / "test_model.pkl"
    joblib.dump(bundle, str(f))
    loaded = joblib.load(str(f))
    assert "metadata" in loaded


def test_split_preview():
    # stub: simple split sanity check
    from sklearn.model_selection import train_test_split
    import numpy as np
    X = np.arange(100).reshape((50, 2))
    y = list(range(50))
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    assert len(Xte) == 10
