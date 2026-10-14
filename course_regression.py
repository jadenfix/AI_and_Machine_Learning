"""Runnable versions of the ten original regression candidates.

Supply a local numeric CSV, or explicitly opt into a fictional offline fixture.
Original exercise source is preserved in archive/. No automatic network access.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor

from ml_workflow import make_fixture


def candidates(seed=0):
    models = {}
    for number, (degree, interactions) in enumerate(((1, False), (2, True), (2, False)), 1):
        model = make_pipeline(PolynomialFeatures(degree, include_bias=False,
                              interaction_only=interactions), LinearRegression())
        models[str(number)] = model
        models[str(number + 3)] = TransformedTargetRegressor(
            regressor=clone(model), func=np.log, inverse_func=np.exp)
    models.update({
        "7": DecisionTreeRegressor(max_depth=3, random_state=seed),
        "8": DecisionTreeRegressor(max_depth=3, min_samples_leaf=10, random_state=seed),
        "9": RandomForestRegressor(n_estimators=100, max_features=1.0, random_state=seed, n_jobs=1),
        "10": RandomForestRegressor(n_estimators=100, max_features=1/3, random_state=seed, n_jobs=1),
    })
    return models


def load_csv(path, target_column):
    raw = Path(path).read_bytes()
    with Path(path).open(newline="") as handle:
        header = next(csv.reader(handle), [])
    if len(header) != len(set(header)):
        raise ValueError("duplicate column names")
    frame = pd.read_csv(path)
    if target_column not in frame.columns:
        raise ValueError(f"missing target column: {target_column}")
    if frame.columns.duplicated().any():
        raise ValueError("duplicate column names")
    features = frame.drop(columns=[target_column])
    target = frame[target_column]
    validate(features, target)
    return features, target, hashlib.sha256(raw).hexdigest()


def validate(features, target):
    if len(features) < 40 or len(features) != len(target) or features.shape[1] == 0:
        raise ValueError("need at least 40 matching rows and one feature")
    if features.shape[1] > 30:
        raise ValueError("at most 30 features supported by quadratic candidate")
    try:
        finite = np.isfinite(features.to_numpy(dtype=float)).all() and np.isfinite(np.asarray(target, dtype=float)).all()
    except (ValueError, TypeError) as exc:
        raise ValueError("all features and target must be numeric") from exc
    if not finite or (np.asarray(target, dtype=float) <= 0).any():
        raise ValueError("finite values and a strictly positive target are required")


def evaluate(features, target, seed=0):
    validate(features, target)
    ids = np.arange(len(target))
    train, test = train_test_split(ids, test_size=.2, random_state=seed)
    x, y = features.iloc[train], target.iloc[train]
    models = candidates(seed)
    scores = {}
    cv = KFold(5, shuffle=True, random_state=seed)
    for name, model in models.items():
        prediction = cross_val_predict(model, x, y, cv=cv)
        if not np.isfinite(prediction).all():
            raise ValueError(f"candidate {name} produced non-finite predictions")
        scores[name] = float(mean_squared_error(y, prediction))
    winner = min(scores, key=lambda name: (scores[name], int(name)))
    fitted = clone(models[winner]).fit(x, y)
    prediction = fitted.predict(features.iloc[test])
    return {"schema_version": 1, "seed": seed, "rows": len(ids),
            "train_ids": train.tolist(), "holdout_ids": test.tolist(),
            "cv_mse": scores, "winner": winner,
            "holdout_mse": float(mean_squared_error(target.iloc[test], prediction)),
            "holdout_predictions": prediction.tolist(),
            "prediction_scale": "original target units (including log candidates)",
            "selection": "training-only five-fold CV; one final holdout evaluation",
            "limits": "IID educational split; no temporal, causal, or deployment claim. BIC across different target transforms is intentionally not compared."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", type=Path)
    source.add_argument("--fixture", action="store_true", help="fictional offline data; not Boston housing")
    parser.add_argument("--target", default="MEDV")
    parser.add_argument("--rows", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.fixture:
            x, y = make_fixture(args.rows, args.seed)
            provenance = {"mode": "fictional_fixture", "original_data_reproduced": False}
        else:
            x, y, digest = load_csv(args.csv, args.target)
            provenance = {"mode": "user_csv", "input_sha256": digest, "target_column": args.target,
                          "original_data_reproduced": "not attested"}
        report = evaluate(x, y, args.seed)
    except (ValueError, OSError, pd.errors.ParserError) as exc:
        parser.error(str(exc))
    report["data"] = provenance
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
