"""Deterministic, leakage-safe model-selection workflow.

The original coursework scripts remain in the repository as archival material.
This module provides a runnable example that does not need network access and
keeps a final holdout set outside model selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


FEATURE_NAMES = [
    "income_index",
    "room_index",
    "age_index",
    "distance_index",
    "services_index",
    "tax_index",
]


def make_fixture(rows: int = 240, seed: int = 8257) -> tuple[pd.DataFrame, pd.Series]:
    """Create a fictional housing-style regression fixture.

    The fixture is intentionally synthetic. It exists to exercise the complete
    evaluation pipeline; it is not evidence about real housing markets.
    """

    if rows < 40:
        raise ValueError("rows must be at least 40 for the five-fold workflow")
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(rows, len(FEATURE_NAMES)))
    features = pd.DataFrame(values, columns=FEATURE_NAMES)
    noise = rng.normal(scale=1.25, size=rows)
    target = (
        24.0
        + 4.8 * features["income_index"]
        + 2.1 * features["room_index"]
        - 1.7 * features["age_index"]
        - 2.5 * features["distance_index"]
        + 1.9 * features["services_index"]
        - 1.1 * features["tax_index"]
        + 3.2 * features["income_index"] * features["room_index"]
        + 1.6 * features["distance_index"] ** 2
        + noise
    )
    if target.min() <= 5.0:
        target = target + (5.0 - target.min())
    return features, pd.Series(target, name="target")


def candidate_models(seed: int = 8257) -> dict[str, Any]:
    """Return deterministic candidate estimators.

    Every transform is inside its pipeline, so each cross-validation fold fits
    preprocessing only on that fold's training rows.
    """

    return {
        "linear": Pipeline(
            [("scale", StandardScaler()), ("model", LinearRegression())]
        ),
        "quadratic_ridge": Pipeline(
            [
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=160,
            min_samples_leaf=2,
            max_features=0.8,
            n_jobs=1,
            random_state=seed,
        ),
        "log_linear": TransformedTargetRegressor(
            regressor=Pipeline(
                [("scale", StandardScaler()), ("model", LinearRegression())]
            ),
            func=np.log,
            inverse_func=np.exp,
            check_inverse=True,
        ),
    }


def _validate_data(features: pd.DataFrame, target: pd.Series) -> None:
    if len(features) != len(target):
        raise ValueError("features and target must have the same number of rows")
    if len(features) < 40:
        raise ValueError("at least 40 rows are required")
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("features must contain only finite numeric values")
    if not np.isfinite(target.to_numpy(dtype=float)).all():
        raise ValueError("target must contain only finite numeric values")
    if (target <= 0).any():
        raise ValueError("target values must be positive for the log-target candidate")


def evaluate_workflow(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    seed: int = 8257,
    test_size: float = 0.20,
) -> dict[str, Any]:
    """Select by training-only CV and evaluate the winner once on a holdout."""

    _validate_data(features, target)
    if not 0.1 <= test_size <= 0.4:
        raise ValueError("test_size must be between 0.1 and 0.4")

    row_ids = np.arange(len(features))
    train_ids, holdout_ids = train_test_split(
        row_ids, test_size=test_size, random_state=seed, shuffle=True
    )
    x_train = features.iloc[train_ids].reset_index(drop=True)
    y_train = target.iloc[train_ids].reset_index(drop=True)
    x_holdout = features.iloc[holdout_ids].reset_index(drop=True)
    y_holdout = target.iloc[holdout_ids].reset_index(drop=True)

    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    models = candidate_models(seed)
    candidates: list[dict[str, Any]] = []
    for name, estimator in models.items():
        oof_predictions = cross_val_predict(estimator, x_train, y_train, cv=cv)
        candidates.append(
            {
                "name": name,
                "cv_rmse": float(mean_squared_error(y_train, oof_predictions) ** 0.5),
                "cv_mae": float(mean_absolute_error(y_train, oof_predictions)),
                "oof_rows": len(oof_predictions),
            }
        )

    candidates.sort(key=lambda item: (item["cv_rmse"], item["name"]))
    winner_name = candidates[0]["name"]
    winner = clone(models[winner_name]).fit(x_train, y_train)
    holdout_predictions = winner.predict(x_holdout)

    return {
        "schema_version": 1,
        "mode": "synthetic_fixture",
        "seed": seed,
        "rows": len(features),
        "feature_count": features.shape[1],
        "train_rows": len(train_ids),
        "holdout_rows": len(holdout_ids),
        "split_integrity": bool(set(train_ids).isdisjoint(set(holdout_ids))),
        "selection_metric": "five_fold_training_oof_rmse",
        "winner": winner_name,
        "candidates": candidates,
        "holdout": {
            "rmse": float(mean_squared_error(y_holdout, holdout_predictions) ** 0.5),
            "mae": float(mean_absolute_error(y_holdout, holdout_predictions)),
            "r2": float(r2_score(y_holdout, holdout_predictions)),
        },
        "claims": [
            "Fictional deterministic fixture",
            "Final holdout excluded from model selection",
            "No real-world accuracy or deployment claim",
        ],
    }


def render_receipt(report: dict[str, Any]) -> str:
    rows = [
        "LEAKAGE-SAFE MODEL SELECTION",
        "============================",
        f"mode      : {report['mode']}",
        f"rows      : {report['train_rows']} train / {report['holdout_rows']} holdout",
        f"integrity : {'PASS' if report['split_integrity'] else 'FAIL'} (disjoint split)",
        "",
        "TRAINING-ONLY 5-FOLD OUT-OF-FOLD SCORES",
    ]
    for item in report["candidates"]:
        marker = "*" if item["name"] == report["winner"] else " "
        rows.append(
            f"{marker} {item['name']:<17} RMSE {item['cv_rmse']:>6.3f}  MAE {item['cv_mae']:>6.3f}"
        )
    holdout = report["holdout"]
    rows.extend(
        [
            "",
            f"winner    : {report['winner']}",
            f"holdout   : RMSE {holdout['rmse']:.3f} / MAE {holdout['mae']:.3f} / R2 {holdout['r2']:.3f}",
            "scope      : synthetic fixture; no real-world accuracy claim",
        ]
    )
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a deterministic, leakage-safe model-selection demonstration."
    )
    parser.add_argument("--rows", type=int, default=240)
    parser.add_argument("--seed", type=int, default=8257)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    features, target = make_fixture(args.rows, args.seed)
    report = evaluate_workflow(features, target, seed=args.seed)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(render_receipt(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
