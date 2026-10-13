import json

import numpy as np
import pandas as pd
import pytest

from ml_workflow import evaluate_workflow, main, make_fixture, render_receipt


def test_fixture_and_report_are_deterministic():
    first_x, first_y = make_fixture(seed=7)
    second_x, second_y = make_fixture(seed=7)
    pd.testing.assert_frame_equal(first_x, second_x)
    pd.testing.assert_series_equal(first_y, second_y)
    assert evaluate_workflow(first_x, first_y, seed=7) == evaluate_workflow(
        second_x, second_y, seed=7
    )


def test_workflow_uses_disjoint_holdout_and_training_oof_predictions():
    features, target = make_fixture(rows=100)
    report = evaluate_workflow(features, target)
    assert report["split_integrity"] is True
    assert report["train_rows"] == 80
    assert report["holdout_rows"] == 20
    assert all(item["oof_rows"] == 80 for item in report["candidates"])
    assert report["winner"] == min(
        report["candidates"], key=lambda item: (item["cv_rmse"], item["name"])
    )["name"]


def test_nonlinear_fixture_selects_quadratic_model():
    features, target = make_fixture(rows=240, seed=8257)
    report = evaluate_workflow(features, target, seed=8257)
    assert report["winner"] == "quadratic_ridge"
    assert report["holdout"]["r2"] > 0.90


@pytest.mark.parametrize(
    "mutator, message",
    [
        (lambda x, y: (x.iloc[:-1], y), "same number"),
        (lambda x, y: (x.assign(income_index=np.inf), y), "finite numeric"),
        (lambda x, y: (x, y.mask(y.index == 0, np.nan)), "finite numeric"),
        (lambda x, y: (x, y.mask(y.index == 0, 0.0)), "positive"),
    ],
)
def test_invalid_data_is_rejected(mutator, message):
    features, target = make_fixture()
    bad_features, bad_target = mutator(features, target)
    with pytest.raises(ValueError, match=message):
        evaluate_workflow(bad_features, bad_target)


def test_cli_writes_machine_readable_receipt(tmp_path, capsys):
    output = tmp_path / "receipt.json"
    assert main(["--rows", "100", "--seed", "7", "--output", str(output)]) == 0
    report = json.loads(output.read_text())
    terminal = capsys.readouterr().out
    assert report["rows"] == 100
    assert report["mode"] == "synthetic_fixture"
    assert "LEAKAGE-SAFE MODEL SELECTION" in terminal
    assert "no real-world accuracy claim" in terminal
    assert render_receipt(report) in terminal
