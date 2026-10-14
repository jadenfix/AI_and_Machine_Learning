import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from course_regression import candidates, evaluate, load_csv
from crimeee import diff_data, fixture, prepare, evaluate as evaluate_crime
from ml_workflow import make_fixture


def test_holdout_changes_do_not_change_selection():
    x, y = make_fixture(80, 2)
    first = evaluate(x, y, seed=2)
    changed = y.copy()
    changed.iloc[first['holdout_ids']] += 10000
    second = evaluate(x, changed, seed=2)
    assert first['cv_mse'] == second['cv_mse']
    assert first['winner'] == second['winner']
    assert second['holdout_mse'] > first['holdout_mse']
    assert set(first['train_ids']).isdisjoint(first['holdout_ids'])
    assert len(first['cv_mse']) == 10


def test_log_candidate_predictions_return_original_units():
    x, y = make_fixture(80, 3)
    model = candidates()['4'].fit(x, y)
    raw_log = model.regressor_.predict(x)
    np.testing.assert_allclose(model.predict(x), np.exp(raw_log))


def test_csv_and_invalid_inputs(tmp_path):
    x, y = make_fixture(60, 4)
    frame = x.assign(MEDV=y)
    path = tmp_path / 'local.csv'
    frame.to_csv(path, index=False)
    loaded_x, loaded_y, digest = load_csv(path, 'MEDV')
    assert len(digest) == 64
    np.testing.assert_allclose(loaded_x, x)
    np.testing.assert_allclose(loaded_y, y)
    frame.loc[0, 'MEDV'] = 0
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match='positive'):
        load_csv(path, 'MEDV')
    with pytest.raises(ValueError, match='missing target'):
        load_csv(path, 'absent')


def test_keyed_differences_are_invariant_to_row_order():
    panel = prepare(fixture())
    first = diff_data(panel)
    shuffled = diff_data(panel.sample(frac=1, random_state=8))
    assert first.equals(shuffled)
    example = first.iloc[0]
    previous = panel[(panel.borough == example.borough) & (panel.week == example.week - 52)].iloc[0]
    current = panel[(panel.borough == example.borough) & (panel.week == example.week)].iloc[0]
    assert example.dlcrime == current.lcrime - previous.lcrime


def test_missing_and_duplicate_pairs_are_rejected():
    panel = prepare(fixture())
    with pytest.raises(ValueError, match='preceding year'):
        diff_data(panel.iloc[1:])
    with pytest.raises(ValueError, match='duplicate'):
        diff_data(__import__('pandas').concat([panel, panel.iloc[:1]]))


def test_three_original_crime_formulas_have_identified_designs():
    report = evaluate_crime(fixture())
    assert report['pairs'] == 1040
    assert len(report['models']) == 3
    assert 'sixweeks +' not in report['models'][2]['formula']
    assert all(np.isfinite(list(m['parameters'].values())).all() for m in report['models'])


def test_cli_requires_source_and_reports_missing_csv(tmp_path):
    root = Path(__file__).resolve().parents[1]
    for script in ['ml1.py', 'model_selection.py', 'crimeee.py']:
        missing_source = subprocess.run([sys.executable, str(root / script), '--output', str(tmp_path/'out.json')], capture_output=True, text=True)
        assert missing_source.returncode == 2
        missing_file = subprocess.run([sys.executable, str(root / script), '--csv', str(tmp_path/'missing.csv'), '--output', str(tmp_path/'out.json')], capture_output=True, text=True)
        assert missing_file.returncode == 2
        assert not (tmp_path/'out.json').exists()


def test_duplicate_csv_headers_fail_before_pandas_mangles_names(tmp_path):
    path = tmp_path / 'duplicate.csv'
    path.write_text('a,a,MEDV\n' + '1,2,3\n' * 40)
    with pytest.raises(ValueError, match='duplicate'):
        load_csv(path, 'MEDV')
