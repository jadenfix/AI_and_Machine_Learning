"""Original crime regressions with keyed panel differences and explicit input.

The absent London file must be supplied by its owner. --fixture is fictional
and verifies calculations only; regression coefficients are not causal effects.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

LOGS = ["lcrime", "lpolice", "lemp", "lun", "lymale", "lwhite"]
REQUIRED = ["borough", "week", "crime", "population", "police", "emp", "un", "ymale", "white"]


def diff_data(frame):
    if frame.duplicated(["borough", "week"]).any():
        raise ValueError("duplicate borough/week observations")
    first = frame[frame.week < 53].copy()
    second = frame[frame.week >= 53].copy()
    first["week"] += 52
    paired = second.merge(first, on=["borough", "week"], suffixes=("", "_previous"),
                          how="outer", validate="one_to_one", indicator=True)
    if not paired["_merge"].eq("both").all():
        raise ValueError("every observation must have the same borough/week in the preceding year")
    result = paired[["borough", "week"]].copy()
    for name in LOGS:
        result["d" + name] = paired[name] - paired[name + "_previous"]
    return result.sort_values(["borough", "week"]).reset_index(drop=True)


def prepare(frame):
    missing = set(REQUIRED) - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("crime panel must contain observations")
    frame = frame[REQUIRED].copy()
    try:
        frame = frame.apply(pd.to_numeric, errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError("all required columns must be numeric") from exc
    if not np.isfinite(frame.to_numpy()).all() or (frame <= 0).any().any():
        raise ValueError("all required values must be finite and positive")
    if not frame.week.between(1, 104).all() or (frame.week % 1 != 0).any() or (frame.borough % 1 != 0).any():
        raise ValueError("week must be an integer from 1 to 104; borough must be an integer")
    frame["lcrime"] = np.log(frame.crime / frame.population)
    frame["lpolice"] = np.log(frame.police / frame.population)
    for name in ("emp", "un", "ymale", "white"):
        frame["l" + name] = np.log(frame[name])
    return frame


def fixture(seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for borough in range(1, 21):
        for week in range(1, 105):
            values = np.exp(rng.normal(5, .25, 7))
            rows.append([borough, week, *values])
    return pd.DataFrame(rows, columns=REQUIRED)


def evaluate(frame):
    data = prepare(frame)
    diff = diff_data(data)
    diff["sixweeks"] = diff.week.between(80, 85).astype(int)
    diff["sixweeks_treat"] = (diff.borough.isin([1, 2, 3, 6, 14]) & diff.sixweeks.eq(1)).astype(int)
    # sixweeks is fully determined by C(week), so adding both is rank-deficient.
    formulas = ["lcrime ~ lpolice + lemp + lun + lymale + lwhite",
                "dlcrime ~ C(week) + dlpolice + dlemp + dlun + dlymale + dlwhite",
                "dlcrime ~ C(week) + sixweeks_treat + dlemp + dlun + dlymale + dlwhite"]
    results = []
    for formula, source in zip(formulas, [data, diff, diff]):
        model = smf.ols(formula, data=source).fit()
        rank = np.linalg.matrix_rank(model.model.exog)
        if rank != model.model.exog.shape[1] or model.df_resid <= 0:
            raise ValueError("regression design is rank-deficient or has no residual degrees of freedom")
        results.append({"formula": formula, "rows": int(model.nobs),
                        "r_squared": float(model.rsquared), "parameters": model.params.to_dict(),
                        "standard_errors": model.bse.to_dict()})
    return {"models": results, "pairs": len(diff), "limits": "Descriptive OLS; no causal identification or original London data reproduction claim."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", type=Path)
    source.add_argument("--fixture", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        frame = fixture() if args.fixture else pd.read_csv(args.csv)
        report = evaluate(frame)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        parser.error(str(exc))
    report["data_mode"] = "fictional_fixture" if args.fixture else "user_csv"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Completed {len(report['models'])} regressions / {report['pairs']} keyed pairs ({report['data_mode']})")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
