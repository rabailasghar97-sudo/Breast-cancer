"""Compare clinical, RNA and histology Cox models on identical patient folds."""
import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODALITIES = ("clinical", "rna", "wsi")
COMBINATIONS = [*combinations(MODALITIES, 2), MODALITIES]


def table(path):
    data = pd.read_csv(path, dtype={"patient_id": str}).set_index("patient_id")
    if data.index.has_duplicates or data.index.isna().any():
        raise ValueError(f"Missing or duplicate patient ID in {path}")
    return data


def features(train, test, modality, limit):
    if modality == "clinical":
        estimator = ColumnTransformer([
            ("age", make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler()), ["age"]),
            ("categories", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["race", "tumour_stage"]),
        ])
        return estimator.fit_transform(train), estimator.transform(test)
    x, y = train.to_numpy(float), test.to_numpy(float)
    if modality == "rna":
        x, y = np.log2(1 + x), np.log2(1 + y)
    columns = np.argsort(-x.var(axis=0), kind="stable")[:limit]
    scaler = StandardScaler().fit(x[:, columns])
    return scaler.transform(x[:, columns]), scaler.transform(y[:, columns])


def fit(x, y, days, event, width=0, seed=42):
    """Ridge-penalized Cox partial likelihood with Breslow ties."""
    events = np.flatnonzero(event)
    mask = days[None, :] >= days[events, None]
    dimensions = x.shape[1]
    rng = np.random.default_rng(seed)
    initial = rng.normal(0, .1, dimensions * width + 2 * width) if width else np.zeros(dimensions)

    def forward(data, params):
        if not width:
            return data @ params, None
        weights = params[:dimensions * width].reshape(dimensions, width)
        bias, head = np.split(params[dimensions * width:], 2)
        hidden = np.tanh(data @ weights + bias)
        return hidden @ head, hidden

    def objective(params):
        risk, hidden = forward(x, params)
        masked = np.where(mask, risk[None, :], -np.inf)
        denom = logsumexp(masked, axis=1)
        derivative = np.exp(masked - denom[:, None]).sum(axis=0)
        derivative[events] -= 1
        derivative /= len(events)
        if width:
            upstream = derivative[:, None] * params[-width:] * (1 - hidden ** 2)
            gradient = np.r_[(x.T @ upstream).ravel(), upstream.sum(axis=0), hidden.T @ derivative]
        else:
            gradient = x.T @ derivative
        return (np.mean(denom - risk[events]) + .05 * (params @ params), gradient + .1 * params)

    result = minimize(objective, initial, jac=True, method="L-BFGS-B", options={"maxiter": 700})
    if not result.success:
        raise RuntimeError(str(result.message))
    fitted, _ = forward(x, result.x)
    scores, _ = forward(y, result.x)
    return (scores - fitted.mean()) / max(fitted.std(), 1e-8)


def folds(events, n, seed):
    if n < 2 or np.bincount(events, minlength=2).min() < n:
        raise ValueError(f"Need at least {n} events and censored patients for each split pool")
    return StratifiedKFold(n, shuffle=True, random_state=seed).split(events, events)


def predict(tables, days, events, outer, inner, limit, width, seed):
    results = {}
    assigned = np.zeros(len(days), int)

    def blocks(train, test):
        return {key: features(tables[key].iloc[train], tables[key].iloc[test], key, limit) for key in MODALITIES}

    def combine(block, names, side):
        return np.column_stack([block[name][side] for name in names])

    for fold, (train, test) in enumerate(folds(events, outer, seed), 1):
        assigned[test] = fold
        block = blocks(train, test)

        def record(name, values):
            results.setdefault(name, np.full(len(days), np.nan))[test] = values

        def outer_fit(x, y, hidden=0):
            return fit(x, y, days[train], events[train], hidden, seed)

        record("clinical_cox", outer_fit(*block["clinical"]))
        for names in [("clinical", "rna"), MODALITIES]:
            arrays = [np.column_stack([block[key][side][:, :8] if key == "rna" else block[key][side]
                                       for key in names]) for side in (0, 1)]
            record("linear_" + "+".join(names), outer_fit(*arrays))

        held_out = np.column_stack([outer_fit(*block[key], width) for key in MODALITIES])
        for column, key in enumerate(MODALITIES):
            record("unimodal_" + key, held_out[:, column])
        stacking = np.empty((len(train), len(MODALITIES)))
        for itrain, ival in folds(events[train], inner, seed + fold):
            fit_rows, val_rows = train[itrain], train[ival]
            inner_blocks = blocks(fit_rows, val_rows)
            for column, key in enumerate(MODALITIES):
                stacking[ival, column] = fit(*inner_blocks[key], days[fit_rows], events[fit_rows], width, seed)
        for names in COMBINATIONS:
            suffix = "+".join(names)
            record("early_" + suffix, outer_fit(combine(block, names, 0), combine(block, names, 1), width))
            columns = [MODALITIES.index(key) for key in names]
            record("late_" + suffix, outer_fit(stacking[:, columns], held_out[:, columns]))
    return assigned, results


def concordance(days, event, risk, assigned, rows):
    credit, count = 0., 0
    for fold in np.unique(assigned):
        indices = rows[assigned[rows] == fold]
        times, scores = days[indices], risk[indices]
        comparable = (event[indices, None] == 1) & (times[:, None] < times[None, :])
        difference = (scores[:, None] - scores[None, :])[comparable]
        credit += np.count_nonzero(difference > 1e-8) + .5 * np.count_nonzero(np.abs(difference) <= 1e-8)
        count += len(difference)
    return (credit / count if count else None), count


def evaluate(days, event, assignment, scores, seed, repeats):
    all_rows = np.arange(len(days))
    by_fold = [all_rows[assignment == fold] for fold in np.unique(assignment)]
    rng = np.random.default_rng(seed)
    draws = {name: [] for name in scores}
    for _ in range(repeats):
        rows = np.concatenate([rng.choice(group, len(group), replace=True) for group in by_fold])
        baseline, _ = concordance(days, event, scores["clinical_cox"], assignment, rows)
        if baseline is not None:
            for name, risk in scores.items():
                estimate, _ = concordance(days, event, risk, assignment, rows)
                draws[name].append(estimate - baseline)
    report = {}
    for name, risk in scores.items():
        estimate, pairs = concordance(days, event, risk, assignment, all_rows)
        report[name] = {"c_index": estimate, "comparable_pairs": pairs,
                        "paired_delta_ci95_vs_clinical": np.quantile(draws[name], [.025,.975]).tolist() if draws[name] else None}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--kind", required=True, choices=["real", "synthetic"])
    parser.add_argument("--endpoint", default="pfi", choices=["pfi", "os"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--inner", type=int, default=4)
    parser.add_argument("--features", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--bootstrap", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42,73,101])
    args = parser.parse_args()
    if min(args.features, args.hidden, args.bootstrap) < 1:
        parser.error("Feature, hidden-unit and bootstrap counts must be positive")
    cohort = table(args.inputs / "cohort.csv").sort_index()
    if args.kind == "real" and (not cohort.index.str.fullmatch(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}").all() or cohort.index.str.startswith("TCGA-ZZ-").any()):
        raise ValueError("Expected real TCGA patient IDs")
    days = cohort[f"{args.endpoint}_days"].to_numpy(float)
    event = cohort[f"{args.endpoint}_event"].to_numpy(float)
    if not np.isfinite(days).all() or (days <= 0).any() or not np.isin(event, [0, 1]).all():
        raise ValueError("Expected positive finite times and binary events")
    event = event.astype(int)
    tables = {"clinical": cohort[["age", "race", "tumour_stage"]].copy()}
    for name in ("race", "tumour_stage"):
        tables["clinical"][name] = tables["clinical"][name].fillna("missing")
    for name in ("rna", "wsi"):
        tables[name] = table(args.inputs / f"{name}.csv").reindex(cohort.index)
        values = tables[name].to_numpy(float)
        if not values.shape[1] or not np.isfinite(values).all() or (name == "rna" and (values < 0).any()):
            raise ValueError(f"Incomplete or invalid {name} features")
    report = {"kind": args.kind, "endpoint": args.endpoint, "patients": len(days),
              "events": int(event.sum()), "outer_folds": args.folds, "inner_folds": args.inner,
              "interval_scope": "Conditional patient bootstrap on fixed out-of-fold predictions",
              "runs": {}}
    for seed in args.seeds:
        assignment, scores = predict(tables, days, event, args.folds, args.inner, args.features, args.hidden, seed)
        report["runs"][seed] = evaluate(days, event, assignment, scores, seed, args.bootstrap)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
