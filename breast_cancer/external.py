"""Score an independently collected, already harmonized cohort."""
import argparse
import json
from pathlib import Path

import numpy as np

from breast_cancer.study import MODALITIES, concordance, features, fit, folds, load_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("development", "external", "output"):
        parser.add_argument(f"--{argument}", required=True, type=Path)
    parser.add_argument("--endpoint", required=True, choices=["pfi", "os"])
    parser.add_argument("--kind", default="real", choices=["real", "synthetic"])
    args = parser.parse_args()
    development, days, events, train = load_inputs(args.development, args.endpoint, args.kind)
    # Independent studies use their own participant identifiers.
    external, test_days, test_events, test = load_inputs(args.external, args.endpoint, "synthetic")
    if len(development.index.intersection(external.index)):
        raise ValueError("Patient IDs overlap between cohorts")
    if args.kind == "real" and external.index.str.startswith("TCGA-").any():
        raise ValueError("Expected a non-TCGA independent cohort")
    if not events.any() or not test_events.any():
        raise ValueError("Both cohorts need observed events")
    for key in MODALITIES:
        if set(train[key].columns) != set(test[key].columns):
            raise ValueError(f"Unequal {key} feature schemas")
        test[key] = test[key][train[key].columns]

    blocks = {key: features(train[key], test[key], key, 64) for key in MODALITIES}
    results = {
        "clinical_cox": fit(*blocks["clinical"], days, events),
        "early_clinical+rna+wsi": fit(
            np.column_stack([blocks[key][0] for key in MODALITIES]),
            np.column_stack([blocks[key][1] for key in MODALITIES]), days, events, 16),
    }
    stacked = np.empty((len(events), len(MODALITIES)))
    for fit_rows, val_rows in folds(events, 4, 42):
        for col, key in enumerate(MODALITIES):
            x, y = features(train[key].iloc[fit_rows], train[key].iloc[val_rows], key, 64)
            stacked[val_rows, col] = fit(x, y, days[fit_rows], events[fit_rows], 16)
    external_scores = np.column_stack([fit(*blocks[key], days, events, 16) for key in MODALITIES])
    results["late_clinical+rna+wsi"] = fit(stacked, external_scores, days, events)

    report = {"kind": args.kind, "endpoint": args.endpoint, "development_patients": len(days),
              "external_patients": len(test_days), "external_events": int(test_events.sum()), "models": {}}
    for name, risk in results.items():
        score, pairs = concordance(test_days, test_events, risk, np.ones(len(risk), int), np.arange(len(risk)))
        report["models"][name] = {"c_index": score, "comparable_pairs": pairs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
