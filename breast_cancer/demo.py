"""Generate explicitly synthetic fixtures for a network-independent smoke run."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)
n = 120
ids = [f"TCGA-ZZ-{i:04d}" for i in range(n)]
latent = rng.normal(size=n)
failure = rng.exponential(1500 * np.exp(-latent))
censor = rng.uniform(500, 3000, n)
pd.DataFrame({"patient_id": ids, "pfi_days": np.minimum(failure, censor),
              "pfi_event": (failure <= censor).astype(int), "age": rng.normal(58, 12, n),
              "race": rng.choice(["a", "b"], n), "tumour_stage": rng.choice(["I", "II", "III"], n)
              }).to_csv(args.output / "cohort.csv", index=False)
for name, width in [("rna",24),("wsi",12)]:
    values = rng.normal(size=(n,width))
    values[:, :3] += latent[:, None]
    if name == "rna":
        values = np.exp(values)
    frame = pd.DataFrame(values, columns=[f"{name}_{i}" for i in range(width)])
    frame.insert(0,"patient_id",ids)
    frame.to_csv(args.output / f"{name}.csv", index=False)
