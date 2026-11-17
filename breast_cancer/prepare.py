"""Join checksum-verified clinical/RNA data and published slide features."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile

import h5py
from huggingface_hub import HfApi, hf_hub_download
import numpy as np
from openpyxl import load_workbook
import pandas as pd

CDR_URL = "https://api.gdc.cancer.gov/data/1b5f413e-a8d1-4d10-92eb-7c4ae739ed81"
CDR_MD5 = "a4591b2dcee39591f59e5e25a6ce75fa"
RNA_URL = "https://zenodo.org/records/22670234/files/tcga-expression-explorer-data-v1.0.tar.gz?download=1"
RNA_SHA = "f1776fc80b6362d9b434c8ada9c7e6079155ba143151d3f349059f622b96b963"
REPO = "sofieneb/uni2h_features"
PATIENT = r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})"
RNA = re.compile(PATIENT + r"-01[A-Z](?:-.*)?$")
SLIDE = re.compile(PATIENT + r"-01[A-Z]-[A-Z0-9]{2}-DX\d+(?:\..+)?\.h5$", re.I)


def download(url, path, algorithm, expected):
    subprocess.run(["curl", "-fsSL", "--retry", "5", "--retry-all-errors", "--connect-timeout",
                    "20", "--max-time", "600", "--output", str(path), url], check=True)
    with path.open("rb") as source:
        if hashlib.file_digest(source, algorithm).hexdigest() != expected:
            raise ValueError(f"Unexpected {algorithm} checksum for {path.name}")


def primary(paths, pattern):
    chosen = {}
    for path in sorted(paths):
        match = pattern.fullmatch(Path(path).name)
        if match:
            chosen.setdefault(match[1].upper(), path)
    return chosen


def clinical_table(path):
    book = load_workbook(BytesIO(path.read_bytes()), read_only=True, data_only=True)
    names = ["bcr_patient_barcode", "type", "PFI.time", "PFI",
             "age_at_initial_pathologic_diagnosis", "race", "ajcc_pathologic_tumor_stage"]
    target = ["patient_id", "type", "pfi_days", "pfi_event", "age", "race", "tumour_stage"]
    try:
        for sheet in book.worksheets:
            values = iter(sheet.values)
            for _ in range(10):
                header = list(next(values, ()))
                if set(names).issubset(header):
                    cols = [header.index(name) for name in names]
                    frame = pd.DataFrame(([row[i] for i in cols] for row in values), columns=target)
                    frame = frame[frame.type.eq("BRCA")].drop(columns="type")
                    for col in ["pfi_days", "pfi_event", "age"]:
                        frame[col] = pd.to_numeric(frame[col], errors="coerce")
                    frame = frame[frame.pfi_days.gt(0) & frame.pfi_event.isin([0, 1])]
                    return frame.set_index("patient_id", verify_integrity=True)
    finally:
        book.close()
    raise ValueError("No eligible TCGA-CDR worksheet found")


def rna_table(archive, work):
    target = work / "tpm.tsv.gz"
    with tarfile.open(archive, "r:gz") as package:
        for member in package:
            if member.isfile() and member.name.endswith("/BRCA_new_TPM.tsv.gz"):
                with package.extractfile(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
                break
        else:
            raise ValueError("BRCA TPM matrix not found")
    header = pd.read_csv(target, sep="\t", nrows=0)
    chosen = primary(header.columns, RNA)
    if not chosen:
        raise ValueError("No primary tumor RNA barcodes found")
    gene = header.columns[0]
    matrix = pd.read_csv(target, sep="\t", usecols=[gene, *chosen.values()],
                         dtype={col: np.float32 for col in chosen.values()}).set_index(gene)
    if matrix.empty or matrix.index.has_duplicates or not matrix.index.astype(str).str.startswith("ENSG").all():
        raise ValueError("Invalid Ensembl row identifiers")
    return matrix.rename(columns={sample: patient for patient, sample in chosen.items()}).T.rename_axis("patient_id")


def pool(path):
    with h5py.File(path, "r") as handle:
        features = handle["features"]
        if features.ndim != 2 or len(features) < 8 or features.shape[1] != 1536:
            raise ValueError("Expected at least eight UNI2-h patches and 1536 features")
        total = np.zeros(1536, dtype=np.float64)
        for start in range(0, len(features), 512):
            block = features[start:start+512]
            if not np.isfinite(block).all():
                raise ValueError("Nonfinite patch features")
            total += block.sum(axis=0, dtype=np.float64)
        return total / len(features)


def run(output, revision, audit_only):
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as name:
        work = Path(name)
        download(CDR_URL, work / "cdr.xlsx", "md5", CDR_MD5)
        clinical = clinical_table(work / "cdr.xlsx")
        api = HfApi()
        pinned = api.dataset_info(REPO, revision=revision).sha
        if not pinned:
            raise ValueError("Missing embedding dataset commit")
        slides = primary((f for f in api.list_repo_files(REPO, repo_type="dataset", revision=pinned)
                          if f.startswith("TCGA/TCGA-BRCA/")), SLIDE)
        download(RNA_URL, work / "rna.tar.gz", "sha256", RNA_SHA)
        rna = rna_table(work / "rna.tar.gz", work)
        candidates = sorted(set(clinical.index) & set(rna.index) & slides.keys())
        summary = {"clinical_pfi": len(clinical), "primary_rna": len(rna),
                   "primary_diagnostic_embeddings": len(slides), "matched_candidates": len(candidates),
                   "candidate_pfi_events": int(clinical.loc[candidates].pfi_event.sum()),
                   "cdr_md5": CDR_MD5, "rna_archive_sha256": RNA_SHA,
                   "embedding_repo": REPO, "embedding_revision": pinned}
        (output / "cohort_audit.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2), flush=True)
        if audit_only:
            return
        if not candidates:
            raise ValueError("No patient overlap across the three sources")

        def process(patient):
            try:
                with tempfile.TemporaryDirectory(dir=work) as cache:
                    path = hf_hub_download(REPO, slides[patient], repo_type="dataset", revision=pinned,
                                           cache_dir=cache)
                    return pool(path)
            except (OSError, ValueError, KeyError) as error:
                print(f"One slide failed processing ({type(error).__name__})", flush=True)
                return None

        vectors = {}
        with ThreadPoolExecutor(max_workers=6) as executor:
            for n, (patient, vector) in enumerate(zip(candidates, executor.map(process, candidates)), 1):
                if vector is not None:
                    vectors[patient] = vector
                if n % 50 == 0:
                    print(f"Checked {n} slides; {len(vectors)} passed", flush=True)
        ids = sorted(vectors)
        if not ids:
            raise ValueError("No matched slide passed download and feature checks")
        clinical.loc[ids].to_csv(output / "cohort.csv")
        rna.loc[ids].to_csv(output / "rna.csv")
        pd.DataFrame.from_dict(vectors, orient="index", columns=[f"uni2h_{i:04d}" for i in range(1536)]).rename_axis("patient_id").to_csv(output / "wsi.csv")
        summary.update(processed=len(ids), pfi_events=int(clinical.loc[ids].pfi_event.sum()))
        (output / "processing.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    run(args.output, args.revision, args.audit_only)
