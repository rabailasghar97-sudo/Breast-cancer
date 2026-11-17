# Breast cancer multimodal prognosis research

Research benchmark for progression-free interval (PFI) in TCGA-BRCA. Inputs are clinical predictors, primary-tumor RNA expression and diagnostic whole-slide histology features. Overall survival is supported for already prepared tables. Clinical predictors are age, race and pathologic stage. A study intended for publication needs assessment of predictor timing and baseline receptor fields.

This repository rebuilds the documented core benchmark from the prior project. It does not contain the old commit history or patient data.

## Run

On GitHub, open **Actions → Breast cancer research benchmark → Run workflow** on main. The initialization push also requests a run. GitHub publishes synthetic metrics and, when successful, the real cohort audit and aggregate benchmark metrics. Real patient tables stay on the runner.

On a machine with Python 3.11, curl and access to the data sources:

```sh
python -m pip install -r requirements.txt
python -m breast_cancer.prepare --output outputs/real
python -m breast_cancer.study --inputs outputs/real --output outputs/real/metrics.json --kind real
```

Use `--audit-only` with the data preparation command to check overlap before downloading embeddings.

The data path checks the [TCGA-CDR](https://gdc.cancer.gov/about-data/publications/PanCan-Clinical-2018) workbook and the [archived GDC STAR-TPM matrix](https://doi.org/10.5281/zenodo.22670234) against expected checksums. It pins the [published UNI2-h features](https://huggingface.co/datasets/sofieneb/uni2h_features) to a revision. These features are supplied for noncommercial academic research and are not redistributed in this repository. It selects one primary-tumor RNA specimen and one diagnostic slide per patient by barcode ordering and mean-pools finite patch vectors. The approach avoids downloading raw slides or running a pathology encoder here. Record exclusions and confirm sample/time eligibility before treating matches as a final cohort.

## Evaluation

The study scores 14 Cox models on identical stratified patient folds: three unimodal MLPs; four early fusion MLPs; four late fusion stacks; and three linear controls. Imputation, encoding, scaling and RNA/image feature selection occur inside training folds. The late-fusion stack learns from inner out-of-fold risk scores. Harrell concordance compares patients within the same held-out fold. Paired bootstrap intervals condition on fixed predictions. Repeated seeds are sensitivity runs on the same patients.

Previous project evidence:

- The [real metadata audit](https://github.com/rabailasghar097-bit/prostate-cancer-progression/actions/runs/43891085249) found 1,058 candidate matches with 139 PFI events.
- The [raw-file pilot](https://github.com/rabailasghar097-bit/prostate-cancer-progression/actions/runs/43892337387) processed 48 patients, including 3 events, and trained 14 models. It used handcrafted slide thumbnail features. Its scores are feasibility results.
- The larger cohort run, native patch attention benchmarking and independent cohort validation require completion and verification. This repository currently implements the mean-pooled embedding benchmark. The prediction tool is deferred.

Before claiming external validation, establish compatible endpoint definitions, collection times, RNA assay preprocessing and slide feature extraction in an independent cohort. Consider CPTAC-BRCA as a candidate after verifying those requirements. Report cohort flow and follow [TRIPOD+AI](https://www.bmj.com/content/385/bmj-2023-078378). Acknowledge the TCGA Research Network and the data and feature providers.
