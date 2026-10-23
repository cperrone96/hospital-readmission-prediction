# Predicting 30-Day Hospital Readmission

Which diabetic inpatients are most likely to be readmitted within 30 days of discharge? Hospitals are penalized for excess readmissions and can only follow up with a fraction of discharged patients, so a calibrated risk score at discharge helps target that follow-up. This project builds and validates one on 101,766 encounters from the UCI Diabetes 130-US Hospitals dataset.

## Key results

Held-out test set: 19,861 encounters from 13,998 patients who appear in neither training nor validation. 30-day readmission rate: 11.3%. Intervals are 95% patient-cluster bootstrap intervals (1,000 resamples).

| Model | ROC-AUC (95% CI) | PR-AUC (95% CI) | Brier score |
|---|---|---|---|
| Prevalence baseline | 0.500 | 0.113 | 0.101 |
| Logistic regression (L2) | 0.664 (0.650 to 0.679) | 0.215 (0.197 to 0.238) | 0.096 |
| **Gradient boosting (selected)** | **0.675 (0.662 to 0.691)** | **0.235 (0.214 to 0.259)** | **0.095** |

- **Model comparison:** gradient boosting was selected on validation ROC-AUC before the test set was scored. On test it beats logistic regression by 0.012 ROC-AUC, paired 95% CI 0.007 to 0.017.
- **Calibration:** calibration slope 1.02 and intercept 0.03 on held-out patients (ideal: 1 and 0), so predicted risks can be read as probabilities.
- **Operating point:** flagging the top 10% of discharges (cutoff fixed on validation) catches 26% of readmissions at 27.9% precision, 2.5 times the base rate.
- **Drivers:** prior-year inpatient visits and discharge disposition dominate permutation importance. Diagnosis group, specialty and payer add little.

![ROC and calibration curves on held-out patients](figures/roc_calibration.png)

## Approach

- **Patient-disjoint 60/20/20 split.** 23% of patients have more than one encounter, accounting for 46% of all encounters. Splitting by `patient_nbr` (`GroupShuffleSplit`) keeps every patient in exactly one partition, so test performance isn't inflated by patients the model has already seen.
- **Discharge-time feature contract.** Only fields known at discharge are features. Identifiers are excluded. Encounters ending in death or hospice are removed because they can't be readmitted.
- **Clinically meaningful encoding.** Admission, discharge and source codes are categorical. ICD-9 diagnoses are grouped into the chapters used by Strack et al. (2014). "Test not performed" for HbA1c and glucose is its own category, not a missing value.
- **Two model families, tuned on validation only.** L2-regularized logistic regression (log-transformed, standardized counts; one-hot categories) and `HistGradientBoostingClassifier` with native categorical splits. Early stopping is off because its internal holdout splits by row; the number of boosting rounds is tuned on the patient-disjoint validation set instead. The test set is scored once.
- **Uncertainty that respects patient clustering.** A patient-cluster bootstrap gives intervals for each model and a paired interval for the difference between them.
- **Evaluation beyond AUC.** PR-AUC for the imbalanced outcome, Brier score, log loss, calibration slope and intercept, top-k% operating points, permutation importance and subgroup diagnostics by gender, race and age.

## Data

- **Source:** [Diabetes 130-US Hospitals for Years 1999-2008](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008), UCI Machine Learning Repository, DOI [10.24432/C5230J](https://doi.org/10.24432/C5230J). License: CC BY 4.0.
- **Size:** 101,766 inpatient encounters of diabetic patients from 130 US hospitals, 1999 to 2008. 99,343 encounters from 69,990 patients remain after excluding death and hospice discharges.
- **Unit of analysis:** one encounter, scored at discharge. Target: readmitted in under 30 days.
- The file in `data/` is the unmodified UCI CSV (gzipped). Checksums and rebuild steps are in [DATA_PROVENANCE.md](DATA_PROVENANCE.md).

## Repository structure

```
├── modeling.py                              # data contract, split, models, evaluation, figures
├── hospital_readmission_prediction.ipynb    # walkthrough of the analysis with outputs
├── data/                                    # UCI source file and code tables
├── metrics/                                 # every reported number (JSON and CSV)
├── figures/                                 # ROC/calibration and permutation importance
└── tests/                                   # split, loader, bootstrap and calibration tests
```

## Reproduce

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python modeling.py          # about 2 minutes; rewrites metrics/ and figures/
pytest -q
jupyter nbconvert --to notebook --execute --inplace hospital_readmission_prediction.ipynb
```

Results are deterministic: a rerun reproduces every file in `metrics/` byte for byte.

## Limitations

- Discrimination is modest (ROC-AUC 0.675). The data are administrative: no labs beyond HbA1c and glucose flags, no vitals, social factors or post-discharge information, which caps what any model can learn.
- The source has no hospital identifier or admission dates, so the model can't be validated across sites or forward in time. Performance in a different hospital or a later period is unknown.
- Readmissions to other hospitals aren't captured, so the target undercounts true readmissions.
- Discrimination varies by subgroup: ROC-AUC is 0.628 for patients aged 80 to 90 and 0.547 for those 90 and over (481 test encounters), versus 0.66 to 0.79 for younger age bands. See `metrics/subgroup_metrics.csv`.
- Feature importance describes what the model uses to predict, not what causes readmission.
