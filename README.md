# Predicting 30-Day Hospital Readmission

This educational analysis estimates whether an eligible diabetic inpatient encounter is followed by a readmission within 30 days. The unit is an index encounter and the prediction timestamp is the end of that encounter (discharge). Features must be observed during the index encounter; identifiers are never model features.

## Design

The source is the Diabetes 130-US Hospitals (1999–2008) dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008), DOI [10.24432/C53919](https://doi.org/10.24432/C53919). Encounters ending in death or hospice are outside the estimand because they cannot subsequently be readmitted. `modeling.py` retains `patient_nbr` only to create a patient-disjoint 60/20/20 train/validation/test split when that source field is present. The local legacy export is de-identified and lacks those two UCI ID columns; the reproducibility gate reports this limitation rather than creating fake patient IDs. Use the authoritative UCI file for the grouped evaluation.

The selected logistic model is compared with a training-prevalence baseline. The held-out report includes ROC-AUC, PR-AUC, Brier score, quantile calibration, cluster-bootstrap intervals, threshold trade-offs, subgroup diagnostics, and standardized-coefficient interpretation. There is no hospital identifier or clinical timestamp in this export, so hospital-aware and chronological sensitivity analyses are not supported.

## Run

```bash
python -m pip install -r requirements.txt
python modeling.py
python -m pytest -q
```

The checked-in data checksum is recorded in `DATA_PROVENANCE.md`. The notebook is a clean-run presentation of the same analysis. This is a public, retrospective prediction exercise—not a clinical decision tool, treatment recommendation, or evidence of causal risk factors. Performance may not transport to another hospital, time period, population, or workflow; deployment would require local validation, calibration monitoring, fairness review, governance, and clinician oversight.
