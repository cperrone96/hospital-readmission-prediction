# Predicting 30-Day Hospital Readmission

Machine-learning models that flag diabetic inpatients at high risk of being
**readmitted within 30 days** — a costly, CMS-penalized hospital quality metric.
Built on the **Diabetes 130-US Hospitals (1999-2008)** dataset: 101,766
encounters across 130 hospitals.

## Notebook

`hospital_readmission_prediction.ipynb`

- Frames the clinical question and defines a binary 30-day-readmission target
- Cleans the data and removes leakage (identifiers, death/hospice discharges
  that could not be readmitted, >40%-missing columns)
- Builds a leakage-safe `ColumnTransformer` pipeline (median-impute + scale for
  numerics; mode-impute + one-hot for categoricals)
- Trains and compares **Logistic Regression, Random Forest, and Gradient
  Boosting** with class-imbalance handling
- Evaluates with **ROC-AUC and PR-AUC** (the right metric for an 8:1 imbalanced
  screening problem) and tunes an **operating threshold** for a care-management
  triage use case
- Interprets the model to surface the top drivers of readmission risk

## Key results

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| Logistic Regression | 0.65 | 0.21 |
| Random Forest | 0.66 | 0.21 |
| **Gradient Boosting** | **0.67** | **0.23** |

At an operating threshold tuned for ~50% recall, the model surfaces half of all
true 30-day readmissions for follow-up — the intended use as a triage flag that
routes care-management resources to the highest-risk discharges. Top risk
drivers: prior inpatient/emergency utilization, number of diagnoses and
medications, length of stay, and age.

## Skills demonstrated

- End-to-end supervised ML on a large, messy real-world healthcare dataset
- Leakage-safe preprocessing pipelines (`ColumnTransformer`, `Pipeline`)
- Class-imbalance handling and metric selection (PR-AUC, threshold tuning)
- Model comparison, evaluation, and interpretation for a clinical decision

## Data

UCI Machine Learning Repository — *Diabetes 130-US Hospitals for Years
1999-2008* (public). A compressed copy is included (`data/diabetic_data.csv.gz`)
so the notebook runs top-to-bottom.

## Run it

```bash
pip install -r requirements.txt
jupyter lab
```
