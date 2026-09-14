"""Leakage-aware, patient-disjoint readmission analysis.

The public notebook is intentionally a thin presentation layer around these
functions.  Keeping the split and metrics here makes the clean-run and test
paths identical.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
DEAD_DISCHARGE_IDS = {11, 13, 14, 19, 20, 21}
TARGET = "readmit_30d"
ID_COLUMNS = ["encounter_id", "patient_nbr"]


def load_and_contract(path: str | Path) -> pd.DataFrame:
    """Load the UCI file and apply the documented discharge-time contract.

    The target is defined among encounters discharged alive/not to hospice.
    All index-encounter fields are available by discharge; IDs are retained
    only for grouping and auditability and never enter the feature matrix.
    """
    frame = pd.read_csv(path, low_memory=False).replace("?", np.nan)
    required = {"readmitted", "discharge_disposition_id"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    frame = frame.loc[~frame["discharge_disposition_id"].isin(DEAD_DISCHARGE_IDS)].copy()
    frame[TARGET] = (frame["readmitted"] == "<30").astype(int)
    return frame


def feature_contract(frame: pd.DataFrame, include_discharge: bool = True):
    """Return X, y, patient groups under the explicit feature contract."""
    excluded = {TARGET, "readmitted", *ID_COLUMNS}
    if not include_discharge:
        excluded.add("discharge_disposition_id")
    X = frame.drop(columns=sorted(excluded), errors="ignore").copy()
    y = frame[TARGET].astype(int).copy()
    groups = frame["patient_nbr"].astype(str).copy() if "patient_nbr" in frame else None
    return X, y, groups


def patient_disjoint_split(X, y, groups, random_state: int = RANDOM_STATE):
    """Create 60/20/20 train/validation/test partitions by patient."""
    if groups is None:
        # The checked-in legacy export lacks patient_nbr. This is explicitly a
        # row split, never presented as patient-disjoint.
        from sklearn.model_selection import train_test_split
        idx = np.arange(len(X))
        train_idx, remainder_idx = train_test_split(idx, test_size=.40, stratify=y, random_state=random_state)
        val_idx, test_idx = train_test_split(remainder_idx, test_size=.50, stratify=y.iloc[remainder_idx], random_state=random_state + 1)
        return train_idx, val_idx, test_idx
    first = GroupShuffleSplit(n_splits=1, test_size=0.40, random_state=random_state)
    train_idx, remainder_idx = next(first.split(X, y, groups))
    rem_groups = groups.iloc[remainder_idx]
    second = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=random_state + 1)
    val_rel, test_rel = next(second.split(X.iloc[remainder_idx], y.iloc[remainder_idx], rem_groups))
    val_idx, test_idx = remainder_idx[val_rel], remainder_idx[test_rel]
    return train_idx, val_idx, test_idx


def _pipeline(X: pd.DataFrame) -> Pipeline:
    cat = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num = [c for c in X.columns if c not in cat]
    prep = ColumnTransformer(
        [
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num),
            ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat),
        ],
        remainder="drop",
    )
    return Pipeline([("preprocess", prep), ("model", LogisticRegression(max_iter=400, solver="liblinear", random_state=RANDOM_STATE))])


def _metrics(y: pd.Series, p: np.ndarray, sample_weight=None) -> dict[str, float]:
    return {
        "prevalence": float(np.average(y, weights=sample_weight)),
        "roc_auc": float(roc_auc_score(y, p, sample_weight=sample_weight)),
        "pr_auc": float(average_precision_score(y, p, sample_weight=sample_weight)),
        "brier": float(brier_score_loss(y, p, sample_weight=sample_weight)),
    }


def cluster_bootstrap(y: Iterable[int], p: Iterable[float], groups: Iterable[str], n_boot: int = 50, seed: int = RANDOM_STATE):
    """Patient-cluster bootstrap intervals without pretending rows are IID."""
    y = np.asarray(y)
    p = np.asarray(p)
    groups = np.asarray(groups)
    codes, unique = pd.factorize(groups, sort=False)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        sampled = rng.choice(len(unique), size=len(unique), replace=True)
        weights = np.bincount(sampled, minlength=len(unique))[codes]
        if len(np.unique(y[weights > 0])) < 2:
            continue
        values.append(_metrics(y, p, sample_weight=weights))
    if not values:
        raise ValueError("cluster bootstrap produced no valid resamples")
    out = {}
    for name in ("roc_auc", "pr_auc", "brier"):
        vals = np.array([v[name] for v in values])
        out[name] = {"low_2.5": float(np.quantile(vals, .025)), "high_97.5": float(np.quantile(vals, .975))}
    return out


def threshold_table(y: Iterable[int], p: Iterable[float], thresholds: Iterable[float] | None = None) -> pd.DataFrame:
    y, p = np.asarray(y), np.asarray(p)
    thresholds = thresholds or (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
    rows = []
    for t in thresholds:
        pred = p >= t
        tp = np.sum((pred == 1) & (y == 1)); fp = np.sum((pred == 1) & (y == 0))
        fn = np.sum((pred == 0) & (y == 1)); tn = np.sum((pred == 0) & (y == 0))
        rows.append({"threshold": t, "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1), "flag_rate": float(pred.mean())})
    return pd.DataFrame(rows)


def subgroup_diagnostics(y: pd.Series, p: np.ndarray, frame: pd.DataFrame, columns=("gender", "race", "age")) -> pd.DataFrame:
    rows = []
    for col in columns:
        if col not in frame:
            continue
        for level, idx in frame[col].fillna("Missing").groupby(frame[col].fillna("Missing")).groups.items():
            idx = np.asarray(list(idx))
            if len(idx) < 100 or len(np.unique(y.iloc[idx])) < 2:
                continue
            rows.append({"feature": col, "group": str(level), "n": int(len(idx)), "prevalence": float(y.iloc[idx].mean()), "pr_auc": float(average_precision_score(y.iloc[idx], p[idx])), "brier": float(brier_score_loss(y.iloc[idx], p[idx]))})
    return pd.DataFrame(rows)


def run_analysis(data_path: str | Path, output_dir: str | Path = "metrics", include_discharge: bool = True) -> dict:
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    frame = load_and_contract(data_path)
    X, y, groups = feature_contract(frame, include_discharge=include_discharge)
    train, val, test = patient_disjoint_split(X, y, groups)
    group_values = groups if groups is not None else pd.Series(np.arange(len(frame)).astype(str), index=frame.index)
    pipe = _pipeline(X)
    pipe.fit(X.iloc[train], y.iloc[train])
    val_p = pipe.predict_proba(X.iloc[val])[:, 1]
    test_p = pipe.predict_proba(X.iloc[test])[:, 1]
    # Prevalence is the honest no-skill probability baseline.
    val_baseline = np.repeat(float(y.iloc[train].mean()), len(val))
    test_baseline = np.repeat(float(y.iloc[train].mean()), len(test))
    val_metrics = {"prevalence_baseline": _metrics(y.iloc[val], val_baseline), "logistic_regression": _metrics(y.iloc[val], val_p)}
    test_metrics = {"prevalence_baseline": _metrics(y.iloc[test], test_baseline), "logistic_regression": _metrics(y.iloc[test], test_p)}
    thresholds = threshold_table(y.iloc[test], test_p)
    calibration_prob, calibration_obs = calibration_curve(y.iloc[test], test_p, n_bins=10, strategy="quantile")
    # Logistic coefficients are interpretable after standardization.  Aggregate
    # one-hot coefficients back to their source fields instead of reporting
    # misleading impurity importance from a tree model.
    names = pipe.named_steps["preprocess"].get_feature_names_out()
    coefs = np.abs(pipe.named_steps["model"].coef_[0])
    importance = pd.DataFrame({"encoded_feature": names, "coefficient_abs": coefs})
    importance["feature"] = importance["encoded_feature"].str.replace(r"^(num|cat)__", "", regex=True).str.split("_").str[0]
    importance = importance.groupby("feature", as_index=False)["coefficient_abs"].sum().sort_values("coefficient_abs", ascending=False)
    thresholds.to_csv(out / "threshold_metrics.csv", index=False)
    pd.DataFrame({"predicted_probability": calibration_prob, "observed_fraction": calibration_obs}).to_csv(out / "calibration.csv", index=False)
    importance.to_csv(out / "permutation_importance.csv", index=False)
    subgroup_diagnostics(y.iloc[test].reset_index(drop=True), test_p, frame.iloc[test].reset_index(drop=True)).to_csv(out / "subgroup_metrics.csv", index=False)
    disjoint = groups is not None and bool(set(groups.iloc[train]).isdisjoint(groups.iloc[val]) and set(groups.iloc[train]).isdisjoint(groups.iloc[test]) and set(groups.iloc[val]).isdisjoint(groups.iloc[test]))
    summary = {"data": {"rows_after_eligibility": int(len(frame)), "patients": int(group_values.nunique()) if groups is not None else None, "target_prevalence": float(y.mean()), "prediction_timestamp": "end of index encounter (discharge)", "include_discharge_features": include_discharge, "hospital_or_date_sensitivity": "not supported: source has no hospital identifier or clinical timestamp"}, "split": {"train_rows": int(len(train)), "validation_rows": int(len(val)), "test_rows": int(len(test)), "patient_disjoint": disjoint, "patient_identifier_available": groups is not None}, "validation": val_metrics, "test": test_metrics, "selected_model": "logistic_regression", "cluster_bootstrap_95pct": cluster_bootstrap(y.iloc[test], test_p, group_values.iloc[test]), "thresholds": thresholds.to_dict(orient="records"), "calibration": {"bins": int(len(calibration_prob)), "brier": test_metrics["logistic_regression"]["brier"]}, "interpretation": "absolute standardized logistic coefficients aggregated to source features; associations are predictive signals, not causal effects"}
    (out / "readmission_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    run_analysis(Path(__file__).parent / "data" / "diabetic_data.csv.gz")
