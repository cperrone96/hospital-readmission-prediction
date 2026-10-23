"""30-day readmission prediction on the UCI Diabetes 130-US Hospitals data.

Design in brief:

* Unit of analysis: an inpatient encounter, scored at discharge. Encounters
  that end in death or hospice are excluded because they cannot be readmitted.
* Split: 60/20/20 train/validation/test **by patient** (``patient_nbr``), so no
  patient contributes encounters to more than one partition.
* Models: a training-prevalence baseline, L2 logistic regression and
  histogram gradient boosting. Hyperparameters and the final model are chosen
  on the validation partition only; the test partition is scored once.
* Uncertainty: patient-cluster bootstrap on the test partition, including a
  paired interval for the difference between the two models.

``run_analysis`` writes every number the README reports to ``metrics/`` and
the figures to ``figures/``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score, roc_curve
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

RANDOM_STATE = 42
N_BOOT = 1000
TARGET = "readmit_30d"
ID_COLUMNS = ("encounter_id", "patient_nbr")
# Discharge dispositions meaning death or hospice (see data/IDS_mapping.csv).
DEATH_OR_HOSPICE_IDS = {"11", "13", "14", "19", "20", "21"}
COUNT_FEATURES = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]
DIAGNOSIS_COLUMNS = ("diag_1", "diag_2", "diag_3")
SUBGROUP_COLUMNS = ("gender", "race", "age")

LOGISTIC_GRID: tuple[dict, ...] = tuple({"C": c} for c in (0.01, 0.1, 1.0))
BOOSTING_GRID: tuple[dict, ...] = (
    {"learning_rate": 0.05, "max_iter": 300, "max_leaf_nodes": 31},
    {"learning_rate": 0.05, "max_iter": 500, "max_leaf_nodes": 15},
    {"learning_rate": 0.03, "max_iter": 800, "max_leaf_nodes": 15},
    {"learning_rate": 0.03, "max_iter": 400, "max_leaf_nodes": 15},
)


# --------------------------------------------------------------------------- data


def load_and_contract(path: str | Path) -> pd.DataFrame:
    """Load the UCI file and apply the discharge-time eligibility contract.

    Only ``?`` marks a missing value in the source. The literal ``None`` in
    ``max_glu_serum`` and ``A1Cresult`` means "test not performed", which is
    informative, so it is kept as its own category instead of being parsed as
    missing (pandas would otherwise treat it as NaN).
    """
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=["?"])
    required = {"readmitted", "discharge_disposition_id", "patient_nbr", *COUNT_FEATURES}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    frame = frame.loc[~frame["discharge_disposition_id"].isin(DEATH_OR_HOSPICE_IDS)].copy()
    frame[COUNT_FEATURES] = frame[COUNT_FEATURES].astype(float)
    for col in DIAGNOSIS_COLUMNS:
        frame[col] = frame[col].map(icd9_chapter)
    frame[TARGET] = (frame["readmitted"] == "<30").astype(int)
    return frame.reset_index(drop=True)


def icd9_chapter(code: object) -> str:
    """Map an ICD-9 code to the diagnosis groups used by Strack et al. (2014)."""
    if not isinstance(code, str) or not code:
        return "missing"
    if code[0] in "VE":
        return "supplementary"
    value = float(code)
    whole = int(value)
    if whole == 250:
        return "diabetes"
    if 390 <= whole <= 459 or whole == 785:
        return "circulatory"
    if 460 <= whole <= 519 or whole == 786:
        return "respiratory"
    if 520 <= whole <= 579 or whole == 787:
        return "digestive"
    if 800 <= whole <= 999:
        return "injury"
    if 710 <= whole <= 739:
        return "musculoskeletal"
    if 580 <= whole <= 629 or whole == 788:
        return "genitourinary"
    if 140 <= whole <= 239:
        return "neoplasms"
    return "other"


def feature_contract(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return features, target and patient groups.

    Identifiers and the raw outcome never enter the feature matrix; the
    patient number is returned separately for grouping only.
    """
    if "patient_nbr" not in frame:
        raise ValueError("patient_nbr is required for a patient-disjoint split")
    X = frame.drop(columns=[TARGET, "readmitted", *ID_COLUMNS], errors="ignore").copy()
    y = frame[TARGET].astype(int).copy()
    groups = frame["patient_nbr"].astype(str).copy()
    return X, y, groups


def patient_disjoint_split(
    X: pd.DataFrame, y: pd.Series, groups: pd.Series, random_state: int = RANDOM_STATE
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split rows 60/20/20 into train/validation/test so patients never cross partitions."""
    first = GroupShuffleSplit(n_splits=1, test_size=0.40, random_state=random_state)
    train_idx, remainder_idx = next(first.split(X, y, groups))
    second = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=random_state + 1)
    val_rel, test_rel = next(second.split(X.iloc[remainder_idx], y.iloc[remainder_idx], groups.iloc[remainder_idx]))
    return train_idx, remainder_idx[val_rel], remainder_idx[test_rel]


# ------------------------------------------------------------------------- models


def _categorical_columns(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if c not in COUNT_FEATURES]


def build_logistic(X: pd.DataFrame, C: float) -> Pipeline:
    """L2 logistic regression: log1p + standardized counts, one-hot categories."""
    prep = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                        ("scale", StandardScaler()),
                    ]
                ),
                COUNT_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
                        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50)),
                    ]
                ),
                _categorical_columns(X),
            ),
        ]
    )
    model = LogisticRegression(C=C, solver="newton-cholesky", max_iter=200)
    return Pipeline([("preprocess", prep), ("model", model)])


def build_boosting(X: pd.DataFrame, learning_rate: float, max_iter: int, max_leaf_nodes: int) -> Pipeline:
    """Histogram gradient boosting with native categorical splits.

    Early stopping is off because its internal holdout is a random row split,
    which would put the same patient on both sides. ``max_iter`` is tuned on
    the patient-disjoint validation partition instead.
    """
    cats = _categorical_columns(X)
    prep = ColumnTransformer(
        [
            ("num", "passthrough", COUNT_FEATURES),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
                        (
                            "ordinal",
                            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan, min_frequency=50),
                        ),
                    ]
                ),
                cats,
            ),
        ]
    )
    mask = [False] * len(COUNT_FEATURES) + [True] * len(cats)
    model = HistGradientBoostingClassifier(
        learning_rate=learning_rate,
        max_iter=max_iter,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=100,
        l2_regularization=1.0,
        categorical_features=mask,
        early_stopping=False,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("preprocess", prep), ("model", model)])


def candidate_models(X: pd.DataFrame) -> list[tuple[str, dict, Pipeline]]:
    """Every (family, hyperparameters, pipeline) evaluated on validation."""
    out = [("logistic_regression", params, build_logistic(X, **params)) for params in LOGISTIC_GRID]
    out += [("gradient_boosting", params, build_boosting(X, **params)) for params in BOOSTING_GRID]
    return out


# ------------------------------------------------------------------------ metrics


def classification_metrics(
    y: Iterable[int], p: Iterable[float], sample_weight: np.ndarray | None = None
) -> dict[str, float]:
    """Discrimination (ROC-AUC, PR-AUC) and probability accuracy (Brier, log loss)."""
    y, p = np.asarray(y), np.asarray(p)
    return {
        "roc_auc": float(roc_auc_score(y, p, sample_weight=sample_weight)),
        "pr_auc": float(average_precision_score(y, p, sample_weight=sample_weight)),
        "brier": float(brier_score_loss(y, p, sample_weight=sample_weight)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), sample_weight=sample_weight)),
    }


def calibration_table(y: Iterable[int], p: Iterable[float], n_bins: int = 10) -> pd.DataFrame:
    """Mean predicted risk vs observed rate in equal-count (quantile) bins."""
    observed, predicted = calibration_curve(np.asarray(y), np.asarray(p), n_bins=n_bins, strategy="quantile")
    return pd.DataFrame(
        {"bin": np.arange(1, len(predicted) + 1), "mean_predicted": predicted, "observed_rate": observed}
    )


def calibration_intercept_slope(y: Iterable[int], p: Iterable[float]) -> dict[str, float]:
    """Logistic recalibration of the outcome on logit(p); ideal is intercept 0, slope 1."""
    p = np.clip(np.asarray(p), 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    fit = LogisticRegression(penalty=None, solver="newton-cholesky").fit(logit, np.asarray(y))
    return {"intercept": float(fit.intercept_[0]), "slope": float(fit.coef_[0, 0])}


def patient_cluster_bootstrap(
    y: Iterable[int],
    predictions: dict[str, np.ndarray],
    groups: Iterable[str],
    n_boot: int = N_BOOT,
    seed: int = RANDOM_STATE,
    paired: tuple[str, str] | None = None,
) -> dict:
    """Percentile intervals from resampling whole patients with replacement.

    All models share each resample, so the optional ``paired`` difference
    (second minus first) accounts for the correlation between them.
    """
    y = np.asarray(y)
    codes, unique = pd.factorize(np.asarray(groups), sort=False)
    rng = np.random.default_rng(seed)
    draws: dict[str, list[dict[str, float]]] = {name: [] for name in predictions}
    for _ in range(n_boot):
        weights = np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))[codes]
        if len(np.unique(y[weights > 0])) < 2:
            continue
        for name, p in predictions.items():
            draws[name].append(classification_metrics(y, p, sample_weight=weights))
    if not draws[next(iter(predictions))]:
        raise ValueError("cluster bootstrap produced no valid resamples")

    def interval(values: Sequence[float]) -> dict[str, float]:
        return {"low_2.5": float(np.quantile(values, 0.025)), "high_97.5": float(np.quantile(values, 0.975))}

    out: dict = {
        name: {m: interval([d[m] for d in rows]) for m in ("roc_auc", "pr_auc", "brier")}
        for name, rows in draws.items()
    }
    if paired is not None:
        first, second = paired
        diff = [b["roc_auc"] - a["roc_auc"] for a, b in zip(draws[first], draws[second])]
        out[f"roc_auc_difference_{second}_minus_{first}"] = interval(diff)
    out["n_resamples"] = len(draws[next(iter(predictions))])
    return out


def operating_points(
    y_test: Iterable[int],
    p_test: Iterable[float],
    p_val: Iterable[float],
    flag_rates: Sequence[float] = (0.05, 0.10, 0.20, 0.30),
) -> pd.DataFrame:
    """Precision/recall when flagging the top-k% of encounters.

    The score cutoff for each flag rate is fixed on the validation partition
    and then applied unchanged to test, as it would be in deployment.
    """
    y, p = np.asarray(y_test), np.asarray(p_test)
    prevalence = y.mean()
    rows = []
    for rate in flag_rates:
        cutoff = float(np.quantile(np.asarray(p_val), 1 - rate))
        flagged = p >= cutoff
        tp = int(np.sum(flagged & (y == 1)))
        precision = tp / max(int(flagged.sum()), 1)
        rows.append(
            {
                "target_flag_rate": rate,
                "score_cutoff": cutoff,
                "test_flag_rate": float(flagged.mean()),
                "precision": precision,
                "recall": tp / max(int(y.sum()), 1),
                "lift_over_prevalence": precision / prevalence,
            }
        )
    return pd.DataFrame(rows)


def subgroup_diagnostics(
    y: pd.Series, p: np.ndarray, frame: pd.DataFrame, columns: Sequence[str] = SUBGROUP_COLUMNS, min_n: int = 200
) -> pd.DataFrame:
    """Discrimination and calibration-in-the-large within demographic groups."""
    y = y.reset_index(drop=True)
    frame = frame.reset_index(drop=True)
    rows = []
    for col in columns:
        levels = frame[col].fillna("Missing")
        for level in sorted(levels.unique()):
            idx = np.flatnonzero(levels.to_numpy() == level)
            if len(idx) < min_n or y.iloc[idx].nunique() < 2:
                continue
            rows.append(
                {
                    "feature": col,
                    "group": str(level),
                    "n": int(len(idx)),
                    "observed_rate": float(y.iloc[idx].mean()),
                    "mean_predicted": float(p[idx].mean()),
                    "roc_auc": float(roc_auc_score(y.iloc[idx], p[idx])),
                    "pr_auc": float(average_precision_score(y.iloc[idx], p[idx])),
                }
            )
    return pd.DataFrame(rows)


def logistic_coefficients(pipe: Pipeline) -> pd.DataFrame:
    """Signed coefficients per encoded feature, as odds ratios per unit change."""
    names = pipe.named_steps["preprocess"].get_feature_names_out()
    coef = pipe.named_steps["model"].coef_[0]
    table = pd.DataFrame({"encoded_feature": names, "coefficient": coef, "odds_ratio": np.exp(coef)})
    table["encoded_feature"] = table["encoded_feature"].str.replace(r"^(num|cat)__", "", regex=True)
    return table.reindex(table["coefficient"].abs().sort_values(ascending=False).index).reset_index(drop=True)


# ----------------------------------------------------------------------- figures


def save_figures(
    y_test: pd.Series,
    test_predictions: dict[str, np.ndarray],
    importance: pd.DataFrame,
    labels: dict[str, str],
    selected: str,
    out_dir: Path,
) -> None:
    """ROC + calibration panel and a permutation-importance chart."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    colors = {"logistic_regression": "#2a6fb0", "gradient_boosting": "#d1495b"}
    fig, (ax_roc, ax_cal) = plt.subplots(1, 2, figsize=(11, 4.6))
    for name, p in test_predictions.items():
        fpr, tpr, _ = roc_curve(y_test, p)
        ax_roc.plot(fpr, tpr, color=colors[name], lw=2, label=f"{labels[name]} (AUC {roc_auc_score(y_test, p):.3f})")
        cal = calibration_table(y_test, p)
        ax_cal.plot(cal["mean_predicted"], cal["observed_rate"], "o-", color=colors[name], lw=2, label=labels[name])
    ax_roc.plot([0, 1], [0, 1], "--", color="#888888", lw=1, label="Prevalence baseline (AUC 0.500)")
    ax_roc.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve, held-out patients")
    ax_roc.legend(loc="lower right", frameon=False, fontsize=9)
    ax_cal.plot([0, 1], [0, 1], "--", color="#888888", lw=1, label="Perfect calibration")
    ax_cal.set(
        xlabel="Mean predicted probability (decile)",
        ylabel="Observed 30-day readmission rate",
        title="Calibration, held-out patients",
    )
    ax_cal.set_xlim(0, 0.35)
    ax_cal.set_ylim(0, 0.35)
    ax_cal.legend(loc="upper left", frameon=False, fontsize=9)
    for ax in (ax_roc, ax_cal):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "roc_calibration.png", dpi=150)
    plt.close(fig)

    top10 = importance.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh(top10["feature"], top10["roc_auc_drop_mean"], xerr=top10["roc_auc_drop_std"], color="#d1495b", alpha=0.85)
    ax.set(
        xlabel="Drop in validation ROC-AUC when the column is shuffled",
        title=f"Permutation importance, {labels[selected].lower()}",
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "permutation_importance.png", dpi=150)
    plt.close(fig)


# -------------------------------------------------------------------------- driver


def run_analysis(
    data_path: str | Path,
    output_dir: str | Path = "metrics",
    figure_dir: str | Path | None = "figures",
    n_boot: int = N_BOOT,
) -> dict:
    """Fit, select on validation, score test once, and write all outputs."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame = load_and_contract(data_path)
    X, y, groups = feature_contract(frame)
    train, val, test = patient_disjoint_split(X, y, groups)
    X_train, y_train = X.iloc[train], y.iloc[train]
    X_val, y_val = X.iloc[val], y.iloc[val]
    X_test, y_test = X.iloc[test], y.iloc[test]

    # 1. Model selection on validation only.
    selection_rows, fitted = [], {}
    for family, params, pipe in candidate_models(X):
        pipe.fit(X_train, y_train)
        scores = classification_metrics(y_val, pipe.predict_proba(X_val)[:, 1])
        selection_rows.append({"model": family, "params": json.dumps(params), **scores})
        best = fitted.get(family)
        if best is None or scores["roc_auc"] > best[1]["roc_auc"]:
            fitted[family] = (pipe, scores, params)
    selection = pd.DataFrame(selection_rows)
    selection.to_csv(out / "model_selection.csv", index=False)
    selected = max(fitted, key=lambda name: fitted[name][1]["roc_auc"])

    # 2. Score the test partition once.
    base_rate = float(y_train.mean())
    val_pred = {name: fitted[name][0].predict_proba(X_val)[:, 1] for name in fitted}
    test_pred = {name: fitted[name][0].predict_proba(X_test)[:, 1] for name in fitted}
    test_metrics = {"prevalence_baseline": classification_metrics(y_test, np.full(len(test), base_rate))}
    test_metrics.update({name: classification_metrics(y_test, p) for name, p in test_pred.items()})
    other = "logistic_regression" if selected == "gradient_boosting" else "gradient_boosting"
    uncertainty = patient_cluster_bootstrap(
        y_test, test_pred, groups.iloc[test], n_boot=n_boot, paired=(other, selected)
    )

    # 3. Calibration, operating points, subgroups and interpretation.
    calibration = {name: calibration_intercept_slope(y_test, p) for name, p in test_pred.items()}
    pd.concat([calibration_table(y_test, p).assign(model=name) for name, p in test_pred.items()]).to_csv(
        out / "calibration.csv", index=False
    )
    thresholds = operating_points(y_test, test_pred[selected], val_pred[selected])
    thresholds.to_csv(out / "threshold_metrics.csv", index=False)
    subgroup_diagnostics(y_test, test_pred[selected], frame.iloc[test]).to_csv(
        out / "subgroup_metrics.csv", index=False
    )
    logistic_coefficients(fitted["logistic_regression"][0]).to_csv(out / "logistic_coefficients.csv", index=False)
    perm = permutation_importance(
        fitted[selected][0], X_val, y_val, scoring="roc_auc", n_repeats=5, random_state=RANDOM_STATE, n_jobs=1
    )
    importance = (
        pd.DataFrame(
            {"feature": X.columns, "roc_auc_drop_mean": perm.importances_mean, "roc_auc_drop_std": perm.importances_std}
        )
        .sort_values("roc_auc_drop_mean", ascending=False)
        .reset_index(drop=True)
    )
    importance.to_csv(out / "permutation_importance.csv", index=False)
    labels = {"logistic_regression": "Logistic regression", "gradient_boosting": "Gradient boosting"}
    if figure_dir is not None:
        save_figures(y_test, test_pred, importance, labels, selected, Path(figure_dir))

    split_groups = [set(groups.iloc[idx]) for idx in (train, val, test)]
    summary = {
        "data": {
            "source_rows": int(pd.read_csv(data_path, usecols=["encounter_id"]).shape[0]),
            "rows_after_eligibility": int(len(frame)),
            "patients": int(groups.nunique()),
            "share_of_patients_with_repeat_encounters": float((groups.value_counts() > 1).mean()),
            "share_of_encounters_from_repeat_patients": float((groups.map(groups.value_counts()) > 1).mean()),
            "target_prevalence": float(y.mean()),
            "prediction_timestamp": "end of index encounter (discharge)",
        },
        "split": {
            "train_rows": int(len(train)),
            "validation_rows": int(len(val)),
            "test_rows": int(len(test)),
            "train_patients": len(split_groups[0]),
            "validation_patients": len(split_groups[1]),
            "test_patients": len(split_groups[2]),
            "test_prevalence": float(y_test.mean()),
            "patient_disjoint": bool(
                split_groups[0].isdisjoint(split_groups[1])
                and split_groups[0].isdisjoint(split_groups[2])
                and split_groups[1].isdisjoint(split_groups[2])
            ),
        },
        "selection_metric": "validation ROC-AUC",
        "selected_hyperparameters": {name: fitted[name][2] for name in fitted},
        "validation": {name: fitted[name][1] for name in fitted},
        "selected_model": selected,
        "test": test_metrics,
        "uncertainty_method": f"patient-cluster bootstrap, {uncertainty['n_resamples']} resamples, percentile 95% intervals",
        "uncertainty_95pct": uncertainty,
        "calibration_intercept_slope": calibration,
        "operating_points_selected_model": thresholds.to_dict(orient="records"),
        "top_permutation_importance": importance.head(10).to_dict(orient="records"),
    }
    (out / "readmission_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    run_analysis(root / "data" / "diabetic_data.csv.gz", root / "metrics", root / "figures")
