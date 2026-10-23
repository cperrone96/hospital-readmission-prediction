import io

import numpy as np
import pandas as pd
import pytest

from modeling import (
    calibration_intercept_slope,
    calibration_table,
    feature_contract,
    icd9_chapter,
    load_and_contract,
    operating_points,
    patient_cluster_bootstrap,
    patient_disjoint_split,
)


def _toy_frame(n_patients: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for patient in range(n_patients):
        for visit in range(rng.integers(1, 4)):
            rows.append(
                {
                    "patient_nbr": f"p{patient}",
                    "encounter_id": f"e{patient}_{visit}",
                    "readmit_30d": int(rng.random() < 0.3),
                    "readmitted": "NO",
                    "x": rng.normal(),
                }
            )
    return pd.DataFrame(rows)


def test_split_is_patient_disjoint_and_covers_every_row():
    frame = _toy_frame()
    X, y, groups = feature_contract(frame)
    train, val, test = patient_disjoint_split(X, y, groups)
    parts = [set(groups.iloc[idx]) for idx in (train, val, test)]
    assert parts[0].isdisjoint(parts[1]) and parts[0].isdisjoint(parts[2]) and parts[1].isdisjoint(parts[2])
    assert sorted(np.concatenate([train, val, test])) == list(range(len(frame)))


def test_feature_contract_excludes_identifiers_and_outcome():
    X, y, groups = feature_contract(_toy_frame())
    assert not {"patient_nbr", "encounter_id", "readmitted", "readmit_30d"} & set(X.columns)
    assert groups.str.startswith("p").all()
    assert set(y.unique()) <= {0, 1}


def test_feature_contract_requires_patient_ids():
    with pytest.raises(ValueError, match="patient_nbr"):
        feature_contract(_toy_frame().drop(columns="patient_nbr"))


def test_loader_keeps_not_measured_as_a_category_and_drops_death_or_hospice():
    header = (
        "encounter_id,patient_nbr,discharge_disposition_id,max_glu_serum,diag_1,diag_2,diag_3,readmitted,"
        + ",".join(
            [
                "time_in_hospital",
                "num_lab_procedures",
                "num_procedures",
                "num_medications",
                "number_outpatient",
                "number_emergency",
                "number_inpatient",
                "number_diagnoses",
            ]
        )
    )
    body = [
        "1,10,1,None,250.83,?,V27,<30,1,1,1,1,1,1,1,1",
        "2,11,11,>300,428,401,?,NO,1,1,1,1,1,1,1,1",  # expired: ineligible
        "3,12,1,?,486,E888,820,>30,1,1,1,1,1,1,1,1",
    ]
    frame = load_and_contract(io.StringIO("\n".join([header, *body])))
    assert frame["encounter_id"].tolist() == ["1", "3"]
    assert frame.loc[0, "max_glu_serum"] == "None"  # test not performed, not missing
    assert pd.isna(frame.loc[1, "max_glu_serum"])  # "?" is the only missing marker
    assert frame["readmit_30d"].tolist() == [1, 0]
    assert frame.loc[0, ["diag_1", "diag_2", "diag_3"]].tolist() == ["diabetes", "missing", "supplementary"]


@pytest.mark.parametrize(
    "code, chapter",
    [
        ("250.01", "diabetes"),
        ("428", "circulatory"),
        ("785", "circulatory"),
        ("486", "respiratory"),
        ("820", "injury"),
        ("715", "musculoskeletal"),
        ("153", "neoplasms"),
        ("V58", "supplementary"),
        ("296", "other"),
    ],
)
def test_icd9_chapter(code, chapter):
    assert icd9_chapter(code) == chapter


def test_cluster_bootstrap_is_deterministic_and_paired():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    weak = np.clip(0.5 + 0.1 * (y - 0.5) + rng.normal(0, 0.2, 400), 0, 1)
    strong = np.clip(0.5 + 0.4 * (y - 0.5) + rng.normal(0, 0.2, 400), 0, 1)
    groups = np.repeat(np.arange(200), 2).astype(str)
    a = patient_cluster_bootstrap(y, {"weak": weak, "strong": strong}, groups, n_boot=200, paired=("weak", "strong"))
    b = patient_cluster_bootstrap(y, {"weak": weak, "strong": strong}, groups, n_boot=200, paired=("weak", "strong"))
    assert a == b
    assert a["roc_auc_difference_strong_minus_weak"]["low_2.5"] > 0
    assert 0 <= a["strong"]["roc_auc"]["low_2.5"] <= a["strong"]["roc_auc"]["high_97.5"] <= 1


def test_calibration_slope_is_one_for_true_probabilities():
    rng = np.random.default_rng(2)
    p = rng.uniform(0.02, 0.6, 50_000)
    y = rng.random(50_000) < p
    fit = calibration_intercept_slope(y, p)
    assert abs(fit["slope"] - 1) < 0.05 and abs(fit["intercept"]) < 0.05


def test_operating_points_use_validation_cutoffs():
    p_val = np.linspace(0, 1, 101)
    out = operating_points([0, 1, 0, 1], [0.1, 0.95, 0.5, 0.92], p_val, flag_rates=[0.10])
    assert out.loc[0, "score_cutoff"] == pytest.approx(0.90)
    assert out.loc[0, "precision"] == 1 and out.loc[0, "recall"] == 1


def test_committed_results_follow_the_selection_protocol():
    """The reported model is the validation winner and the split is patient-disjoint."""
    import json
    from pathlib import Path

    summary = json.loads((Path(__file__).resolve().parents[1] / "metrics" / "readmission_metrics.json").read_text())
    assert summary["split"]["patient_disjoint"] is True
    validation = summary["validation"]
    assert summary["selected_model"] == max(validation, key=lambda name: validation[name]["roc_auc"])


def test_calibration_table_orders_predicted_and_observed_correctly():
    rng = np.random.default_rng(3)
    p = rng.uniform(0, 0.5, 20_000)
    y = rng.random(20_000) < p / 2  # model over-predicts by a factor of two
    table = calibration_table(y, p)
    assert table["mean_predicted"].is_monotonic_increasing
    assert (table["mean_predicted"] > table["observed_rate"]).all()
