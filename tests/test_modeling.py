import numpy as np
import pandas as pd

from modeling import cluster_bootstrap, feature_contract, patient_disjoint_split, threshold_table


def test_group_split_is_patient_disjoint():
    x = pd.DataFrame({"x": range(40)})
    y = pd.Series([0, 1] * 20)
    groups = pd.Series([f"p{i // 2}" for i in range(40)])
    train, val, test = patient_disjoint_split(x, y, groups)
    assert set(groups.iloc[train]).isdisjoint(groups.iloc[val])
    assert set(groups.iloc[train]).isdisjoint(groups.iloc[test])
    assert set(groups.iloc[val]).isdisjoint(groups.iloc[test])


def test_feature_contract_keeps_patient_only_as_group():
    frame = pd.DataFrame({"patient_nbr": [1, 2], "encounter_id": [9, 10], "readmitted": ["<30", "NO"], "discharge_disposition_id": [1, 1], "x": [3, 4]})
    frame["readmit_30d"] = [1, 0]
    x, y, groups = feature_contract(frame)
    assert "patient_nbr" not in x and "encounter_id" not in x
    assert groups.tolist() == ["1", "2"]
    assert y.tolist() == [1, 0]


def test_cluster_bootstrap_is_deterministic_and_has_interval():
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    p = np.array([.1, .8, .2, .7, .2, .6, .3, .9])
    groups = np.array(["a", "a", "b", "b", "c", "c", "d", "d"])
    a = cluster_bootstrap(y, p, groups, n_boot=20)
    b = cluster_bootstrap(y, p, groups, n_boot=20)
    assert a == b
    assert 0 <= a["roc_auc"]["low_2.5"] <= a["roc_auc"]["high_97.5"] <= 1


def test_threshold_table_contains_operating_metrics():
    out = threshold_table([0, 1, 0, 1], [.1, .8, .4, .9], thresholds=[.5])
    assert out.loc[0, "recall"] == 1
    assert out.loc[0, "precision"] == 1
