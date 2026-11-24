import pandas as pd
import pytest

from evaluate import (ablation, bootstrap_delta, brier_score, calibration_bins,
                      mcnemar, metrics, normalize_label)


def frame(labels, predictions):
    return pd.DataFrame({"label": labels, "pred": predictions})


def test_normalize_label_handles_aliases_and_whitespace():
    assert normalize_label("YES") == "true"
    assert normalize_label("supported") == "true"
    assert normalize_label("NO") == "false"
    assert normalize_label("not   supported") == "false"
    assert normalize_label("contradicted") == "false"
    assert normalize_label("UNVERIFIABLE") == "unverifiable"
    assert normalize_label(None) == "unverifiable"
    assert normalize_label("nonsense") == "unverifiable"


def test_metrics_perfect_three_way_predictions():
    df = frame(["true", "false", "unverifiable"], ["true", "false", "unverifiable"])
    result = metrics(df, "pred")
    assert result["accuracy"] == 1.0
    assert result["macro_f1"] == 1.0
    assert result["missed_falsehood_rate"] == 0.0
    assert result["abstention_rate"] == pytest.approx(1 / 3, abs=1e-4)


def test_metrics_missed_falsehood_and_abstention():
    df = frame(["false", "false", "true"], ["true", "false", "unverifiable"])
    result = metrics(df, "pred")
    assert result["missed_falsehood_rate"] == 0.5
    assert result["abstention_rate"] == pytest.approx(1 / 3, abs=1e-4)


def test_mcnemar_counts_only_discordant_pairs():
    truth = pd.Series(["true"] * 5)
    better = pd.Series(["true"] * 5)
    worse = pd.Series(["false"] * 4 + ["true"])
    result = mcnemar(better, worse, truth)
    assert result["d_better"] == 4
    assert result["b_better"] == 0
    assert 0.0 <= result["p_value"] <= 1.0


def test_mcnemar_without_discordance_is_not_significant():
    truth = pd.Series(["true", "true"])
    pred = pd.Series(["true", "true"])
    assert mcnemar(pred, pred, truth) == {"d_better": 0, "b_better": 0, "p_value": 1.0}


def test_bootstrap_delta_is_deterministic_and_bounded():
    truth = pd.Series(["true"] * 10)
    better = pd.Series(["true"] * 10)
    worse = pd.Series(["false"] * 10)
    first = bootstrap_delta(better, worse, truth, n=200, seed=7)
    second = bootstrap_delta(better, worse, truth, n=200, seed=7)
    assert first == second
    assert first["delta"] == 1.0
    assert first["ci_low"] == 1.0
    assert first["ci_high"] == 1.0


def test_brier_score_uses_binary_labels_only():
    df = pd.DataFrame({"label": ["true", "false", "unverifiable"], "D": [0.9, 0.1, 0.5]})
    assert brier_score(df) == 0.01


def test_calibration_bins_cover_all_binary_rows():
    df = pd.DataFrame({"label": ["true", "false"], "D": [0.05, 0.95]})
    bins = calibration_bins(df)
    assert sum(row["n"] for row in bins) == 2
    assert all(0.0 <= row["mean_pred"] <= 1.0 for row in bins)


def test_ablation_produces_all_rules_and_cascade_matches_predictions():
    df = pd.DataFrame({
        "label": ["true", "false"],
        "pred_d": ["true", "false"],
        "F": [1.0, 0.0],
        "P": [1.0, 0.0],
        "D": [0.9, 0.1],
        "G": [0.9, 0.1],
    })
    rules = ablation(df)
    assert set(rules["rule"]) == {"cascade", "d_only", "f_only", "p_only", "g_only"}
    cascade = rules.set_index("rule").loc["cascade"]
    assert cascade["accuracy"] == 1.0
    assert cascade["missed_falsehood"] == 0.0
