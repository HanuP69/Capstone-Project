import pytest

from pipeline import Gate


WEIGHTS = {"w_fact": 0.40, "w_policy": 0.25, "w_ground": 0.35}


@pytest.mark.parametrize("F,P,D,C,decision,label", [
    (0.05, 1.0, 0.90, 0.00, "BLOCK", "false"),
    (0.00, 1.0, 0.90, 0.00, "BLOCK", "false"),
    (1.00, 0.05, 0.90, 0.00, "BLOCK", "false"),
    (1.00, 0.00, 0.90, 0.00, "BLOCK", "false"),
    (1.00, 1.00, 0.00, 0.62, "BLOCK", "false"),
    (1.00, 1.00, 0.5999, 0.62, "BLOCK", "false"),
    (1.00, 1.00, 0.5999, 0.6199, "FLAG", "unverifiable"),
    (1.00, 1.00, 0.6000, 0.62, "PASS", "true"),
    (1.00, 1.00, 0.6000, 0.00, "PASS", "true"),
    (1.00, 1.00, 0.7000, 0.99, "PASS", "true"),
    (0.06, 1.00, 0.00, 0.00, "FLAG", "unverifiable"),
])
def test_gate_truth_table(F, P, D, C, decision, label):
    gate = Gate(weights=WEIGHTS, tau_pass=0.6, tau_contra=0.62)
    _, actual_decision, actual_label = gate.decide(F, P, D, C)
    assert actual_decision == decision
    assert actual_label == label


def test_gate_score_uses_fixed_weights():
    gate = Gate(weights={"w_fact": 1.0, "w_policy": 0.0, "w_ground": 0.0},
                tau_pass=0.6, tau_contra=0.62)
    assert gate.score(0.8, 0.1, 0.2) == pytest.approx(0.8)


def test_gate_score_is_clipped():
    gate = Gate(weights=WEIGHTS, tau_pass=0.6, tau_contra=0.62)
    assert 0.0 <= gate.score(-1.0, -1.0, -1.0) <= 1.0
    assert 0.0 <= gate.score(2.0, 2.0, 2.0) <= 1.0


def test_gate_requires_grounding_even_when_contradiction_is_absent():
    gate = Gate(weights=WEIGHTS, tau_pass=0.6, tau_contra=0.62)
    _, decision, label = gate.decide(1.0, 1.0, 0.0, 0.0)
    assert (decision, label) == ("FLAG", "unverifiable")
