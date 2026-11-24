import pytest

from quality import has_placeholder, is_clean_absent, is_fragment, is_hedge, is_well_formed


@pytest.mark.parametrize("text", [
    "The fee varies by institution.",
    "Please check your department portal.",
    "The amount typically depends on the programme.",
    "Consult the academic office for the exact policy.",
])
def test_hedges_are_detected(text):
    assert is_hedge(text)


def test_plain_factual_sentence_is_not_a_hedge():
    assert not is_hedge("The minimum attendance is 75 percent.")


@pytest.mark.parametrize("text", [
    "The fee is $X.",
    "Submit the form by [insert date].",
    "The fee is TBD.",
    "Contact <office> for details.",
])
def test_placeholders_are_detected(text):
    assert has_placeholder(text)


@pytest.mark.parametrize("text", [
    "15000/-) and Mess Fee (Rs.18000/-) for first semester",
    "the fee is 1000",
    "In the meeting",
    "The fee is 1000",
    "The fee is 1000 Note: -1.",
    "The fee is 1000 (for hostellers",
])
def test_fragments_are_rejected(text):
    assert is_fragment(text)
    assert not is_well_formed(text)


def test_complete_sentence_is_well_formed():
    text = "The minimum attendance required for end semester examinations is 75 percent."
    assert not is_fragment(text)
    assert is_well_formed(text)
    assert is_clean_absent(text)


def test_clean_absent_rejects_hedge_and_placeholder():
    assert not is_clean_absent("The fee varies by institution.")
    assert not is_clean_absent("The fee is $X.")
