import pytest

from stage1 import cleaned_text_to_ids, phones_to_cleaned_text, strip_arpabet_stress


def test_arpabet_realized_substitution_maps_to_matcha_ipa_symbols():
    cleaned = phones_to_cleaned_text("DH AH W AE B IH T", "arpabet")

    assert cleaned == "ðəwæbɪt"


def test_arpabet_supports_different_lengths_for_deletions_and_insertions():
    canonical = phones_to_cleaned_text("DH AH R AE B IH T", "arpabet")
    deletion = phones_to_cleaned_text("DH AH AE B IH T", "arpabet")
    insertion = phones_to_cleaned_text("DH AH R W AE B IH T", "arpabet")

    assert len(deletion) < len(canonical)
    assert len(insertion) > len(canonical)


def test_arpabet_stress_digits_are_ignored():
    assert strip_arpabet_stress("AH0") == "AH"
    assert phones_to_cleaned_text("R AE1 B IH0 T", "arpabet") == "ɹæbɪt"


def test_word_boundary_tokens_become_spaces():
    assert phones_to_cleaned_text("DH AH | R AE B IH T", "arpabet") == "ðə ɹæbɪt"


def test_ipa_input_bypasses_arpabet_mapping():
    assert phones_to_cleaned_text("ð ə w æ b ɪ t", "ipa") == "ðəwæbɪt"


def test_cleaned_text_to_ids_inserts_blank_tokens_by_default():
    token_ids = cleaned_text_to_ids("ðə", add_blank=True)

    assert len(token_ids) == 5
    assert token_ids[0] == 0
    assert token_ids[2] == 0
    assert token_ids[4] == 0


def test_unknown_arpabet_phone_raises_helpful_error():
    with pytest.raises(ValueError, match="Unknown ARPAbet phone"):
        phones_to_cleaned_text("DH XX", "arpabet")
