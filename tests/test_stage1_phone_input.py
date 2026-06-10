import argparse
import inspect

import pytest

from stage1 import (
    arpabet_phone_to_ipa,
    cleaned_text_to_ids,
    phones_to_cleaned_text,
    read_phone_input,
    strip_arpabet_stress,
    synthesize_from_phones,
)


def test_arpabet_realized_substitution_maps_to_matcha_ipa_symbols():
    cleaned = phones_to_cleaned_text("DH AH W AE B IH T", "arpabet")

    assert cleaned == "ðəwæbɪt"


def test_arpabet_supports_different_lengths_for_deletions_and_insertions():
    canonical = phones_to_cleaned_text("DH AH R AE B IH T", "arpabet")
    deletion = phones_to_cleaned_text("DH AH AE B IH T", "arpabet")
    insertion = phones_to_cleaned_text("DH AH R W AE B IH T", "arpabet")

    assert len(deletion) < len(canonical)
    assert len(insertion) > len(canonical)


def test_arpabet_stress_digits_are_available_for_base_phone_lookup():
    assert strip_arpabet_stress("AH0") == "AH"


def test_arpabet_primary_and_secondary_stress_are_preserved():
    assert arpabet_phone_to_ipa("AE1") == "ˈæ"
    assert arpabet_phone_to_ipa("AE2") == "ˌæ"
    assert arpabet_phone_to_ipa("AE0") == "æ"
    assert phones_to_cleaned_text("R AE1 B IH0 T", "arpabet") == "ɹˈæbɪt"


def test_word_boundary_tokens_become_spaces():
    assert phones_to_cleaned_text("DH AH0 | R AE1 B IH0 T", "arpabet") == "ðə ɹˈæbɪt"


def test_phone_words_insert_word_boundaries():
    args = argparse.Namespace(
        phones=None,
        phones_file=None,
        phone_words=["DH AH0", "R AE1 B IH0 T"],
    )

    assert read_phone_input(args) == "DH AH0 | R AE1 B IH0 T"
    assert phones_to_cleaned_text(read_phone_input(args), "arpabet") == "ðə ɹˈæbɪt"


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


def test_synthesis_vocoding_is_wrapped_in_inference_mode():
    source = inspect.getsource(synthesize_from_phones)

    assert "with torch.inference_mode():" in source
    assert "waveform = to_waveform" in source
