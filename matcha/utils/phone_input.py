"""Utilities for explicit PED-TTS phone input.

These helpers intentionally avoid importing :mod:`matcha.text` so callers can
parse explicit phones without requiring the phonemizer/G2P stack at import time.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from typing import Sequence

ARPABET_TO_IPA = {
    "AA": "ɑ",
    "AE": "æ",
    "AH": "ə",
    "AO": "ɔ",
    "AW": "aʊ",
    "AY": "aɪ",
    "B": "b",
    "CH": "tʃ",
    "D": "d",
    "DH": "ð",
    "EH": "ɛ",
    "ER": "ɚ",
    "EY": "eɪ",
    "F": "f",
    "G": "ɡ",
    "HH": "h",
    "IH": "ɪ",
    "IY": "i",
    "JH": "dʒ",
    "K": "k",
    "L": "l",
    "M": "m",
    "N": "n",
    "NG": "ŋ",
    "OW": "oʊ",
    "OY": "ɔɪ",
    "P": "p",
    "R": "ɹ",
    "S": "s",
    "SH": "ʃ",
    "T": "t",
    "TH": "θ",
    "UH": "ʊ",
    "UW": "u",
    "V": "v",
    "W": "w",
    "Y": "j",
    "Z": "z",
    "ZH": "ʒ",
}

VOWEL_ARPABET = {
    "AA",
    "AE",
    "AH",
    "AO",
    "AW",
    "AY",
    "EH",
    "ER",
    "EY",
    "IH",
    "IY",
    "OW",
    "OY",
    "UH",
    "UW",
}
WORD_BOUNDARY_TOKENS = {"|", "/", "SP", "SPACE", "PAUSE", "SIL"}
PUNCTUATION_TOKENS = {".", ",", "!", "?", ";", ":"}
_STRESS_RE = re.compile(r"[0-2]$")


def get_matcha_symbols() -> list[str]:
    """Load Matcha-TTS symbols without importing the phonemizer-backed text package."""
    symbols_path = Path(__file__).resolve().parents[1] / "text" / "symbols.py"
    spec = importlib.util.spec_from_file_location("matcha_text_symbols_for_phone_input", symbols_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.symbols


def strip_arpabet_stress(phone: str) -> str:
    """Remove CMUdict-style stress digits from an ARPAbet phone token."""
    return _STRESS_RE.sub("", phone.upper())


def get_arpabet_stress(phone: str) -> str | None:
    """Return a CMUdict-style stress digit if one is present."""
    phone = phone.strip()
    return phone[-1] if phone and phone[-1] in {"0", "1", "2"} else None


def arpabet_phone_to_ipa(phone: str) -> str | None:
    """Convert one ARPAbet token to IPA, preserving primary/secondary stress."""
    base_phone = strip_arpabet_stress(phone)
    ipa = ARPABET_TO_IPA.get(base_phone)
    if ipa is None:
        return None

    stress = get_arpabet_stress(phone)
    if base_phone in VOWEL_ARPABET and stress == "1":
        return f"ˈ{ipa}"
    if base_phone in VOWEL_ARPABET and stress == "2":
        return f"ˌ{ipa}"
    return ipa


def tokenize_phones(phones: str) -> list[str]:
    """Split a phone string while allowing comma-separated examples."""
    return [token for token in phones.replace(",", " , ").split() if token]


def phone_value_to_string(value: str | Sequence[str] | Sequence[Sequence[str]]) -> str:
    """Normalize a JSON phone value into the string form used by the parser.

    Supported values:
    - ``"DH AH0 | W AE1 B IH0 T"``
    - ``["DH", "AH0", "|", "W", "AE1", "B", "IH0", "T"]``
    - ``[["DH", "AH0"], ["W", "AE1", "B", "IH0", "T"]]``
    """
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence):
        if not value:
            return ""
        if all(isinstance(item, str) for item in value):
            return " ".join(value)
        if all(isinstance(item, Sequence) and not isinstance(item, str) for item in value):
            return " | ".join(" ".join(str(phone) for phone in word) for word in value)
    raise TypeError("Phone values must be a string, a list of phone strings, or a list of per-word phone lists")


def phones_to_cleaned_text(phones: str | Sequence[str] | Sequence[Sequence[str]], phone_format: str) -> str:
    """Convert explicit phones to a cleaned IPA-symbol string accepted by Matcha-TTS."""
    if phone_format not in {"arpabet", "ipa"}:
        raise ValueError(f"Unsupported phone format: {phone_format}")

    phone_string = phone_value_to_string(phones)
    pieces: list[str] = []
    unknown_phones: list[str] = []
    for token in tokenize_phones(phone_string):
        upper_token = token.upper()
        if upper_token in WORD_BOUNDARY_TOKENS:
            pieces.append(" ")
        elif token in PUNCTUATION_TOKENS:
            pieces.append(token)
        elif phone_format == "arpabet":
            ipa_phone = arpabet_phone_to_ipa(token)
            if ipa_phone is not None:
                pieces.append(ipa_phone)
            else:
                unknown_phones.append(token)
        else:
            pieces.append(token)

    if unknown_phones:
        known = ", ".join(sorted(ARPABET_TO_IPA))
        unknown = ", ".join(unknown_phones)
        raise ValueError(f"Unknown ARPAbet phone(s): {unknown}. Known phones: {known}")

    cleaned_text = "".join(pieces)
    validate_cleaned_text_symbols(cleaned_text)
    return cleaned_text


def validate_cleaned_text_symbols(cleaned_text: str) -> None:
    """Validate that every character is in the current Matcha-TTS symbol inventory."""
    allowed_symbols = set(get_matcha_symbols())
    unknown_symbols = sorted({symbol for symbol in cleaned_text if symbol not in allowed_symbols})
    if unknown_symbols:
        unknown = " ".join(repr(symbol) for symbol in unknown_symbols)
        raise ValueError(f"Phone string contains symbol(s) outside Matcha-TTS vocabulary: {unknown}")


def intersperse_blank(sequence: list[int], blank_id: int = 0) -> list[int]:
    """Insert Matcha-TTS blank tokens around and between IDs."""
    result = [blank_id] * (len(sequence) * 2 + 1)
    result[1::2] = sequence
    return result


def cleaned_text_to_ids(cleaned_text: str, add_blank: bool = True) -> list[int]:
    """Convert a cleaned IPA-symbol string to Matcha-TTS token IDs."""
    symbol_to_id = {symbol: index for index, symbol in enumerate(get_matcha_symbols())}
    token_ids = [symbol_to_id[symbol] for symbol in cleaned_text]
    if add_blank:
        token_ids = intersperse_blank(token_ids, 0)
    return token_ids


def phones_to_ids(
    phones: str | Sequence[str] | Sequence[Sequence[str]], phone_format: str = "arpabet", add_blank: bool = True
) -> tuple[list[int], str, str]:
    """Convert phones to Matcha token IDs.

    Returns ``(token_ids, cleaned_text, phone_string)``.
    """
    phone_string = phone_value_to_string(phones)
    cleaned_text = phones_to_cleaned_text(phone_string, phone_format)
    return cleaned_text_to_ids(cleaned_text, add_blank=add_blank), cleaned_text, phone_string
