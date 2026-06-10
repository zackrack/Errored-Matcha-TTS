import json

import pytest

torch = pytest.importorskip("torch")

from matcha.data.ped_text_mel_datamodule import PEDTextMelBatchCollate, PEDTextMelDataset, parse_jsonl_records
from matcha.utils.phone_input import phone_value_to_string, phones_to_cleaned_text


def write_jsonl(path, records):
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


def test_phone_value_to_string_supports_token_and_word_lists():
    assert phone_value_to_string(["DH", "AH0", "|", "W", "AE1"]) == "DH AH0 | W AE1"
    assert phone_value_to_string([["DH", "AH0"], ["W", "AE1", "B", "IH0", "T"]]) == ("DH AH0 | W AE1 B IH0 T")
    assert phones_to_cleaned_text([["DH", "AH0"], ["W", "AE1", "B", "IH0", "T"]], "arpabet") == ("ðə wˈæbɪt")


def test_parse_jsonl_records_requires_wav(tmp_path):
    jsonl = tmp_path / "bad.jsonl"
    jsonl.write_text('{"realized_phones": ["DH", "AH0"]}\n', encoding="utf-8")

    with pytest.raises(KeyError, match="wav"):
        parse_jsonl_records(jsonl)


def test_ped_dataset_uses_realized_phones_as_x(tmp_path, monkeypatch):
    jsonl = tmp_path / "train.jsonl"
    write_jsonl(
        jsonl,
        [
            {
                "wav": "wavs/utt_001.wav",
                "text": "The rabbit.",
                "canonical_phones": ["DH", "AH0", "R", "AE1", "B", "IH0", "T"],
                "realized_phones": ["DH", "AH0", "W", "AE1", "B", "IH0", "T"],
                "error_ops": ["none", "none", "sub", "none", "none", "none", "none"],
            }
        ],
    )

    monkeypatch.setattr(PEDTextMelDataset, "get_mel", lambda self, filepath: torch.ones(80, 12))
    dataset = PEDTextMelDataset(jsonl, n_spks=1, phone_format="arpabet", phone_field="realized_phones")
    item = dataset[0]

    assert item["x_text"] == "ðəwˈæbɪt"
    assert item["raw_phones"] == "DH AH0 W AE1 B IH0 T"
    assert item["filepath"].endswith("wavs/utt_001.wav")
    assert item["spk"] is None
    assert item["x"].dtype == torch.int32
    assert item["y"].shape == (80, 12)


def test_ped_collate_pads_variable_realized_phone_lengths(tmp_path, monkeypatch):
    jsonl = tmp_path / "train.jsonl"
    write_jsonl(
        jsonl,
        [
            {"wav": "a.wav", "realized_phones": ["DH", "AH0", "W", "AE1", "B", "IH0", "T"]},
            {"wav": "b.wav", "realized_phones": ["DH", "AH0", "AE1", "B", "IH0", "T"]},
        ],
    )

    monkeypatch.setattr(PEDTextMelDataset, "get_mel", lambda self, filepath: torch.ones(80, 11))
    dataset = PEDTextMelDataset(jsonl, n_spks=1, phone_format="arpabet", phone_field="realized_phones")
    batch = PEDTextMelBatchCollate(n_spks=1)([dataset[0], dataset[1]])

    assert batch["x"].shape[0] == 2
    assert batch["x_lengths"][0] != batch["x_lengths"][1]
    assert batch["y"].shape[1] == 80
    assert batch["spks"] is None
    assert batch["durations"] is None
    assert batch["raw_phones"][0] != batch["raw_phones"][1]


def test_multispeaker_records_map_string_speakers_to_indices(tmp_path, monkeypatch):
    jsonl = tmp_path / "train.jsonl"
    write_jsonl(
        jsonl,
        [
            {"wav": "a.wav", "realized_phones": ["DH", "AH0"], "speaker_id": "child_b"},
            {"wav": "b.wav", "realized_phones": ["DH", "AH0"], "speaker_id": "child_a"},
        ],
    )

    monkeypatch.setattr(PEDTextMelDataset, "get_mel", lambda self, filepath: torch.ones(80, 11))
    dataset = PEDTextMelDataset(jsonl, n_spks=2, phone_format="arpabet", phone_field="realized_phones", seed=1234)
    batch = PEDTextMelBatchCollate(n_spks=2)([dataset[0], dataset[1]])

    assert dataset.speaker_id_to_index == {"child_a": 0, "child_b": 1}
    assert batch["spks"].dtype == torch.long
    assert set(batch["spks"].tolist()) == {0, 1}
