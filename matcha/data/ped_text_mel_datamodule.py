"""PED-TTS realized-phone JSONL datamodule for Stage 2.

Stage 2 keeps the Matcha-TTS model architecture unchanged. The only semantic
change is that ``batch["x"]`` comes from a JSONL record's realized phone field
instead of from text/G2P.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.utils.data
import torchaudio as ta
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from matcha.utils.audio import mel_spectrogram
from matcha.utils.model import fix_len_compatibility, normalize
from matcha.phone_input import phones_to_ids, ssml_to_ids


def clean_text_segment(text: str) -> str:
    """Phonemize ordinary text segments surrounding SSML phoneme tags."""
    try:
        from matcha.text.cleaners import english_cleaners2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "SSML PED manifests require the phonemizer dependencies used by english_cleaners2. "
            "Install the project dependencies, including phonemizer/espeak, or provide explicit realized_phones instead."
        ) from exc

    return english_cleaners2(text)


class PEDTextMelDataModule(LightningDataModule):
    """Lightning datamodule for realized-phone-only PED-TTS training."""

    def __init__(
        self,
        train_filelist_path,
        valid_filelist_path,
        batch_size,
        num_workers,
        pin_memory,
        phone_format,
        phone_field,
        add_blank,
        n_spks,
        n_fft,
        n_feats,
        sample_rate,
        hop_length,
        win_length,
        f_min,
        f_max,
        data_statistics,
        seed,
        load_durations,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

    def setup(self, stage: Optional[str] = None):  # pylint: disable=unused-argument
        self.trainset = PEDTextMelDataset(  # pylint: disable=attribute-defined-outside-init
            self.hparams.train_filelist_path,
            self.hparams.n_spks,
            self.hparams.phone_format,
            self.hparams.phone_field,
            self.hparams.add_blank,
            self.hparams.n_fft,
            self.hparams.n_feats,
            self.hparams.sample_rate,
            self.hparams.hop_length,
            self.hparams.win_length,
            self.hparams.f_min,
            self.hparams.f_max,
            self.hparams.data_statistics,
            self.hparams.seed,
            self.hparams.load_durations,
        )
        self.validset = PEDTextMelDataset(  # pylint: disable=attribute-defined-outside-init
            self.hparams.valid_filelist_path,
            self.hparams.n_spks,
            self.hparams.phone_format,
            self.hparams.phone_field,
            self.hparams.add_blank,
            self.hparams.n_fft,
            self.hparams.n_feats,
            self.hparams.sample_rate,
            self.hparams.hop_length,
            self.hparams.win_length,
            self.hparams.f_min,
            self.hparams.f_max,
            self.hparams.data_statistics,
            self.hparams.seed,
            self.hparams.load_durations,
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.trainset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=PEDTextMelBatchCollate(self.hparams.n_spks),
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.validset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=PEDTextMelBatchCollate(self.hparams.n_spks),
        )

    def teardown(self, stage: Optional[str] = None):  # pylint: disable=unused-argument
        pass  # pylint: disable=unnecessary-pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):  # pylint: disable=unused-argument
        pass  # pylint: disable=unnecessary-pass


class PEDTextMelDataset(torch.utils.data.Dataset):
    """JSONL dataset that uses realized phones as Matcha-TTS input tokens."""

    def __init__(
        self,
        filelist_path,
        n_spks,
        phone_format="arpabet",
        phone_field="realized_phones",
        add_blank=True,
        n_fft=1024,
        n_mels=80,
        sample_rate=22050,
        hop_length=256,
        win_length=1024,
        f_min=0.0,
        f_max=8000,
        data_parameters=None,
        seed=None,
        load_durations=False,
    ):
        self.filelist_path = Path(filelist_path)
        self.records = parse_jsonl_records(self.filelist_path)
        self.n_spks = n_spks
        self.phone_format = phone_format
        self.phone_field = phone_field
        self.add_blank = add_blank
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max
        self.load_durations = load_durations
        self.data_parameters = data_parameters if data_parameters is not None else {"mel_mean": 0, "mel_std": 1}
        self.speaker_id_to_index = build_speaker_id_map(self.records) if n_spks > 1 else {}

        random.seed(seed)
        random.shuffle(self.records)

    def get_datapoint(self, record):
        filepath = resolve_record_wav_path(record, self.filelist_path.parent)
        text, cleaned_text, raw_phones = self.get_text(record)
        mel = self.get_mel(filepath)
        spk = self.get_speaker(record)
        durations = self.get_durations(filepath, text) if self.load_durations else None

        return {
            "x": text,
            "y": mel,
            "spk": spk,
            "filepath": str(filepath),
            "x_text": cleaned_text,
            "raw_phones": raw_phones,
            "utt_id": record.get("utt_id"),
            "text": record.get("text") or record.get("sentence"),
            "word": record.get("word"),
            "variant": record.get("variant"),
            "surface_ipa": record.get("surface_ipa"),
            "canonical_ipa": record.get("canonical_ipa"),
            "ssml": record.get("ssml"),
            "canonical_phones": record.get("canonical_phones") or record.get("canonical_ipa"),
            "realized_phones": record.get("realized_phones")
            or record.get("surface_ipa")
            or record.get(self.phone_field),
            "error_ops": record.get("error_ops"),
            "durations": durations,
        }

    def get_speaker(self, record):
        if self.n_spks <= 1:
            return None
        if "speaker_id" not in record:
            raise KeyError("PED JSONL records require speaker_id when n_spks > 1")
        return self.speaker_id_to_index[str(record["speaker_id"])]

    def get_text(self, record):
        if self.phone_field not in record:
            raise KeyError(f"PED manifest record is missing required phone field: {self.phone_field}")

        if self.phone_format == "ssml_ipa":
            token_ids, cleaned_text, raw_phones = ssml_to_ids(
                record[self.phone_field], text_cleaner=clean_text_segment, add_blank=self.add_blank
            )
        else:
            token_ids, cleaned_text, raw_phones = phones_to_ids(
                record[self.phone_field], phone_format=self.phone_format, add_blank=self.add_blank
            )
        return torch.IntTensor(token_ids), cleaned_text, raw_phones

    def get_durations(self, filepath, text):
        filepath = Path(filepath)
        data_dir, name = filepath.parent.parent, filepath.stem

        try:
            dur_loc = data_dir / "durations" / f"{name}.npy"
            durs = torch.from_numpy(np.load(dur_loc).astype(int))
        except FileNotFoundError as e:
            raise FileNotFoundError(
                "Tried loading PED durations but no duration file existed at "
                f"{dur_loc}. Generate durations first or set load_durations=false."
            ) from e

        assert len(durs) == len(text), f"Length of durations {len(durs)} and text {len(text)} do not match"
        return durs

    def get_mel(self, filepath):
        audio, sr = ta.load(filepath)
        assert sr == self.sample_rate, f"Expected sample rate {self.sample_rate}, got {sr} for {filepath}"
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        mel = mel_spectrogram(
            audio,
            self.n_fft,
            self.n_mels,
            self.sample_rate,
            self.hop_length,
            self.win_length,
            self.f_min,
            self.f_max,
            center=False,
        ).squeeze()
        mel = normalize(mel, self.data_parameters["mel_mean"], self.data_parameters["mel_std"])
        return mel

    def __getitem__(self, index):
        return self.get_datapoint(self.records[index])

    def __len__(self):
        return len(self.records)


class PEDTextMelBatchCollate:
    """Pad PED realized-phone examples into the batch format MatchaTTS expects."""

    def __init__(self, n_spks):
        self.n_spks = n_spks

    def __call__(self, batch):
        batch_size = len(batch)
        y_max_length = max(item["y"].shape[-1] for item in batch)
        y_max_length = fix_len_compatibility(y_max_length)
        x_max_length = max(item["x"].shape[-1] for item in batch)
        n_feats = batch[0]["y"].shape[-2]

        y = torch.zeros((batch_size, n_feats, y_max_length), dtype=torch.float32)
        x = torch.zeros((batch_size, x_max_length), dtype=torch.long)
        durations = torch.zeros((batch_size, x_max_length), dtype=torch.long)

        y_lengths, x_lengths = [], []
        spks = []
        filepaths, x_texts, raw_phones = [], [], []
        texts, canonical_phones, realized_phones, error_ops = [], [], [], []
        utt_ids, words, variants, surface_ipa, canonical_ipa, ssml = [], [], [], [], [], []
        for i, item in enumerate(batch):
            y_, x_ = item["y"], item["x"]
            y_lengths.append(y_.shape[-1])
            x_lengths.append(x_.shape[-1])
            y[i, :, : y_.shape[-1]] = y_
            x[i, : x_.shape[-1]] = x_
            spks.append(item["spk"])
            filepaths.append(item["filepath"])
            x_texts.append(item["x_text"])
            raw_phones.append(item["raw_phones"])
            texts.append(item["text"])
            canonical_phones.append(item["canonical_phones"])
            realized_phones.append(item["realized_phones"])
            error_ops.append(item["error_ops"])
            utt_ids.append(item["utt_id"])
            words.append(item["word"])
            variants.append(item["variant"])
            surface_ipa.append(item["surface_ipa"])
            canonical_ipa.append(item["canonical_ipa"])
            ssml.append(item["ssml"])
            if item["durations"] is not None:
                durations[i, : item["durations"].shape[-1]] = item["durations"]

        y_lengths = torch.tensor(y_lengths, dtype=torch.long)
        x_lengths = torch.tensor(x_lengths, dtype=torch.long)
        spks = torch.tensor(spks, dtype=torch.long) if self.n_spks > 1 else None

        return {
            "x": x,
            "x_lengths": x_lengths,
            "y": y,
            "y_lengths": y_lengths,
            "spks": spks,
            "filepaths": filepaths,
            "x_texts": x_texts,
            "raw_phones": raw_phones,
            "texts": texts,
            "canonical_phones": canonical_phones,
            "realized_phones": realized_phones,
            "error_ops": error_ops,
            "utt_ids": utt_ids,
            "words": words,
            "variants": variants,
            "surface_ipa": surface_ipa,
            "canonical_ipa": canonical_ipa,
            "ssml": ssml,
            "durations": durations if not torch.eq(durations, 0).all() else None,
        }


def parse_jsonl_records(filelist_path: Path) -> list[dict[str, Any]]:
    """Read PED manifest records from JSONL or a JSON array, skipping blank lines."""
    path = Path(filelist_path)
    manifest_text = path.read_text(encoding="utf-8")
    if manifest_text.lstrip().startswith("["):
        records = json.loads(manifest_text)
        if not isinstance(records, list):
            raise TypeError(f"PED manifest {path} must contain a JSON array or JSONL records")
    else:
        records = []
        for line_number, line in enumerate(manifest_text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL record on line {line_number} of {path}: {exc}") from exc

    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise TypeError(f"PED manifest record {index} in {path} must be an object")
        if "wav" not in record:
            raise KeyError(f"PED manifest record {index} in {path} is missing required field: wav")
    return records


def build_speaker_id_map(records: list[dict[str, Any]]) -> dict[str, int]:
    """Build a deterministic string speaker-id to integer-index map."""
    speaker_ids = sorted({str(record["speaker_id"]) for record in records if "speaker_id" in record})
    return {speaker_id: index for index, speaker_id in enumerate(speaker_ids)}


def resolve_record_wav_path(record: dict[str, Any], jsonl_dir: Path) -> Path:
    """Resolve a record wav path relative to its JSONL file when needed."""
    filepath = Path(record["wav"])
    return filepath if filepath.is_absolute() else jsonl_dir / filepath
