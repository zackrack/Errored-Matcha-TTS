from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from matcha.utils.checkpoints import load_lightning_checkpoint_trusted, torch_load_trusted


class LightningLikeWithWeightsOnly:
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, map_location=None, weights_only=None):
        return {
            "checkpoint_path": checkpoint_path,
            "map_location": map_location,
            "weights_only": weights_only,
        }


class LightningLikeWithoutWeightsOnly:
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, map_location=None):
        return {
            "checkpoint_path": checkpoint_path,
            "map_location": map_location,
        }


def test_load_lightning_checkpoint_trusted_sets_weights_only_false_when_supported():
    loaded = load_lightning_checkpoint_trusted(
        LightningLikeWithWeightsOnly,
        "model.ckpt",
        map_location="cpu",
    )

    assert loaded == {
        "checkpoint_path": "model.ckpt",
        "map_location": "cpu",
        "weights_only": False,
    }


def test_load_lightning_checkpoint_trusted_supports_older_lightning_signature():
    loaded = load_lightning_checkpoint_trusted(
        LightningLikeWithoutWeightsOnly,
        "model.ckpt",
        map_location="cpu",
    )

    assert loaded == {
        "checkpoint_path": "model.ckpt",
        "map_location": "cpu",
    }


def test_torch_load_trusted_loads_checkpoint(tmp_path: Path):
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"ok": True}, checkpoint_path)

    assert torch_load_trusted(checkpoint_path, map_location="cpu") == {"ok": True}
