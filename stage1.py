#!/usr/bin/env python3
"""Stage 1 PED-TTS prototype: explicit phone inference without G2P.

This script bypasses Matcha-TTS text normalization/phonemization and feeds an
explicit phone sequence directly to the existing Matcha-TTS encoder. It is a
no-training prototype: ARPAbet phones are mapped into the current IPA-symbol
vocabulary used by the pretrained Matcha checkpoints.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from matcha.utils.phone_input import (
    ARPABET_TO_IPA,
    arpabet_phone_to_ipa,
    cleaned_text_to_ids,
    phones_to_cleaned_text,
    strip_arpabet_stress,
)


def read_phone_input(args: argparse.Namespace) -> str:
    """Read phones from --phones, --phones_file, or --phone_words."""
    if args.phones is not None:
        return args.phones
    if args.phone_words is not None:
        return " | ".join(args.phone_words)
    return Path(args.phones_file).read_text(encoding="utf-8").strip()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage 1 PED-TTS: synthesize from explicit phones, bypassing Matcha-TTS G2P."
    )
    phone_source = parser.add_mutually_exclusive_group(required=True)
    phone_source.add_argument("--phones", type=str, help="Explicit phone sequence to synthesize.")
    phone_source.add_argument("--phones_file", type=str, help="File containing an explicit phone sequence.")
    phone_source.add_argument(
        "--phone_words",
        nargs="+",
        help="One quoted phone sequence per word; Stage 1 inserts word boundaries between entries.",
    )
    parser.add_argument(
        "--phone_format",
        choices=("arpabet", "ipa"),
        default="arpabet",
        help="Input phone format. ARPAbet is mapped to the existing Matcha IPA-symbol vocabulary.",
    )
    parser.add_argument("--model", default="matcha_ljspeech", choices=("matcha_ljspeech", "matcha_vctk"))
    parser.add_argument("--checkpoint_path", default=None, help="Optional custom Matcha-TTS checkpoint path.")
    parser.add_argument("--vocoder", default=None, choices=("hifigan_T2_v1", "hifigan_univ_v1"))
    parser.add_argument("--output_wav", type=Path, default=Path("stage1_outputs/error.wav"))
    parser.add_argument("--spk", type=int, default=None, help="Speaker ID for multispeaker checkpoints.")
    parser.add_argument("--steps", type=int, default=10, help="Number of ODE steps.")
    parser.add_argument("--temperature", type=float, default=0.667)
    parser.add_argument("--speaking_rate", type=float, default=1.0)
    parser.add_argument("--denoiser_strength", type=float, default=0.00025)
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference.")
    parser.add_argument("--no_blank", action="store_true", help="Do not insert Matcha-TTS blank tokens.")
    parser.add_argument("--dry_run", action="store_true", help="Print converted symbols/IDs without loading the model.")
    return parser


def resolve_model_paths(args: argparse.Namespace):
    """Download public checkpoints when needed and return Matcha/vocoder paths."""
    from matcha.cli import MATCHA_URLS, MULTISPEAKER_MODEL, SINGLESPEAKER_MODEL, VOCODER_URLS
    from matcha.utils.utils import assert_model_downloaded, get_user_data_dir

    save_dir = get_user_data_dir()
    if args.checkpoint_path is None:
        matcha_path = save_dir / f"{args.model}.ckpt"
        assert_model_downloaded(matcha_path, MATCHA_URLS[args.model])
    else:
        matcha_path = Path(args.checkpoint_path)

    model_defaults = SINGLESPEAKER_MODEL.get(args.model) or MULTISPEAKER_MODEL.get(args.model)
    if args.vocoder is None:
        args.vocoder = model_defaults["vocoder"] if model_defaults is not None else "hifigan_univ_v1"
    if args.spk is None and args.model in MULTISPEAKER_MODEL:
        args.spk = MULTISPEAKER_MODEL[args.model]["spk"]

    vocoder_path = save_dir / args.vocoder
    assert_model_downloaded(vocoder_path, VOCODER_URLS[args.vocoder])
    return matcha_path, vocoder_path


def synthesize_from_phones(args: argparse.Namespace) -> Path:
    """Run explicit-phone synthesis and write the output WAV."""
    raw_phones = read_phone_input(args)
    cleaned_text = phones_to_cleaned_text(raw_phones, args.phone_format)
    token_ids = cleaned_text_to_ids(cleaned_text, add_blank=not args.no_blank)

    print(f"[stage 1] Raw phones: {raw_phones}")
    print(f"[stage 1] Matcha IPA-symbol input: {cleaned_text}")
    print(f"[stage 1] Token IDs: {token_ids}")
    if args.dry_run:
        print("[stage 1] Dry run requested; skipping model/vocoder loading and WAV generation.")
        return args.output_wav

    import soundfile as sf
    import torch

    from matcha.cli import get_device, load_matcha, load_vocoder, to_waveform

    device = get_device(args)
    x = torch.tensor(token_ids, dtype=torch.long, device=device)[None]
    x_lengths = torch.tensor([x.shape[-1]], dtype=torch.long, device=device)
    spk = torch.tensor([args.spk], device=device, dtype=torch.long) if args.spk is not None else None

    matcha_path, vocoder_path = resolve_model_paths(args)
    model = load_matcha(args.model if args.checkpoint_path is None else "custom_model", matcha_path, device)
    vocoder, denoiser = load_vocoder(args.vocoder, vocoder_path, device)

    with torch.inference_mode():
        output = model.synthesise(
            x,
            x_lengths,
            n_timesteps=args.steps,
            temperature=args.temperature,
            spks=spk,
            length_scale=args.speaking_rate,
        )
        waveform = to_waveform(output["mel"], vocoder, denoiser, args.denoiser_strength)

    args.output_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output_wav, waveform, 22050, "PCM_24")
    print(f"[stage 1] Success. Generated explicit-phone audio: {args.output_wav.resolve()}")
    return args.output_wav


def main() -> int:
    args = build_arg_parser().parse_args()
    synthesize_from_phones(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
