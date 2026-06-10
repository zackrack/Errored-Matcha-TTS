#!/usr/bin/env python3
"""Stage 0.5 smoke test for a fresh Matcha-TTS checkout.

This script verifies that Matcha-TTS can be invoked and can generate one WAV file.
It prefers the installed ``matcha-tts`` console command so it exercises the package
entry point. If the console command is not available, it falls back to
``python -m matcha.cli``, which is useful when running directly from a cloned repo.

By default it uses the public LJSpeech Matcha-TTS checkpoint and HiFi-GAN vocoder.
Those files are downloaded by Matcha-TTS on first use, so the first run can take a
while and requires network access.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_TEXT = "Stage zero point five: Matcha T T S is installed and running."


def build_command(args: argparse.Namespace) -> list[str]:
    """Build the Matcha-TTS command used for the smoke test."""
    matcha_tts = shutil.which("matcha-tts")
    command = [matcha_tts] if matcha_tts else [sys.executable, "-m", "matcha.cli"]

    command.extend(
        [
            "--text",
            args.text,
            "--output_folder",
            str(args.output_dir),
            "--model",
            args.model,
            "--steps",
            str(args.steps),
            "--speaking_rate",
            str(args.speaking_rate),
            "--temperature",
            str(args.temperature),
        ]
    )

    if args.vocoder is not None:
        command.extend(["--vocoder", args.vocoder])
    if args.cpu:
        command.append("--cpu")

    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one WAV with Matcha-TTS to verify a clone/install works."
    )
    parser.add_argument(
        "--text",
        default=DEFAULT_TEXT,
        help="Text to synthesize for the smoke test.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("stage_05_outputs"),
        help="Directory where the smoke-test WAV/PNG/NumPy files will be written.",
    )
    parser.add_argument(
        "--model",
        default="matcha_ljspeech",
        choices=("matcha_ljspeech", "matcha_vctk"),
        help="Public Matcha-TTS model to use.",
    )
    parser.add_argument(
        "--vocoder",
        default=None,
        choices=("hifigan_T2_v1", "hifigan_univ_v1"),
        help="Optional vocoder override. By default Matcha-TTS chooses one for the model.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Number of ODE steps for synthesis.",
    )
    parser.add_argument(
        "--speaking_rate",
        type=float,
        default=1.0,
        help="Speaking-rate/length-scale value passed to Matcha-TTS.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.667,
        help="Sampling temperature passed to Matcha-TTS.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference. By default Matcha-TTS uses CUDA when available.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    command = build_command(args)
    print("[stage 0.5] Running Matcha-TTS smoke test:", flush=True)
    print(" ".join(str(part) for part in command), flush=True)

    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        print(f"[stage 0.5] Matcha-TTS failed with exit code {completed.returncode}.", file=sys.stderr)
        print(
            "[stage 0.5] If this is a fresh clone, install dependencies first with `python -m pip install -e .`.",
            file=sys.stderr,
        )
        print(
            "[stage 0.5] If dependencies are installed, reinstall this checkout so the console script uses "
            "the latest code. PyTorch 2.6+ requires the trusted-checkpoint loading fix in this repo.",
            file=sys.stderr,
        )
        return completed.returncode

    wavs = sorted(args.output_dir.glob("*.wav"))
    if not wavs:
        print(f"[stage 0.5] Matcha-TTS finished, but no WAV was found in {args.output_dir}.", file=sys.stderr)
        return 1

    newest_wav = max(wavs, key=lambda path: path.stat().st_mtime)
    print(f"[stage 0.5] Success. Generated audio: {newest_wav.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
