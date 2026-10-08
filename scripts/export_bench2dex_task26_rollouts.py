"""Export one H.264 overhead MP4 per Bench2Dex evaluation episode."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile


CHANNELS = ("none", "cov", "inv", "inv_cov")


def _read_results(eval_root: Path) -> list[dict]:
    rows: list[dict] = []
    for channel in CHANNELS:
        path = eval_root / channel / "per_episode.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    row["channel"] = channel
                    rows.append(row)
    return rows


def _find_record(eval_root: Path, channel: str, episode: int, success: bool) -> Path:
    outcome = "success" if success else "failure"
    candidates = list(
        (eval_root / channel).glob(f"**/records/{outcome}/episode_{episode:06d}*.hdf5")
    )
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one recording for {channel} episode {episode}, found {len(candidates)}"
        )
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bench2dex-root", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_results(args.eval_root)
    if len(rows) != 100:
        raise RuntimeError(f"Expected exactly 100 rollout results, found {len(rows)}")

    manifest: list[dict] = []
    with tempfile.TemporaryDirectory(dir=args.output_dir) as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        for rollout_index, row in enumerate(rows, start=1):
            channel = row["channel"]
            episode = int(row["episode_index"])
            success = bool(row["success"])
            outcome = "success" if success else "failure"
            source = _find_record(args.eval_root, channel, episode, success)
            stem = f"rollout_{rollout_index:03d}_{channel}_episode_{episode:03d}_{outcome}"
            intermediate = tmp_dir_path / f"{stem}_mp4v.mp4"
            output = args.output_dir / f"{stem}.mp4"

            subprocess.run(
                [
                    sys.executable,
                    str(args.bench2dex_root / "tools/export/export_video.py"),
                    str(source),
                    "--output",
                    str(intermediate),
                    "--fps",
                    str(args.fps),
                    "--cell-width",
                    "640",
                    "--cell-height",
                    "480",
                ],
                check=True,
            )
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    str(intermediate),
                    "-c:v",
                    "libx264",
                    "-preset",
                    "medium",
                    "-crf",
                    "23",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(output),
                ],
                check=True,
            )
            manifest.append(
                {
                    "rollout": rollout_index,
                    "channel": channel,
                    "episode": episode,
                    "success": success,
                    "video": output.name,
                    "source_hdf5": str(source),
                }
            )

    with (args.output_dir / "rollout_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest[0].keys())
        writer.writeheader()
        writer.writerows(manifest)
    (args.output_dir / "rollout_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    by_channel = {}
    for channel in CHANNELS:
        channel_rows = [row for row in manifest if row["channel"] == channel]
        successes = sum(bool(row["success"]) for row in channel_rows)
        by_channel[channel] = {
            "episodes": len(channel_rows),
            "successes": successes,
            "success_rate": successes / len(channel_rows) if channel_rows else None,
        }
    total_successes = sum(bool(row["success"]) for row in manifest)
    summary = {
        "episodes": len(manifest),
        "successes": total_successes,
        "success_rate": total_successes / len(manifest),
        "by_channel": by_channel,
    }
    (args.output_dir / "success_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
