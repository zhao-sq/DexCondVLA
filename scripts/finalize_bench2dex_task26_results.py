"""Validate Task26 videos and write final statistics into the evaluation doc."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import subprocess


RESULTS_START = "<!-- RESULTS_START -->"
RESULTS_END = "<!-- RESULTS_END -->"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--documentation", type=Path, required=True)
    parser.add_argument("--completion-marker", type=Path, required=True)
    args = parser.parse_args()

    summary_path = args.rollout_dir / "success_summary.json"
    manifest_path = args.rollout_dir / "rollout_manifest.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if summary.get("episodes") != 100 or len(manifest) != 100:
        raise RuntimeError(
            f"Expected 100 episodes/videos, got summary={summary.get('episodes')} manifest={len(manifest)}"
        )

    for row in manifest:
        video = args.rollout_dir / row["video"]
        if not video.is_file() or video.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty video: {video}")
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,pix_fmt,nb_frames",
                "-of",
                "json",
                str(video),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        streams = json.loads(probe.stdout).get("streams", [])
        if len(streams) != 1:
            raise RuntimeError(f"Expected one video stream in {video}")
        stream = streams[0]
        if stream.get("codec_name") != "h264" or stream.get("pix_fmt") != "yuv420p":
            raise RuntimeError(f"Unexpected video encoding for {video}: {stream}")
        if int(stream.get("nb_frames", 0)) <= 0:
            raise RuntimeError(f"Video has no frames: {video}")

    completed_at = datetime.now().astimezone().isoformat(timespec="seconds")
    lines = [
        f"- Completed at: `{completed_at}`",
        f"- Episodes: **{summary['episodes']}**",
        f"- Successes: **{summary['successes']}**",
        f"- Overall success rate: **{summary['success_rate']:.1%}**",
        f"- Validated H.264 overhead videos: **{len(manifest)}**",
        f"- Video directory: `{args.rollout_dir}`",
        "",
        "| Channel | Episodes | Successes | Success rate |",
        "| --- | ---: | ---: | ---: |",
    ]
    for channel in ("none", "cov", "inv", "inv_cov"):
        row = summary["by_channel"][channel]
        lines.append(
            f"| `{channel}` | {row['episodes']} | {row['successes']} | {row['success_rate']:.1%} |"
        )
    result_text = "\n".join(lines)

    documentation = args.documentation.read_text(encoding="utf-8")
    pattern = re.compile(
        re.escape(RESULTS_START) + r".*?" + re.escape(RESULTS_END), re.DOTALL
    )
    if not pattern.search(documentation):
        raise RuntimeError(f"Results markers not found in {args.documentation}")
    args.documentation.write_text(
        pattern.sub(f"{RESULTS_START}\n{result_text}\n{RESULTS_END}", documentation),
        encoding="utf-8",
    )
    args.completion_marker.parent.mkdir(parents=True, exist_ok=True)
    args.completion_marker.write_text(
        f"Task26 evaluation and video validation completed at {completed_at}\n"
        f"successes={summary['successes']} episodes={summary['episodes']} "
        f"success_rate={summary['success_rate']:.6f}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
