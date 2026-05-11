from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, List, Dict


AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a"}


def discover_audio(input_dir: Path) -> List[Path]:
    files = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS]
    return sorted(files)


def load_manifest(manifest_path: Path) -> List[Dict]:
    rows = []
    for line in manifest_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def resolve_items(input_dir: str | None, manifest_path: str | None) -> List[Dict]:
    if not input_dir and not manifest_path:
        raise ValueError("Provide either --input-dir or --manifest.")
    if input_dir and manifest_path:
        raise ValueError("Use only one of --input-dir or --manifest.")

    if manifest_path:
        rows = load_manifest(Path(manifest_path))
        items = []
        for row in rows:
            audio = Path(row["audio_filepath"]).expanduser().resolve()
            session = row.get("session_name") or audio.stem
            items.append({"audio_path": audio, "session_name": session})
        return items

    audio_files = discover_audio(Path(input_dir).expanduser().resolve())
    return [{"audio_path": p, "session_name": p.stem} for p in audio_files]


def write_jsonl(rows: Iterable[Dict], path: Path) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def write_summary(successes: List[Dict], failures: List[Dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "successful_files": len(successes),
        "failed_files": len(failures),
        "successes": successes,
        "failures": failures,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True))

    with (out_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["status", "session_name", "audio_path", "rttm_path", "error"])
        for row in successes:
            writer.writerow(["success", row["session_name"], row["audio_path"], row.get("rttm_path", ""), ""])
        for row in failures:
            writer.writerow(["failed", row["session_name"], row["audio_path"], "", row["error"]])
