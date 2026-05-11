from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from common import resolve_items, write_jsonl, write_summary


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run NVIDIA Streaming Sortformer on SADA audio.")
    ap.add_argument("--input-dir", help="Folder containing SADA audio files.")
    ap.add_argument("--manifest", help="JSONL manifest with audio_filepath and optional session_name.")
    ap.add_argument("--output-dir", required=True, help="Output directory for RTTM files and summaries.")
    ap.add_argument(
        "--model-name",
        default="nvidia/diar_streaming_sortformer_4spk-v2.1",
        help="NeMo/Hugging Face model name or local checkpoint path.",
    )
    ap.add_argument(
        "--mode",
        choices=["streaming", "non-streaming"],
        default="streaming",
        help="Use the documented streaming settings or an offline-ish long-chunk preset.",
    )
    ap.add_argument("--nemo-root", help="Path to a local NVIDIA NeMo clone. Can also be set through NEMO_ROOT.")
    ap.add_argument("--python-bin", default=sys.executable, help="Python executable used to run the NeMo example script.")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--postprocessing-yaml", help="Optional NeMo post-processing YAML file.")
    ap.add_argument("--spkcache-len", type=int, default=188)
    ap.add_argument("--spkcache-update-period", type=int)
    ap.add_argument("--fifo-len", type=int)
    ap.add_argument("--chunk-len", type=int)
    ap.add_argument("--chunk-right-context", type=int)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the manifest and command only, without running NeMo inference.",
    )
    return ap.parse_args()


def defaults_for_mode(mode: str) -> dict:
    if mode == "streaming":
        return {
            "spkcache_update_period": 300,
            "fifo_len": 40,
            "chunk_len": 340,
            "chunk_right_context": 40,
        }
    return {
        "spkcache_update_period": 2000,
        "fifo_len": 2000,
        "chunk_len": 2000,
        "chunk_right_context": 0,
    }


def find_rttm_outputs(search_root: Path) -> dict[str, Path]:
    found = {}
    for path in search_root.rglob("*.rttm"):
        found[path.stem] = path
    return found


def main() -> int:
    args = parse_args()
    nemo_root = Path(args.nemo_root or "").expanduser().resolve() if args.nemo_root else None
    if not nemo_root:
        env_root = Path(Path.cwd().anchor) if False else None
        import os
        root_env = os.environ.get("NEMO_ROOT")
        if root_env:
            nemo_root = Path(root_env).expanduser().resolve()
    if not nemo_root:
        raise SystemExit("Provide --nemo-root or set NEMO_ROOT to a local NVIDIA NeMo clone.")

    script_path = nemo_root / "examples" / "speaker_tasks" / "diarization" / "neural_diarizer" / "e2e_diarize_speech.py"
    if not script_path.exists():
        raise SystemExit(f"Could not find NeMo example script: {script_path}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = output_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    items = resolve_items(args.input_dir, args.manifest)
    manifest_rows = [
        {
            "audio_filepath": str(item["audio_path"]),
            "offset": 0,
            "duration": None,
            "label": "infer",
            "text": "-",
            "num_speakers": None,
            "rttm_filepath": None,
            "uem_filepath": None,
        }
        for item in items
    ]
    manifest_path = manifest_dir / "sada_sortformer_manifest.jsonl"
    write_jsonl(manifest_rows, manifest_path)

    preset = defaults_for_mode(args.mode)
    spkcache_update_period = args.spkcache_update_period or preset["spkcache_update_period"]
    fifo_len = args.fifo_len or preset["fifo_len"]
    chunk_len = args.chunk_len or preset["chunk_len"]
    chunk_right_context = args.chunk_right_context or preset["chunk_right_context"]

    cmd = [
        args.python_bin,
        str(script_path),
        f"model_path={args.model_name}",
        f"dataset_manifest={manifest_path}",
        f"batch_size={args.batch_size}",
        f"spkcache_len={args.spkcache_len}",
        f"spkcache_update_period={spkcache_update_period}",
        f"fifo_len={fifo_len}",
        f"chunk_len={chunk_len}",
        f"chunk_right_context={chunk_right_context}",
    ]
    if args.postprocessing_yaml:
        cmd.append(f"postprocessing_yaml={args.postprocessing_yaml}")

    run_log = log_dir / "sortformer_command.txt"
    run_log.write_text(" ".join(cmd))

    if args.dry_run:
        dry = {
            "mode": args.mode,
            "manifest_path": str(manifest_path),
            "command": cmd,
            "items": len(items),
        }
        (output_dir / "dry_run.json").write_text(json.dumps(dry, indent=2))
        write_summary([], [], output_dir)
        return 0

    proc = subprocess.run(cmd, cwd=str(output_dir), capture_output=True, text=True)
    (log_dir / "stdout.log").write_text(proc.stdout)
    (log_dir / "stderr.log").write_text(proc.stderr)

    if proc.returncode != 0:
        raise SystemExit(
            "Sortformer inference command failed. Check logs in "
            f"{log_dir} for details."
        )

    found = find_rttm_outputs(output_dir)
    successes = []
    failures = []
    for item in items:
        session_name = item["session_name"]
        audio_path = str(item["audio_path"])
        rttm_path = found.get(session_name)
        if rttm_path:
            successes.append(
                {
                    "session_name": session_name,
                    "audio_path": audio_path,
                    "rttm_path": str(rttm_path),
                }
            )
        else:
            failures.append(
                {
                    "session_name": session_name,
                    "audio_path": audio_path,
                    "error": "No RTTM file found after NeMo run.",
                }
            )

    write_summary(successes, failures, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
