from __future__ import annotations

import argparse
import traceback
from pathlib import Path

from common import resolve_items, write_summary


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run DiariZen on SADA audio and save RTTM outputs.")
    ap.add_argument("--input-dir", help="Folder containing SADA audio files.")
    ap.add_argument("--manifest", help="JSONL manifest with audio_filepath and optional session_name.")
    ap.add_argument("--output-dir", required=True, help="Output directory for RTTM files and summaries.")
    ap.add_argument(
        "--model-name",
        default="BUT-FIT/diarizen-wavlm-large-s80-md-v2",
        help="DiariZen model name or local path.",
    )
    ap.add_argument("--device", default="auto", help="Kept for logging; actual device handling depends on DiariZen.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    rttm_dir = output_dir / "rttm_outputs"
    log_dir = output_dir / "logs"
    rttm_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    items = resolve_items(args.input_dir, args.manifest)

    try:
        from diarizen.pipelines.inference import DiariZenPipeline
    except Exception as exc:
        raise SystemExit(
            "Could not import DiariZenPipeline. Please install DiariZen first. "
            f"Original error: {exc}"
        )

    pipeline = DiariZenPipeline.from_pretrained(args.model_name, rttm_out_dir=str(rttm_dir))

    successes = []
    failures = []

    for item in items:
        audio_path = Path(item["audio_path"])
        session_name = item["session_name"]
        try:
            pipeline(str(audio_path), sess_name=session_name)
            rttm_path = rttm_dir / f"{session_name}.rttm"
            successes.append(
                {
                    "session_name": session_name,
                    "audio_path": str(audio_path),
                    "rttm_path": str(rttm_path),
                }
            )
        except Exception as exc:
            err_path = log_dir / f"{session_name}.error.txt"
            err_path.write_text(traceback.format_exc())
            failures.append(
                {
                    "session_name": session_name,
                    "audio_path": str(audio_path),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    write_summary(successes, failures, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
