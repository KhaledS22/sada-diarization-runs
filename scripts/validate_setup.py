from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
DATA = ROOT / "data"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Validate the SADA diarization run package without running the models.")
    ap.add_argument("--sample-audio", help="Optional path to one real audio file for a stronger dry-run check.")
    ap.add_argument("--nemo-root", help="Optional local NVIDIA NeMo clone path for Sortformer dry-run validation.")
    ap.add_argument("--python-bin", default=sys.executable)
    return ap.parse_args()


def check_import(module_name: str) -> str:
    return "available" if importlib.util.find_spec(module_name) else "missing"


def run(cmd: list[str]) -> dict:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return {
        "command": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def main() -> int:
    args = parse_args()
    report = {
        "package_root": str(ROOT),
        "files_present": {},
        "imports": {},
        "dry_runs": [],
    }

    expected_files = [
        ROOT / "README.md",
        ROOT / "requirements.txt",
        DATA / "sada_manifest.example.jsonl",
        SCRIPTS / "common.py",
        SCRIPTS / "run_diarizen_sada.py",
        SCRIPTS / "run_sortformer_sada.py",
    ]
    for path in expected_files:
        report["files_present"][str(path.relative_to(ROOT))] = path.exists()

    for module_name in ["soundfile", "torch", "diarizen", "nemo"]:
        report["imports"][module_name] = check_import(module_name)

    py_compile_cmd = [
        args.python_bin,
        "-m",
        "py_compile",
        str(SCRIPTS / "common.py"),
        str(SCRIPTS / "run_diarizen_sada.py"),
        str(SCRIPTS / "run_sortformer_sada.py"),
    ]
    report["py_compile"] = run(py_compile_cmd)

    if args.sample_audio:
        sample_audio = str(Path(args.sample_audio).expanduser().resolve())
        diarizen_cmd = [
            args.python_bin,
            str(SCRIPTS / "run_diarizen_sada.py"),
            "--manifest",
            str(DATA / "sada_manifest.example.jsonl"),
            "--output-dir",
            str(ROOT / "outputs" / "validate_diarizen"),
        ]
        report["dry_runs"].append(
            {
                "name": "diarizen_cli_help",
                **run([args.python_bin, str(SCRIPTS / "run_diarizen_sada.py"), "--help"]),
            }
        )

        tmp_manifest = ROOT / "data" / "_validate_manifest.jsonl"
        tmp_manifest.write_text(json.dumps({"audio_filepath": sample_audio, "session_name": Path(sample_audio).stem}) + "\n")
        try:
            report["dry_runs"].append(
                {
                    "name": "diarizen_import_path_check",
                    **run(
                        [
                            args.python_bin,
                            str(SCRIPTS / "run_diarizen_sada.py"),
                            "--manifest",
                            str(tmp_manifest),
                            "--output-dir",
                            str(ROOT / "outputs" / "validate_diarizen"),
                        ]
                    ),
                }
            )
        finally:
            tmp_manifest.unlink(missing_ok=True)

        if args.nemo_root:
            tmp_manifest = ROOT / "data" / "_validate_manifest_sortformer.jsonl"
            tmp_manifest.write_text(json.dumps({"audio_filepath": sample_audio, "session_name": Path(sample_audio).stem}) + "\n")
            try:
                report["dry_runs"].append(
                    {
                        "name": "sortformer_dry_run",
                        **run(
                            [
                                args.python_bin,
                                str(SCRIPTS / "run_sortformer_sada.py"),
                                "--manifest",
                                str(tmp_manifest),
                                "--output-dir",
                                str(ROOT / "outputs" / "validate_sortformer"),
                                "--nemo-root",
                                str(Path(args.nemo_root).expanduser().resolve()),
                                "--dry-run",
                            ]
                        ),
                    }
                )
            finally:
                tmp_manifest.unlink(missing_ok=True)
        else:
            report["dry_runs"].append(
                {
                    "name": "sortformer_cli_help",
                    **run([args.python_bin, str(SCRIPTS / "run_sortformer_sada.py"), "--help"]),
                }
            )

    out = ROOT / "validation_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
