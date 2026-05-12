# SADA diarization run scripts

This repo contains run scripts only. It does not include SADA audio, manifests, or diarization results.

## Files

- `scripts/run_sortformer.sh`
- `scripts/run_sortformer_sada.py`
- `scripts/run_diarizen.sh`
- `scripts/run_diarizen_sada.py`
- `scripts/validate_setup.py`
- `data/sada_manifest.example.jsonl`

## Requirements

- Python 3.10+
- `torch`
- `soundfile`
- helper packages from `requirements.txt`

Install the helper packages:

```bash
pip install -r requirements.txt
```

If a model needs Hugging Face access:

```bash
export HF_TOKEN=...
```

or:

```bash
huggingface-cli login
```

GPU is recommended for practical runtime.

## Input

Use either:

- a folder that contains SADA audio files
- or a JSONL manifest in the format shown in `data/sada_manifest.example.jsonl`

Outputs are written under the output directory you pass at run time.

## Sortformer

Model:

- `nvidia/diar_streaming_sortformer_4spk-v2.1`

You need a local NeMo clone:

```bash
pip install "nemo-toolkit[all]"
git clone https://github.com/NVIDIA-NeMo/NeMo.git
export NEMO_ROOT=/path/to/NeMo
```

Folder mode:

```bash
zsh scripts/run_sortformer.sh <input_dir> <output_dir> <nemo_root> streaming
```

Manifest mode:

```bash
zsh scripts/run_sortformer.sh <manifest.jsonl> <output_dir> <nemo_root> streaming python manifest
```

For the offline-style preset, replace `streaming` with `non-streaming`.

## DiariZen

Model:

- `BUT-FIT/diarizen-wavlm-large-s80-md-v2`

Recommended install flow:

```bash
git clone https://github.com/BUTSpeechFIT/DiariZen.git
cd DiariZen
pip install -r requirements.txt
pip install -e .
```

Folder mode:

```bash
zsh scripts/run_diarizen.sh <input_dir> <output_dir>
```

Manifest mode:

```bash
zsh scripts/run_diarizen.sh <manifest.jsonl> <output_dir> python manifest
```

## Outputs

Each run writes:

- `rttm_outputs/*.rttm`
- `summary.json`
- `summary.csv`
- `logs/`

These are raw diarization outputs.

## Validation

Quick structural check:

```bash
python scripts/validate_setup.py
```

Optional stronger check with one real audio file:

```bash
python scripts/validate_setup.py \
  --sample-audio /path/to/sample.wav \
  --nemo-root /path/to/NeMo
```

This writes `validation_report.json`.

## Notes

- `run_sortformer_sada.py` wraps the official NeMo example inference script.
- `run_diarizen_sada.py` expects `DiariZenPipeline.from_pretrained(...)` to be available.
- Check each model card for license restrictions before use.
