# SADA diarization run scripts

Run SADA diarization with:

- `nvidia/diar_streaming_sortformer_4spk-v2.1`
- `BUT-FIT/diarizen-wavlm-large-s80-md-v2`

This repo is code-only. Audio data and generated outputs stay outside the repo.

## Files

- `scripts/run_sortformer_sada.py`
- `scripts/run_diarizen_sada.py`
- `scripts/run_sortformer.sh`
- `scripts/run_diarizen.sh`
- `scripts/validate_setup.py`
- `data/sada_manifest.example.jsonl`

## Requirements

- Python 3.10+
- `torch`
- `soundfile`

Install helper packages:

```bash
pip install -r requirements.txt
```

If a model requires Hugging Face access:

```bash
export HF_TOKEN=...
```

or:

```bash
huggingface-cli login
```

## Sortformer run

Inputs:
- `input_dir` or `manifest`
- `output_dir`
- local `NeMo` clone path

Expected input layout:

- if using `input_dir`, the script scans audio recursively
- if using `manifest`, use the JSONL format shown in `data/sada_manifest.example.jsonl`

Install NeMo and clone the repo:

```bash
pip install "nemo-toolkit[all]"
git clone https://github.com/NVIDIA-NeMo/NeMo.git
export NEMO_ROOT=/path/to/NeMo
```

Streaming command:

```bash
zsh scripts/run_sortformer.sh <input_dir> <output_dir> <nemo_root> streaming
```

Non-streaming-style command:

```bash
zsh scripts/run_sortformer.sh <input_dir> <output_dir> <nemo_root> non-streaming
```

Manifest mode:

```bash
zsh scripts/run_sortformer.sh <manifest.jsonl> <output_dir> <nemo_root> streaming python manifest
```

Outputs:
- `<output_dir>/rttm_outputs` or NeMo-produced RTTM files under the run directory
- `<output_dir>/summary.json`
- `<output_dir>/summary.csv`
- `<output_dir>/logs`

## DiariZen run

Inputs:
- `input_dir` or `manifest`
- `output_dir`

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

Outputs:
- `<output_dir>/rttm_outputs/*.rttm`
- `<output_dir>/summary.json`
- `<output_dir>/summary.csv`
- `<output_dir>/logs`

## Validation

Quick structural validation:

```bash
python scripts/validate_setup.py
```

Optional stronger validation with one real audio file:

```bash
python scripts/validate_setup.py \
  --sample-audio /path/to/sample.wav \
  --nemo-root /path/to/NeMo
```

This writes `validation_report.json`.

## Notes

- `run_sortformer_sada.py` wraps the official NeMo example inference script.
- `run_diarizen_sada.py` expects `DiariZenPipeline.from_pretrained(...)` to be available.
- Outputs are raw diarization outputs.
- Some models may require GPU for practical runtime.
- Check each model card for license restrictions. DiariZen weights are non-commercial.
