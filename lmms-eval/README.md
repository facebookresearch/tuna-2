# Tuna Multimodal Understanding Evaluation

Evaluation harness for Tuna models on multimodal understanding benchmarks, built on [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval). Thanks for their awesome work!

## Installation

```bash
cd lmms-eval
pip install -e .
```

## Usage

### Tuna-2 (7B, no encoder)

```bash
CKPT_PATH=/path/to/tuna_2_pixel_7b.pt \
GPU=0 bash run_eval_none_encoder_7b.sh
```

### Tuna-R (7B, SigLIP pixel)

```bash
CKPT_PATH=/path/to/tuna_2r_pixel_7b.pt \
GPU=0 bash run_eval_siglip_pixel_7b.sh
```

### Tuna (2B, VAE)

```bash
CKPT_PATH=/path/to/tuna_2b.pt \
GPU=0 bash run_eval_vae_2b.sh
```

### Options

All scripts accept environment variables:

| Variable | Default | Description |
|---|---|---|
| `CKPT_PATH` | *(required)* | Path to the model checkpoint |
| `GPU` | `0` | GPU device index |
| `NUM_GPUS` | `1` | Number of GPUs for distributed eval |
| `TASKS` | *(see script)* | Comma-separated benchmark list |
| `OUTPUT_DIR` | `./outputs/eval/<variant>` | Results output directory |

### Available Benchmarks

`ai2d`, `gqa`, `ocrbench`, `vstar_bench`, `realworldqa`, `chartqa`, `mmvet`, `seedbench_2_plus`, `countbench`, `mmvp`, `visulogic`, `mmmu_val`

Example — run only on a subset:

```bash
CKPT_PATH=/path/to/ckpt.pt \
TASKS=gqa,ocrbench \
bash run_eval_none_encoder_7b.sh
```

## Model Configs

| Config | Variant | Size |
|---|---|---|
| `lmms_eval/models/configs/tuna_none_encoder_7b.yaml` | Tuna-2 | 7B |
| `lmms_eval/models/configs/tuna_siglip_pixel_7b.yaml` | Tuna-R | 7B |
| `lmms_eval/models/configs/tuna_vae_2b.yaml` | Tuna | 2B |

## Results

Results are saved to `OUTPUT_DIR` as JSON files with per-sample logs. Use `--log_samples` (enabled by default) to save detailed per-question outputs.
