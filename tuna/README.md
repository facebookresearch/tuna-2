# Training Guide

This guide covers fine-tuning Tuna models from the released foundation checkpoint.

## Quick Start

```bash
# Single GPU
python -m tuna.scripts.train \
    model.load_stage1_model=/path/to/foundation_ckpt.pt

# Multi-GPU (FSDP)
torchrun --standalone --nproc-per-node=8 \
    -m tuna.scripts.train \
    model.load_stage1_model=/path/to/foundation_ckpt.pt
```

## Foundation Checkpoint

The released foundation checkpoint has a small number of layers randomly re-initialized in both the LLM backbone and the diffusion head. All other components (vision encoder, projections, embeddings, etc.) are fully preserved. The checkpoint loads exactly like a standard checkpoint — no key remapping or filtering needed. With a short fine-tuning pass, the re-initialized layers can be quickly re-learned and the model restored to full quality.

## Data Format

### Text-to-Image (T2I)

```jsonl
{"image": "imgs/image1.png", "caption": "a woman with blonde-highlighted bob hair taking a selfie, wearing a pink lace sleeveless top"}
{"image": "imgs/image2.png", "caption": "a surreal close-up photograph of an elderly person's face with a miniature clothes iron placed under their eye"}
```

### Multimodal Understanding (MMU)

```jsonl
{"image": "imgs/image1.png", "conversations": [{"from": "human", "value": "Describe this image."}, {"from": "gpt", "value": "A woman with blonde-highlighted bob hair is taking a selfie. She is wearing a pink lace sleeveless top and standing in front of a wooden paneled wall."}]}
```

### Text-Only

```jsonl
{"conversations": [{"from": "human", "value": "What is the capital of France?"}, {"from": "gpt", "value": "The capital of France is Paris, known for the Eiffel Tower, Louvre Museum, and its rich cultural heritage."}]}
```

### Image Editing

```jsonl
{"raw_image": "imgs/edit0.png", "out_image": "imgs/edit0_target.png", "instruction": "add warm golden sunset lighting to the scene"}
```

See `data_examples/` for complete working examples of each format.

## Training Config

Config: [`configs/train/train.yaml`](../configs/train/train.yaml)

All configuration uses [Hydra](https://hydra.cc/). Any field can be overridden on the CLI.

### Data Streams

Four data streams mixed by weighted sampling:

| Stream | Weight | Data Type | max_text_length | batch_size | Description |
|---|---|---|---|---|---|
| `t2i` | 50% | `t2i` | 2048 | 2 | Text-to-image generation |
| `edit` | 20% | `edit_interleaved` | 4096 | 3 | Image editing |
| `mmu` | 20% | `mmu` | 4096 | 3 | Multimodal understanding |
| `text` | 10% | `mmu_text` | 4096 | 3 | Text-only conversations |

Each stream can have its own `batch_size` and `num_workers`. If omitted, the global `training.batch_size` / `training.num_workers` are used as fallback.

```bash
# Override per-stream batch size
torchrun ... -m tuna.scripts.train \
    data.streams.t2i.batch_size=4 \
    data.streams.mmu.batch_size=2
```

Multi-resolution training is enabled by default for image streams (`multi_resolution: true`). Each sample is resized to its closest aspect-ratio bucket:

| Resolution | Tokens |
|---|---|
| 512 x 512 | 1024 |
| 448 x 576 | 1008 |
| 576 x 448 | 1008 |
| 384 x 672 | 1008 |
| 672 x 384 | 1008 |

### Model Variants

| Config | Variant | Size | Noise Scheduler | Description |
|---|---|---|---|---|
| `model=tuna_2_pixel_7b` | Tuna-2 | 7B | JiT | No encoder, Conv2d patchify |
| `model=tuna_2r_pixel_7b` | Tuna-R | 7B | JiT | SigLIP pixel, no VAE |
| `model=tuna_7b` | Tuna | 7B | Flow matching | SigLIP + WAN 2.2 VAE |
| `model=tuna_2b` | Tuna | 2B | Flow matching | SigLIP + WAN 2.2 VAE |

```bash
# Train Tuna-R instead of Tuna-2
torchrun ... -m tuna.scripts.train \
    model=tuna_2r_pixel_7b \
    model.load_stage1_model=/path/to/tuna_r_ckpt.pt
```

### Model Parameters

These live in `configs/model/` and can be overridden on the CLI via `model.<param>=<value>`.

| Parameter | Default (Tuna-2 7B) | Description |
|---|---|---|
| `load_stage1_model` | null | Path to checkpoint to load |
| `frozen_params` | `['vae']` | List of parameter name patterns to freeze |
| `ntp_coeff` | 1.0 | Next-token prediction loss coefficient |
| `flow_coeff` | 1.0 | Flow matching / diffusion loss coefficient |
| `und_max_t0` | 1.0 | Max timestep for understanding (MMU) tasks |
| `mmu_noise_prob` | 0.1 | Probability of adding noise to MMU inputs |
| `mmu_noise_level` | 0.1 | Maximum noise level for MMU inputs |
| `noise_scale` | 2.0 | JiT noise scale (pixel variants only) |
| `enable_mask_token` | true | Enable DeTok-style masked image modelling (Tuna-2 only) |
| `masked_image_ratio` | 0.3 | Maximum masking ratio |
| `masked_image_ratio_min` | -0.7 | Minimum masking ratio (negative = chance of no masking) |
| `gradient_checkpointing` | true | Enable activation checkpointing to reduce GPU memory |
| `attention_backend` | sdpa | Attention implementation: `sdpa` or `flexattention` |

```bash
# Example: freeze the LLM backbone, only train diffusion head
torchrun ... -m tuna.scripts.train \
    model.frozen_params="['tuna','vision_model']"

# Example: disable masked image modelling
torchrun ... -m tuna.scripts.train \
    model.enable_mask_token=false
```

### Training Parameters

These live under `training:` in the yaml and are overridden via `training.<param>=<value>`.

| Parameter | Default | Description |
|---|---|---|
| `learning_rate` | 2.0e-5 | AdamW learning rate |
| `weight_decay` | 1.0e-2 | AdamW weight decay |
| `warmup_steps` | 1000 | Linear warmup steps, then constant LR |
| `batch_size` | 4 | Global fallback batch size (per-stream overrides take priority) |
| `num_workers` | 8 | Global fallback dataloader workers |
| `gradient_accumulation_steps` | 1 | Gradient accumulation steps |
| `mixed_precision` | bf16 | Mixed precision: `bf16`, `fp16`, or null |
| `ema_decay` | 0.9999 | EMA decay rate (null to disable) |
| `save_every` | 500 | Save checkpoint every N steps |
| `keep_last` | 3 | Keep only the last N checkpoints |
| `gc_interval` | 5001 | Garbage collection interval (steps) |
| `max_epochs` | 1 | Maximum training epochs |
| `max_steps_per_epoch` | 1000000 | Maximum steps per epoch |
| `output_dir` | ./outputs/train | Output directory for checkpoints, logs, tensorboard |

Training auto-resumes from the latest checkpoint in `{output_dir}/checkpoints/` if one exists.

### FSDP

Controlled via `training.fsdp.*`:

| Parameter | Default | Description |
|---|---|---|
| `enable` | true | Enable FSDP |
| `sharding_strategy` | SHARD_GRAD_OP | `SHARD_GRAD_OP`, `FULL_SHARD`, or `NO_SHARD` |
| `state_dict_type` | SHARDED_STATE_DICT | State dict format for checkpointing |
| `mixed_precision.param_dtype` | bf16 | Parameter dtype |
| `mixed_precision.reduce_dtype` | bf16 | Gradient reduction dtype |

```bash
# Use FULL_SHARD for tighter memory
torchrun ... -m tuna.scripts.train \
    training.fsdp.sharding_strategy=FULL_SHARD
```

## Examples

```bash
# Smoke test (4 steps)
python -m tuna.scripts.train \
    training.max_steps_per_epoch=4 \
    training.batch_size=1

# Full training with custom data
torchrun --standalone --nproc-per-node=8 \
    -m tuna.scripts.train \
    model.load_stage1_model=/path/to/foundation_ckpt.pt \
    data.streams.t2i.jsonl_path=/data/my_t2i.jsonl \
    data.streams.t2i.image_root=/data/images \
    data.streams.mmu.jsonl_path=/data/my_mmu.jsonl \
    data.streams.mmu.image_root=/data/images \
    data.streams.edit.jsonl_path=/data/my_edit.jsonl \
    data.streams.edit.image_root=/data/images \
    data.streams.text.jsonl_path=/data/my_text.jsonl \
    training.output_dir=./outputs/my_run

# Adjust learning rate and disable EMA
torchrun ... -m tuna.scripts.train \
    training.learning_rate=1e-5 \
    training.ema_decay=null

# Set random seed
python -m tuna.scripts.train seed=42
```

