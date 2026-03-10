# autoresearch VLA

This is a VLA fork of `autoresearch`: a small repo for autonomous, fixed-budget offline research on a vision-language-action model.

The repo keeps the same core philosophy as the original project:

- `prepare.py` is the fixed harness.
- `train.py` is the file the agent mutates.
- `program.md` is the human-written research policy.

The difference is that the workload is no longer text next-token prediction. Instead, the repo assumes a drone-style offline dataset with frames, instructions, optional state, and action targets.

## How It Works

Training runs for a fixed 5-minute wall-clock budget. The agent edits `train.py`, runs a training job, reads the validation summary, and decides whether to keep or discard the change.

The fixed validation target is no longer `val_bpb`. The harness now reports an offline VLA metric stack:

- `val_action_ce`
- `val_action_top1`
- `val_waypoint_mse` when waypoints are present
- `invalid_action_rate`
- `latency_ms`
- `score`

`score` is the keep/discard metric. Higher is better.

## Project Structure

```text
prepare.py      fixed dataset contract, splits, preprocessing, action quantization, evaluation
train.py        multimodal VLA model, optimizer, training loop; this is what the agent edits
program.md      autonomous research policy
pyproject.toml  dependencies
```

## Dataset Contract

The dataset root is resolved from:

1. `$AUTORESEARCH_VLA_DATASET`
2. `~/.cache/autoresearch/vla_dataset`

The root must contain:

- `dataset.json`
- `episodes.jsonl`

`dataset.json` defines the fixed schema for:

- image size and channels
- optional state dimensions and normalization
- action dimensions and quantization bounds
- optional waypoint dimensions
- score weights

`episodes.jsonl` contains one JSON object per episode. Each row points to tensor payloads stored as `.pt`, `.pth`, or `.npy`.

Example fields:

```json
{
  "episode_id": "ep-000001",
  "instruction": "take off and hover near the red gate",
  "frame_paths": ["frames/ep-000001.pt"],
  "actions_path": "actions/ep-000001.pt",
  "states_path": "states/ep-000001.pt",
  "waypoints_path": "waypoints/ep-000001.pt",
  "split": "train"
}
```

To print the example schema from the harness:

```bash
uv run prepare.py --print-schema
```

## Quick Start

Requirements:

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)
- preferably a single NVIDIA GPU for the intended 5-minute research loop

Install dependencies:

```bash
uv sync
```

Point the repo at your dataset:

```bash
export AUTORESEARCH_VLA_DATASET=/path/to/your/vla_dataset
```

Validate the fixed harness:

```bash
uv run prepare.py
```

Run one experiment manually:

```bash
uv run train.py
```

## Steady-State Workflow

After the bootstrap conversion, the intended workflow is:

1. The human curates `program.md`.
2. The harness in `prepare.py` stays fixed.
3. The autonomous researcher edits only `train.py`.
4. Each run is compared under the same 5-minute budget using the final printed `score`.

The highest-leverage research directions for this fork are:

- action tokenization
- temporal context and frame stride
- state fusion
- multimodal fusion
- optimizer and schedule tuning
- capacity scaling under a fixed runtime budget

## Design Choices

- **Frozen evaluation harness.** The dataset contract, preprocessing, split logic, action quantization, and score definition are fixed in `prepare.py` so the agent cannot silently change the benchmark.
- **Single mutable research file.** `train.py` remains the only file the autonomous loop should edit.
- **Offline-first metric stack.** The repo intentionally starts with cheap offline proxy metrics rather than simulator rollouts on every run.
- **Tensor-backed inputs.** The first version expects `.pt` or `.npy` tensors for data payloads to avoid adding extra image-decoding dependencies.

## Notes

- This fork is currently offline-only. It does not include a simulator promotion stage yet.
- The first bootstrap rewrite touched `prepare.py`, `train.py`, and `program.md`. After that, steady-state autoresearch should go back to mutating only `train.py`.
- If your dataset is malformed or missing required manifests, `prepare.py` will fail fast instead of trying to guess.

## License

MIT
