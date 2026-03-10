# autoresearch VLA

This repo is for autonomous offline research on a vision-language-action model.

The repo has one important invariant:

- `prepare.py` is the fixed harness. It owns the dataset contract, episode splits, preprocessing, action quantization, and evaluation.
- `train.py` is the mutable research surface. This is the only file the autonomous researcher edits during the experiment loop.
- `program.md` is the human-authored policy for how the agent should research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date, for example `mar10-vla`. The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from the current mainline branch.
3. **Read the in-scope files**:
   - `README.md`
   - `prepare.py`
   - `train.py`
4. **Verify the fixed dataset exists**:
   - Resolve dataset root from `$AUTORESEARCH_VLA_DATASET`, or fall back to `~/.cache/autoresearch/vla_dataset`.
   - Confirm the dataset contains `dataset.json` and `episodes.jsonl`.
   - If the dataset is missing, stop and tell the human what is missing. Do not invent or silently alter the dataset.
5. **Validate the harness**: run `uv run prepare.py` to confirm the manifests parse cleanly.
6. **Initialize `results.tsv`**: create it with just the header row. The baseline will be written after the first run.
7. **Confirm and go**: once setup is sound, begin the experiment loop.

## Experimentation

Each experiment runs under a **fixed 5-minute training budget** enforced by `train.py`. Launch experiments with:

```bash
uv run train.py
```

### What you CAN do

- Modify `train.py`.
- Change model architecture, fusion strategy, temporal context, optimizer settings, model size, batch size, patch size, and loss weights.

### What you CANNOT do

- Modify `prepare.py` during the experiment loop.
- Modify the dataset manifests, dataset split logic, preprocessing rules, action quantization rules, or evaluation weights.
- Install new packages or add dependencies.
- Change the meaning of the printed summary metrics.

### The optimization target

The primary metric is:

```text
score = - action_ce - waypoint_penalty - latency_penalty - invalid_action_penalty
```

Higher `score` is better.

Secondary diagnostics:

- `val_action_ce`
- `val_action_top1`
- `val_waypoint_mse`
- `invalid_action_rate`
- `latency_ms`
- `peak_vram_mb`

Do not optimize raw action cross-entropy in isolation. A change that lowers `val_action_ce` but worsens the overall `score` is a regression and should be discarded.

### Decision rules

- Keep a change if it improves `score` meaningfully under the same 5-minute budget.
- Reject a change if it worsens `score`, increases invalid action behavior materially, or creates a large latency regression.
- Prefer simpler changes when improvements are tied or nearly tied.
- Do not keep ugly complexity for tiny gains.

### Priority order for experiments

1. Action representation and action-token decoding behavior
2. Temporal context: history length, stride, and past-action conditioning
3. Multimodal fusion of vision, language, and state
4. Optimizer and schedule tuning
5. Capacity scaling

## Output format

At the end of a successful run, `train.py` prints a summary like:

```text
---
val_action_ce:      1.234567
val_action_top1:    0.456789
val_waypoint_mse:   0.123456
invalid_action_rate:0.000000
latency_ms:         8.42
score:              -1.379113
training_seconds:   300.0
total_seconds:      322.1
peak_vram_mb:       14321.7
windows_per_second: 412.8
total_windows:      123456
num_steps:          987
num_params_M:       42.7
depth:              8
```

For quick extraction from the log:

```bash
grep "^score:\|^val_action_ce:\|^latency_ms:\|^peak_vram_mb:" run.log
```

## Logging results

Log every experiment to `results.tsv` as tab-separated values. Do not commit this file.

Use this header:

```text
commit	score	action_ce	latency_ms	memory_gb	status	description
```

Columns:

1. git commit hash, short form
2. final `score` from the run, or `0.000000` for crashes
3. `val_action_ce`, or `0.000000` for crashes
4. `latency_ms`, or `0.0` for crashes
5. peak memory in GB, rounded to one decimal place
6. status: `keep`, `discard`, or `crash`
7. a short description of the experiment

Example:

```text
commit	score	action_ce	latency_ms	memory_gb	status	description
a1b2c3d	-1.379113	1.234567	8.4	14.0	keep	baseline multimodal decoder
b2c3d4e	-1.241800	1.198300	8.7	14.4	keep	add past-action conditioning
c3d4e5f	-1.410500	1.180100	14.9	14.3	discard	larger patch projector hurts latency-adjusted score
d4e5f6g	0.000000	0.000000	0.0	0.0	crash	double width causes OOM
```

## The experiment loop

The experiment runs on a dedicated branch, for example `autoresearch/mar10-vla`.

LOOP FOREVER:

1. Inspect the current git state and identify the current baseline commit.
2. Edit only `train.py` with one concrete experimental idea.
3. Commit the change.
4. Run the experiment: `uv run train.py > run.log 2>&1`
5. Extract the summary metrics from `run.log`.
6. If the summary is missing, treat the run as a crash. Read the traceback, attempt a fast fix if the issue is trivial, otherwise log it as a crash and move on.
7. Append the result to `results.tsv`.
8. If `score` improved, advance the branch and keep the commit.
9. If `score` is equal or worse, revert to the previous baseline commit.

## Timeout and crash policy

- A healthy run should finish in roughly 5 minutes plus startup and evaluation overhead.
- If a run exceeds 10 minutes, kill it, log it as a failure, and revert.
- If a crash is caused by a trivial bug introduced in `train.py`, fix it and rerun once.
- If the core idea is broken, log the crash and move on.

## Never Stop

Once the loop has started, do not stop after partial progress and do not ask the human for permission to continue. Keep going until you are manually interrupted.

Only pause if there is a true external blocker, such as:

- the dataset is missing or malformed
- the environment is broken in a way you cannot repair from the repo
- the human must make a decision that changes the experiment objective

Otherwise continue researching indefinitely. The user may be asleep; autonomous continuation is the expected mode.
