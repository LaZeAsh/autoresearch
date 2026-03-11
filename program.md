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

1. Look at the git state: the current branch/commit we're on
2. Tune train.py with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: uv run train.py > run.log 2>&1 (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: grep "^val_bpb:\|^peak_vram_mb:" run.log
6. If the grep output is empty, the run crashed. Run tail -n 50 run.log to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If val_bpb improved (lower), you "advance" the branch, keeping the git commit
9. If val_bpb is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher
trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

Timeout: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

Crashes: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

NEVER STOP: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working indefinitely until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!

## Ideas

You are working on a model for autonomous drone focused on task completion. Think about what different architectures have to offer

These are some of my ideas that you can build on / try out:

- Follow SmolVLA approach, initialize action head weights from scratch otherwise initialize action head and run LoRA
- Look into Physical Intelligence's research with Pi 0 / 0.5 / 0.6, try using their checkpoints / architecture with the dataset
- Try a RL approach to training the action head, with a reward function that encourages task completion, improving long horizon tasks
- Different research papers to look into:
  - AIR-VLA: https://arxiv.org/html/2601.21602v2
  - DroneVLA: https://arxiv.org/abs/2601.13809
  - VLA=AN: https://arxiv.org/abs/2512.15258
  - Real-time chunking (RTC): https://arxiv.org/abs/2506.07339

If you pick a major architectural idea stick with it for at least 20 iterations before trying another major approach. This is to ensure that you are able to evaluate the impact of the change on the overall performance of the model maintaing consistency.

## Resources

You are running on a RTX 5090 (32 gb vram), take advantage of it