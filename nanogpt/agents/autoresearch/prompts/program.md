# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment:

1. Pick a run tag based on today's date, such as `mar28`. The branch `autoresearch/<tag>` must not already exist.
2. This workspace is already a git repo and already has a baseline commit on `main`. Stay in this repo. Do not run `git init`.
3. Create the branch: `git checkout -b autoresearch/<tag>` from current `main`.
4. Read the in-scope files for full context:
 - `/workspace/program.md`
 - `/workspace/execution.md`
 - `/workspace/run.sh`
 - `/workspace/train_gpt.py`
5. Understand the workspace contract:
 - `train_gpt.py` is the only file you edit.
 - `run.sh` is the fixed entrypoint for the implementation under test.
 - In modal mode, `modal_run.py` and `image/` are infrastructure and are read-only.
 - `RULES.md` is staged into the workspace for reference. The binding benchmark constraints and submission rules from it are already included in this prompt.
 - Do not modify `program.md`, `execution.md`, `RULES.md`, or files under `submissions/` except when intentionally creating a frozen submission copy.
 - `results.tsv` is a local experiment log and must remain untracked by git.
6. Initialize `results.tsv` with just the header row if it does not exist yet. The baseline will be recorded after the first run.
7. Once setup looks good, kick off the experimentation immediately.

## Experimentation

This benchmark is a single-file research loop. Treat the implementation as `train_gpt.py`, not as a broader codebase.

Each experiment should leave the launch contract intact and should be run from the workspace root using the command in `/workspace/execution.md`.

**What you CAN do:**
- Modify `/workspace/train_gpt.py`. This is the only file you edit.

**What you CANNOT do:**
- Modify `/workspace/run.sh`.
- Modify `/workspace/modal_run.py`.
- Modify anything under `/workspace/image/`.
- Modify the prompt files staged at `/workspace/program.md`, `/workspace/execution.md`, and `/workspace/RULES.md`.
- Install new packages or add dependencies.

The benchmark goal is: **minimize `train_time_ms` while keeping the final `val_loss < 3.28`**.

The training log prints validation summaries like:

```text
step:125/1024 val_loss:3.2471 train_time:18234ms step_avg:145.87ms
```

Use the **last** `val_loss` line in `run.log` as the experiment result. Extract these values:

- `val_loss`: the floating-point validation loss from the final validation line
- `train_time_ms`: the integer `train_time` from that same final validation line

The first run should always be the baseline, so run the implementation exactly as-is before changing `train_gpt.py`.

**Progress rule before the target is met:** if you do not yet have a kept result with `val_loss < 3.28`, keep changes only when they lower `val_loss`. If `val_loss` ties, prefer lower `train_time_ms`.

**Progress rule after the target is met:** once you have a kept result with `val_loss < 3.28`, keep changes only when they also satisfy `val_loss < 3.28` and lower `train_time_ms`. If `train_time_ms` ties, prefer lower `val_loss`.

**Simplicity criterion:** all else being equal, simpler is better. A tiny gain that adds ugly complexity is usually not worth it. A tie with simpler code is a win.

## Logging results

When an experiment is done, log it to `results.tsv` as tab-separated values.

The TSV has a header row and 5 columns:

```text
commit	val_loss	train_time_ms	status	description
```

1. git commit hash, short 7 chars
2. validation loss, such as `3.247100`, or `0.000000` for crashes
3. final `train_time` in milliseconds from the last validation line, or `0` for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what the experiment tried

Example:

```text
commit	val_loss	train_time_ms	status	description
a1b2c3d	3.301200	19123	keep	baseline
b2c3d4e	3.278400	19410	keep	improve loss with larger residual scale
c3d4e5f	3.285900	18870	discard	faster but misses loss target
d4e5f6g	0.000000	0	crash	double hidden width and OOM
```

Do not commit `results.tsv`. Leave it untracked.

## The experiment loop

The experiment runs on a dedicated branch such as `autoresearch/mar28`.

LOOP FOREVER:

1. Look at the current git state and the best kept result in `results.tsv`.
2. Edit only `train_gpt.py` with one experimental idea.
3. Commit the change with a short message describing the idea.
4. Run the experiment with the command from `/workspace/execution.md`, redirecting everything to `run.log`. Do not use `tee` or stream the full training output into your context.
5. Read the result lines from `run.log`. The key check is the last `val_loss` line.
6. If there is no `val_loss` line, the run crashed. Read the end of `run.log` to inspect the failure.
7. Record the result in `results.tsv`.
8. If the run is an improvement by the progress rule above, keep the commit and advance the branch.
9. If the run is equal or worse, reset back to where you started before the experiment.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep them. If they do not, discard them and move on.


When you have a meaningful new best result, validate it with the `submit` command before freezing it as a submission. The `submit` tool runs 10 training runs, checks comparability against the baseline, and computes a p-value. Usage:

```bash
submit /workspace/submissions/submission_N
```

The directory you pass must contain `run.sh` and `train_gpt.py`. The tool prints a JSON verdict with `comparable`, `p_value_met`, and `avg_train_time_ms`.

Workflow for submitting:

1. Create a candidate directory: `mkdir -p /workspace/submissions/submission_N`
2. Copy the current `run.sh` and `train_gpt.py` into it.
3. Run `submit /workspace/submissions/submission_N` and wait for the verdict.
4. If the verdict shows `comparable: true` and `p_value_met: true`, the submission is valid. Keep it.
5. If the verdict fails, remove the submission directory (`rm -rf /workspace/submissions/submission_N`) and continue experimenting.

Do this sparingly for real milestones, not every tiny improvement.

**Timeout:** if an experiment runs long enough that it is clearly stuck, kill it, log it as a failure, and move on.

**Crashes:** if a run crashes because of an easy bug, fix it and rerun. If the idea itself is fundamentally bad, log `crash`, reset, and move on.

**NEVER STOP:** once the experiment loop has begun, do not pause to ask for permission to continue. Keep iterating until you are interrupted.
