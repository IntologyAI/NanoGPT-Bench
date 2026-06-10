# NanoGPT-Bench

**NanoGPT-Bench** is a benchmark for evaluating AI systems' ability to perform open-ended, long-horizon frontier ML research. It is built on top of the popular GPT-2 pretraining speedrun challenge [*NanoGPT Speedrun*](https://github.com/kellerjordan/modded-nanogpt), and measures how well autonomous coding agents can recover historical human progress on the leaderboard.

## Overview

In NanoGPT-Bench, agents work *fully autonomously* — with no human intervention and no internet access — to improve a strong human starting point on the *NanoGPT Speedrun*. Agents have a fixed compute budget for experimentation and submit candidate solutions through a `submit` command that:

1. Checks competition rules via an LLM judge (mirroring the original *NanoGPT Speedrun* review process).
2. Retimes the candidate across ten runs to confirm a statistically significant speedup.

The benchmark is parameterized by the starting human record and the compute budget, so the setup can be refreshed over time to avoid contamination.

### Why NanoGPT-Bench?

We've found three properties important for autonomous research evaluation:

1. **An open-ended problem** that requires agents to come up with ideas themselves, not just follow instructions.
2. **A strong, optimized starting point** so progress can't be confounded by low-hanging fruit.
3. **A long-horizon human reference** that highlights current deficiencies and indicates room for improvement.

The *NanoGPT Speedrun* is uniquely suited as an environment for autonomous research evaluation: it has a long history of expert human submissions, a clear validation oracle, and an open-ended optimization target.

### Initial Results

We evaluated three frontier coding agents — Codex (GPT-5.4 xhigh), Claude Code (Opus 4.6 Max), and a Claude Code variant using [Autoresearch](https://github.com/karpathy/autoresearch)-style prompting — each with a 512 H100-hour compute budget, starting from the September 3rd, 2025 human world record.

All baselines recover **less than 10%** of the speedup achieved by human world records over the subsequent five months (September 3rd, 2025 – January 19th, 2026):

| Baseline | % of Human Progress Recovered |
| --- | --- |
| Autoresearch (Opus 4.6 Max) | 9.3% |
| Codex (GPT-5.4 xhigh) | 8.6% |
| Claude Code (Opus 4.6 Max) | 8.2% |

Agents spent the majority of their compute on hyperparameter tuning. By contrast, ~77% of human world records introduce algorithmic changes. See the [blog post](#) for the full analysis.

![Figure 1. Best training time achieved by agents over a fixed H100 GPU hour budget, starting from the human world record as of September 3rd, 2025. Progress is shown as a percentage of the speedup achieved by the January 19th, 2026 human world record. All coding agent baselines were given a budget of 512 H100 GPU hours each, and recover less than 10% of the human world record progress.](assets/figure1.png)

## Repository Layout

```
NanoGPT-Bench/
├── assets/                        # README figures and other static media
│   └── figure1.png
├── nanogpt/                       # Host-side harness (driver, agents, prompts, launchers)
│   ├── driver.py                  # Container launcher invoked by nanogpt/run/*.sh
│   ├── prompts/                   # Shared agent prompts mounted into every run
│   │   ├── RULES.md
│   │   ├── problem.txt
│   │   ├── local_prompt.md
│   │   └── resume_prompt.md
│   ├── agents/                    # Per-agent harnesses (install.sh + run.sh entrypoint)
│   │   ├── claude/
│   │   ├── codex/
│   │   └── autoresearch/          # also carries its own `prompts/` overlay
│   └── run/                       # Top-level launcher scripts (entry points for a user)
│       ├── claude_local.sh
│       ├── claude_autoresearch_local.sh
│       └── codex_local.sh
├── image/                         # Docker build context (training environment + submit validator)
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── submit.sh                  # installed in-container as `submit`
│   ├── codebase/data/             # FineWeb10B shard fetcher (baked into the image)
│   └── tools/                     # submit validator (comparability judge + p-value retiming);
│                                  # copied to /opt/nanogpt/tools inside the container
└── human_baselines/               # Snapshot of historical human-record submissions (run.sh +
                                   # train_gpt.py per record). The 2025-09-03_FA3 record is the
                                   # comparability anchor; its serialized form is also baked
                                   # into image/tools/baseline_code.txt.
```

## Running the Benchmark

1. Build the image (one-time):
   ```bash
   docker build -t nanogpt-bench image
   ```
   The build prefetches 9 FineWeb10B training shards plus the validation shard into `/workspace/data/fineweb10B/` inside the image.

2. The Docker volume `nanogpt-bench-data` is mounted at `/workspace/data` at runtime. On first launch Docker auto-populates it from the shards baked into the image; subsequent runs reuse it. Override with `--data-volume` (or `BENCHMARK_DATA_VOLUME`) if you want a different volume name.

3. Export the credentials your chosen agent needs and the session-hours budget, then launch one of:
   ```bash
   export ANTHROPIC_API_KEY=...
   export BENCHMARK_SESSION_HOURS=24
   bash nanogpt/run/claude_local.sh
   # or: bash nanogpt/run/claude_autoresearch_local.sh
   # or: OPENAI_API_KEY=... CODEX_API_KEY=$OPENAI_API_KEY bash nanogpt/run/codex_local.sh
   ```

Each launcher invokes `nanogpt/driver.py`, which copies the `2025-09-03_FA3` human record into a fresh timestamped workspace under `runs/`, mounts the `nanogpt-bench-data` volume into the container, and runs the agent's `run.sh`. The driver streams the container logs to the terminal and persists the agent's events and renderer output under the run directory. Agents validate intermediate candidates by calling `submit /workspace/submissions/submission_N`, which runs the in-container comparability + p-value check and exits `0` on success.

## How to Cite

If you use NanoGPT-Bench in your research, please cite:

```bibtex
@misc{intology2026nanogptbench,
  title  = {NanoGPT-Bench: Evaluating Autonomous Research Agents on the NanoGPT Speedrun},
  author = {Intology},
  year   = {2026},
  howpublished = {\url{https://github.com/IntologyAI/NanoGPT-Bench}},
}
```

## Acknowledgements

NanoGPT-Bench is built on top of [modded-nanogpt](https://github.com/kellerjordan/modded-nanogpt) by Keller Jordan and the *NanoGPT Speedrun* community, whose record submissions provide the human reference trajectory used in this benchmark.
