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

## Code

*Placeholder — code release coming soon.*

## Running the Benchmark

*Placeholder — instructions for running baselines and submitting new agents coming soon.*

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
