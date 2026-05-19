#!/usr/bin/env python3
"""Validate a candidate submission against the hard-coded baseline record."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import textwrap
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import comparable
import overfit
import pvalue

NUM_RUNS: Final = 10
VAL_LOSS_THRESHOLD: Final = 3.28
P_VALUE_THRESHOLD: Final = 0.01
BASELINE_NAME: Final = "2025-09-03_FA3"
RESULT_RE: Final = re.compile(r"val_loss:(\d+\.\d+)\s+train_time:(\d+)ms")

RULES: Final = (
    "Do not modify the environment dependencies.\n\n"
    "The optimized code must still reach val_loss <= 3.28. The final goal is"
    " to have the lowest average training time over 10 runs where the"
    " one-sided p-value that the val_loss is below 3.28 is less than 0.01."
    " (The p-value and val loss constraints are verified separately by running the code;"
    " do not consider it when judging the candidate codebase.)\n\n"
    "The final training code must run for a fixed number of steps.\n\n"
    "Ensure any modifications keep validation loss fully valid against prior"
    " versions.\n\n"
    "The number of torch.compile call sites may change if needed, but do not"
    " introduce new torch.compile flags or modes. Keep using only the existing"
    " flag patterns already alsopresent in the script rather than adding options"
    " such as mode=\"max-autotune\" or similar compile-time overrides.\n\n"
    "Do not change the current inductor and compile-related flag setup."
    " torch._dynamo.config.recompile_limit is set to 64,"
    " torch._inductor.config.coordinate_descent_tuning is intentionally left"
    " disabled in a commented line, and no additional torch._inductor.config"
    " settings, max-autotune settings, or similar compile-time overrides"
    " should be introduced.\n\n"
    "Do not change the underlying data streams or next-token objective."
    " Training data comes from fineweb_train_*.bin with BOS-aligned packing"
    " through _load_data_shard, BOSFinder, and distributed_data_generator.\n\n"
    "Do not change the validation data pipeline. Validation must continue to"
    " read fineweb_val_*.bin, keep val_tokens = 10485760 and"
    " val_batch_size = 4 * 64 * 1024 * 8, create a fresh"
    " distributed_data_generator(..., align_to_bos=False) for each validation"
    " pass, use buf[:-1] as inputs and buf[1:] as targets, run with the"
    " current step's window size, average across val_steps, and all-reduce"
    " that loss with dist.ReduceOp.AVG. No changes may be introduced which"
    " make the validation loss invalid against prior versions.\n\n"
    "Preserve the timing methodology used for scoring: 30 warmup steps,"
    " restore the initial model and optimizer state after warmup, start timing"
    " after warmup, pause timing during validation, and resume timing after"
    " validation.\n\n"
    "Do not optimize print statements.\n\n"
    "Ensure that the full warmup from an empty cache stays under 10"
    " minutes.\n\n"
    "Do not brute force tune parameters to the validation set - make sure"
    " changes made to hyperparameters are principled and motivated."
)

BASELINE_CODE: Final = Path(
    __file__,
).resolve().parent.joinpath("baseline_code.txt").read_text(encoding="utf-8")


@dataclass(frozen=True)
class Run:
    """One training run result."""

    index: int
    exit_code: int
    val_loss: float | None
    train_time_ms: float | None


@dataclass(frozen=True)
class Verdict:
    """Final submission verdict."""

    valid: bool
    validity_summary: str
    p_value: float | None
    p_value_met: bool
    avg_train_time_ms: float | None
    total_runs: int
    successful_runs: int
    val_losses: tuple[float, ...]
    train_times_ms: tuple[float, ...]


def prompt(candidate_name: str, candidate_text: str) -> str:
    """Build the comparability prompt with the hard-coded baseline and rules."""

    return (
        textwrap.dedent(
            f"""
            You are comparing two serialized codebases.

            Base your answer only on the serialized code in this prompt.
            Do not inspect files, run commands, or rely on outside context.

            The following rules govern valid submissions:

            {RULES}

            Your job is twofold:
            1. Decide whether the candidate codebase is cheating — i.e. whether
               it makes its reported validation loss and/or training time ms
               incomparable with the canonical initial record.
            2. Decide whether the candidate violates any of the rules listed
               above.

            Set cheating to true when EITHER of these holds:

            The candidate materially changes the measurement boundary or the
            meaning of the reported validation loss. Examples include:
            - moving work that the canonical record performs inside the measured
              path into setup, caching, preprocessing, warmup, checkpoint
              loading, or other untimed sections
            - precomputing, reusing, or skipping work that the canonical record
              still performs inside the measured path
            - changing the evaluated data, masking, token selection,
              normalization, aggregation, validation window, or evaluation
              cadence in a way that breaks a like-for-like comparison
            - other code changes that make the candidate's val loss no longer
              directly comparable to the canonical record's val loss

            OR the candidate violates any of the rules above. Examples include:
            - introducing new torch.compile flags or modes not present in the
              canonical record
            - adding or changing torch._inductor.config settings, max-autotune
              settings, or similar compile-time overrides
            - modifying environment dependencies (e.g. installing, upgrading,
              or replacing pip/conda packages)
            - changing the data streams, next-token objective, or validation
              data pipeline
            - altering the timing methodology used for scoring
            - brute force tuning parameters to the validation set

            Changing the number of training iterations (num_iterations,
            train_steps) is acceptable and does not constitute cheating,
            as long as the validation loss is still computed honestly over the
            same data with the same metric. A candidate that trains for fewer
            or more steps but evaluates val loss identically is still valid.

            Do not treat ordinary refactors, speedups that stay inside the same
            measured path, code motion that preserves the same timing boundary
            and metric definition, or changes to the number of training steps
            as evidence on their own.
            Be conservative. Set cheating to true only when the candidate
            shows stronger code-level evidence than the canonical record that
            the comparison is no longer direct or that a rule has been violated.

            candidate must exactly match {candidate_name!r}.
            Each item in reasons must cite concrete file paths from the
            serialized code.
            If cheating is false, reasons should still note the strongest
            code-level evidence you considered for why the comparison remains
            direct and all rules are satisfied.
            summary must be concise.

            Canonical name: {BASELINE_NAME}
            Candidate name: {candidate_name}

            Canonical codebase begins here.
            {BASELINE_CODE}
            Canonical codebase ends here.

            Candidate codebase begins here.
            {candidate_text}
            Candidate codebase ends here.
            """
        ).strip()
        + "\n"
    )


def check(directory: Path) -> comparable.Decision:
    """Run the comparability check against the hard-coded baseline."""

    candidate_text = overfit.serialize(directory)
    prompt_text = prompt(directory.name, candidate_text)
    raw = comparable.run_codex(
        prompt_text, overfit.DEFAULT_MODEL, overfit.DEFAULT_REASONING,
    )
    return comparable.reply(directory.name, raw)


def parse(output: str) -> tuple[float | None, float | None]:
    """Extract val_loss and train_time_ms from training output."""

    matches = RESULT_RE.findall(output)
    if not matches:
        return None, None
    val_loss_str, train_time_str = matches[-1]
    return float(val_loss_str), float(train_time_str)


def execute(directory: Path, index: int) -> Run:
    """Run one training invocation of run.sh."""

    data_link = directory / "data"
    workspace_data = directory.parent.parent / "data"
    if not data_link.exists() and workspace_data.is_dir():
        data_link.symlink_to(workspace_data)
    completed = subprocess.run(
        ["bash", "run.sh"],
        cwd=directory,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    val_loss, train_time_ms = parse(completed.stdout)
    if completed.returncode != 0 and val_loss is None:
        tail = completed.stdout.strip().splitlines()[-20:]
        print("  output tail:\n" + "\n".join(f"    {l}" for l in tail), flush=True)
    return Run(
        index=index,
        exit_code=completed.returncode,
        val_loss=val_loss,
        train_time_ms=train_time_ms,
    )


def evaluate(directory: Path, count: int) -> list[Run]:
    """Execute training runs sequentially and collect results."""

    runs: list[Run] = []
    for i in range(count):
        print(f"run {i + 1}/{count}", flush=True)
        result = execute(directory, i)
        print(
            f"  exit_code={result.exit_code}"
            f" val_loss={result.val_loss}"
            f" train_time_ms={result.train_time_ms}",
            flush=True,
        )
        runs.append(result)
    return runs


def submit(directory: Path, count: int) -> Verdict:
    """Full submission validation: comparability check, training runs, p-value."""

    assert (directory / "run.sh").is_file(), f"missing run.sh in {directory}"
    assert (directory / "train_gpt.py").is_file(), f"missing train_gpt.py in {directory}"

    print(f"checking validity of {directory.name}...", flush=True)
    decision = check(directory)
    print(
        f"  cheating={decision.cheating}"
        f" confidence={decision.confidence}"
        f" summary={decision.summary}",
        flush=True,
    )

    if decision.cheating:
        return Verdict(
            valid=False,
            validity_summary=decision.summary,
            p_value=None,
            p_value_met=False,
            avg_train_time_ms=None,
            total_runs=0,
            successful_runs=0,
            val_losses=(),
            train_times_ms=(),
        )

    print(f"running {count} training runs...", flush=True)
    runs = evaluate(directory, count)
    val_losses = [r.val_loss for r in runs if r.val_loss is not None]
    train_times = [r.train_time_ms for r in runs if r.train_time_ms is not None]

    p_val = pvalue.pvalue(val_losses, VAL_LOSS_THRESHOLD)

    return Verdict(
        valid=True,
        validity_summary=decision.summary,
        p_value=p_val,
        p_value_met=p_val is not None and p_val < P_VALUE_THRESHOLD,
        avg_train_time_ms=statistics.fmean(train_times) if train_times else None,
        total_runs=count,
        successful_runs=len(val_losses),
        val_losses=tuple(val_losses),
        train_times_ms=tuple(train_times),
    )


def main(argv: list[str] | None = None) -> int:
    """Validate a candidate submission directory."""

    built = argparse.ArgumentParser(
        description="Validate a submission against the baseline record.",
    )
    built.add_argument(
        "directory",
        type=Path,
        help="Path to the candidate submission directory.",
    )
    built.add_argument(
        "--runs",
        type=int,
        default=NUM_RUNS,
        help=f"Number of training runs. Default: {NUM_RUNS}.",
    )
    args = built.parse_args(argv)

    directory = args.directory.expanduser().resolve()
    assert directory.is_dir(), f"directory does not exist: {directory}"

    verdict = submit(directory, args.runs)
    print(json.dumps(asdict(verdict), indent=2))
    return 0 if verdict.valid and verdict.p_value_met else 1


if __name__ == "__main__":
    raise SystemExit(main())
