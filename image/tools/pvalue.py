#!/usr/bin/env python3
"""Compute per-subfolder validation-loss p-values and relative training times for a batch of repeated runs."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import statistics
import sys
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Final

DEFAULT_THRESHOLD: Final = 3.28
DEFAULT_TSV_NAME: Final = "pvalue.tsv"
DEFAULT_JSON_NAME: Final = "pvalue.json"
BETA_EPS: Final = 3.0e-14
BETA_FPMIN: Final = 1.0e-300
BETA_MAX_ITER: Final = 200
TRAIN_TIME_MS_RE: Final = re.compile(r"val_loss:\d+(?:\.\d+)?\s+train_time:(\d+)ms")


@dataclass(frozen=True)
class Run:
    """One repeated run result."""

    group: str
    exit_code: int | None
    val_loss: float | None
    train_time_ms: float | None


@dataclass(frozen=True)
class Summary:
    """Aggregated statistics for one subfolder."""

    group: str
    runs: int
    scored_runs: int
    failures: int
    below_threshold: int
    mean_val_loss: float | None
    best_val_loss: float | None
    mean_train_time_ms: float | None
    mean_seconds_faster: float | None
    p_value: float | None


def path_arg(raw: str) -> Path:
    """Resolve and validate a batch directory path."""

    path = Path(raw).expanduser().resolve()
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"directory does not exist: {path}")
    return path


def parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    built = argparse.ArgumentParser(
        description=(
            "Compute the one-sided p-value that the final validation loss in each "
            "batch subfolder is below a threshold."
        )
    )
    built.add_argument("batch_dir", type=path_arg, help="Batch directory containing repeated runs (results.tsv or rep_*/run.json).")
    built.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Validation-loss threshold for the one-sided test. Default: {DEFAULT_THRESHOLD}.",
    )
    built.add_argument("--json", action="store_true", help="Print JSON instead of TSV.")
    return built


def number(raw: str | None) -> float | None:
    """Parse an optional floating-point field."""

    if raw is None:
        return None
    value = raw.strip()
    if not value:
        return None
    return float(value)


def integer(raw: str | None) -> int | None:
    """Parse an optional integer field."""

    if raw is None:
        return None
    value = raw.strip()
    if not value:
        return None
    return int(value)


def train_time_ms(log_path: Path) -> float | None:
    """Parse the final cumulative train time in milliseconds from one console log."""

    if not log_path.is_file():
        return None
    matches = TRAIN_TIME_MS_RE.findall(log_path.read_text(encoding="utf-8"))
    if not matches:
        return None
    return float(matches[-1])


def results(path: Path) -> list[Run]:
    """Load repeated run summaries from a results.tsv file."""

    batch_dir = path.parent
    rows: list[Run] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            group = (row.get("record") or "").strip()
            if not group:
                raise ValueError(f"Missing record name in {path}")
            tt = number(row.get("train_time_ms"))
            if tt is None:
                local = (row.get("local_dir") or "").strip()
                if local:
                    local_path = Path(local)
                    tt = train_time_ms(local_path / "console.log")
                    if tt is None:
                        tt = train_time_ms(batch_dir / group / local_path.name / "console.log")
            rows.append(
                Run(
                    group=group,
                    exit_code=integer(row.get("exit_code")),
                    val_loss=number(row.get("val_loss")),
                    train_time_ms=tt,
                )
            )
    return rows


def reps(path: Path) -> list[Run]:
    """Load repeated run summaries from rep_*/run.json files."""

    rows: list[Run] = []
    for child in sorted(path.iterdir()):
        if not child.is_dir() or not child.name.startswith("rep_"):
            continue
        run_path = child / "run.json"
        if not run_path.is_file():
            continue
        payload = json.loads(run_path.read_text(encoding="utf-8"))
        record = payload.get("record")
        group = str(record) if record is not None else path.name
        val_loss = payload.get("val_loss")
        raw_tt = payload.get("train_time_ms")
        tt = float(raw_tt) if raw_tt is not None else train_time_ms(child / "console.log")
        rows.append(
            Run(
                group=group,
                exit_code=int(payload["exit_code"]) if payload.get("exit_code") is not None else None,
                val_loss=float(val_loss) if val_loss is not None else None,
                train_time_ms=tt,
            )
        )
    return rows


def rows(path: Path) -> list[Run]:
    """Load repeated runs from a batch directory."""

    results_path = path / "results.tsv"
    if results_path.is_file():
        loaded = results(results_path)
        if loaded:
            return loaded
    direct = reps(path)
    if direct:
        return direct
    nested: list[Run] = []
    for child in sorted(path.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        nested.extend(reps(child))
    if nested:
        return nested
    raise FileNotFoundError(f"Could not find results.tsv or rep_*/run.json under {path}")


def grouped(runs: list[Run]) -> dict[str, list[Run]]:
    """Group runs by subfolder name."""

    groups: dict[str, list[Run]] = {}
    for run in runs:
        bucket = groups.setdefault(run.group, [])
        bucket.append(run)
    return groups


def betacf(a: float, b: float, x: float) -> float:
    """Evaluate the continued fraction for the incomplete beta function."""

    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < BETA_FPMIN:
        d = BETA_FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, BETA_MAX_ITER + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < BETA_FPMIN:
            d = BETA_FPMIN
        c = 1.0 + aa / c
        if abs(c) < BETA_FPMIN:
            c = BETA_FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < BETA_FPMIN:
            d = BETA_FPMIN
        c = 1.0 + aa / c
        if abs(c) < BETA_FPMIN:
            c = BETA_FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) <= BETA_EPS:
            return h
    raise ValueError("incomplete beta did not converge")


def regbeta(a: float, b: float, x: float) -> float:
    """Return the regularized incomplete beta function."""

    if not 0.0 <= x <= 1.0:
        raise ValueError(f"x must be between 0 and 1, got {x}")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0
    log_term = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    log_term += a * math.log(x) + b * math.log1p(-x)
    front = math.exp(log_term)
    pivot = (a + 1.0) / (a + b + 2.0)
    if x < pivot:
        return front * betacf(a, b, x) / a
    return 1.0 - front * betacf(b, a, 1.0 - x) / b


def student_cdf(t_stat: float, degrees: int) -> float:
    """Return the Student-t cumulative probability for one statistic."""

    if degrees < 1:
        raise ValueError(f"degrees must be positive, got {degrees}")
    x = degrees / (degrees + t_stat * t_stat)
    tail = 0.5 * regbeta(0.5 * degrees, 0.5, x)
    if t_stat >= 0.0:
        return 1.0 - tail
    return tail


def pvalue(values: list[float], threshold: float) -> float | None:
    """Return the one-sided p-value that the mean is below the threshold."""

    if not values:
        return None
    mean = statistics.fmean(values)
    if len(values) == 1:
        return 0.0 if mean < threshold else 1.0
    stdev = statistics.stdev(values)
    if stdev == 0.0:
        return 0.0 if mean < threshold else 1.0
    t_stat = (mean - threshold) / (stdev / math.sqrt(len(values)))
    return student_cdf(t_stat, len(values) - 1)


def summarize(group: str, runs: list[Run], threshold: float) -> Summary:
    """Aggregate one run group into a summary row."""

    values = [run.val_loss for run in runs if run.val_loss is not None]
    scored = [value for value in values]
    times = [run.train_time_ms for run in runs if run.train_time_ms is not None]
    return Summary(
        group=group,
        runs=len(runs),
        scored_runs=len(scored),
        failures=sum(1 for run in runs if run.exit_code not in (None, 0)),
        below_threshold=sum(1 for value in scored if value < threshold),
        mean_val_loss=statistics.fmean(scored) if scored else None,
        best_val_loss=min(scored) if scored else None,
        mean_train_time_ms=statistics.fmean(times) if times else None,
        mean_seconds_faster=None,
        p_value=pvalue(scored, threshold),
    )


def relative(items: list[Summary]) -> list[Summary]:
    """Return summaries with mean train-time deltas against the first group."""

    if not items:
        return items
    baseline = items[0].mean_train_time_ms
    if baseline is None:
        return items
    return [
        replace(
            item,
            mean_seconds_faster=round((baseline - item.mean_train_time_ms) / 1000.0, 4)
            if item.mean_train_time_ms is not None
            else None,
        )
        for item in items
    ]


def summaries(path: Path, threshold: float) -> list[Summary]:
    """Return aggregated per-group summaries for one batch directory."""

    runs = rows(path)
    items = [summarize(group, grouped_runs, threshold) for group, grouped_runs in sorted(grouped(runs).items())]
    return relative(items)


def payload(batch_dir: Path, threshold: float, items: list[Summary]) -> dict[str, object]:
    """Build the JSON payload for one batch."""

    return {
        "batch_dir": str(batch_dir),
        "threshold": threshold,
        "groups": [asdict(item) for item in items],
    }


def json_text(batch_dir: Path, threshold: float, items: list[Summary]) -> str:
    """Render JSON output for one batch."""

    return json.dumps(payload(batch_dir, threshold, items), indent=2) + "\n"


def fieldnames() -> list[str]:
    """Return the TSV field order."""

    return [
        "group",
        "runs",
        "scored_runs",
        "failures",
        "below_threshold",
        "threshold",
        "mean_val_loss",
        "best_val_loss",
        "mean_train_time_ms",
        "mean_seconds_faster",
        "p_value",
    ]


def tsv_text(threshold: float, items: list[Summary]) -> str:
    """Render TSV output for one batch."""

    sink = io.StringIO()
    writer = csv.DictWriter(sink, fieldnames=fieldnames(), delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for item in items:
        row = asdict(item)
        row["threshold"] = threshold
        writer.writerow(row)
    return sink.getvalue()


def output_path(batch_dir: Path, as_json: bool) -> Path:
    """Return the persisted output path for one batch."""

    name = DEFAULT_JSON_NAME if as_json else DEFAULT_TSV_NAME
    return batch_dir / name


def write_output(batch_dir: Path, text: str, as_json: bool) -> Path:
    """Persist rendered output into the batch directory."""

    path = output_path(batch_dir, as_json)
    path.write_text(text, encoding="utf-8")
    return path


def main() -> None:
    """Run the standalone batch p-value tool."""

    args = parser().parse_args()
    items = summaries(args.batch_dir, args.threshold)
    if args.json:
        text = json_text(args.batch_dir, args.threshold, items)
        write_output(args.batch_dir, text, as_json=True)
        sys.stdout.write(text)
        return
    text = tsv_text(args.threshold, items)
    write_output(args.batch_dir, text, as_json=False)
    sys.stdout.write(text)


if __name__ == "__main__":
    main()
