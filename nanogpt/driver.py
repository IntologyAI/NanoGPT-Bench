"""Run a benchmark agent inside the container image.

The driver copies a benchmark source snapshot into a fresh timestamped
workspace under ``runs/``, snapshots an agent directory plus any named
support directories into ``support/``, mounts that per-run state into the
container, and invokes the agent's ``run.sh`` entrypoint with the provided
arguments.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import time
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
from pathlib import Path
from types import FrameType
from typing import Final


ROOT: Final = Path(__file__).resolve().parents[1]
BASELINES: Final = ROOT / "human_baselines"
RUNS: Final = ROOT / "runs"
DEFAULT_IMAGE: Final = "nanogpt-bench"
DEFAULT_TIMEOUT_SEC: Final = 24 * 60 * 60
DEFAULT_DATA_VOLUME: Final = "nanogpt-bench-data"
DEFAULT_RECORD: Final = "2025-09-03_FA3"
INTERRUPT_GRACE_SEC: Final = 30
POLL_SEC: Final = 1.0


@dataclass(frozen=True)
class Var:
    """One environment variable passed into the run container."""

    name: str
    value: str


@dataclass(frozen=True)
class Support:
    """One named support directory copied under ``/runner``."""

    name: str
    path: Path


@dataclass(frozen=True)
class Spec:
    """The resolved configuration for one benchmark agent run."""

    agent_name: str
    agent_dir: Path
    agent_args: tuple[str, ...]
    support_dirs: tuple[Support, ...]
    image: str
    source_dir: Path
    runs_dir: Path
    data_volume: str
    timeout_sec: int
    gpus: str | None
    passthrough: tuple[str, ...]
    explicit: tuple[Var, ...]
    run_name: str | None


@dataclass(frozen=True)
class Paths:
    """The timestamped paths created for one run."""

    run_id: str
    run_dir: Path
    workspace: Path
    support: Path
    agent_dir: Path
    metadata: Path
    container_log: Path
    inspect_json: Path


@dataclass(frozen=True)
class Result:
    """The terminal outcome of one containerized agent run."""

    container_name: str
    started_at: str
    ended_at: str
    elapsed_sec: float
    exit_code: int | None
    timed_out: bool


@dataclass(frozen=True)
class Container:
    """The current Docker state for one named container."""

    exists: bool
    running: bool
    exit_code: int | None


@dataclass(frozen=True)
class Relay:
    """One live log relay mirrored to the terminal and run directory."""

    source: subprocess.Popen[bytes]
    sink: subprocess.Popen[bytes]


@dataclass
class State:
    """Mutable driver state shared by normal and interrupted exits."""

    spec: Spec
    paths: Paths
    create_cmd: tuple[str, ...] = ()
    container_name: str | None = None
    log_stream: Relay | None = None
    started_at: str | None = None
    started: float | None = None
    ended_at: str | None = None
    elapsed_sec: float | None = None
    exit_code: int | None = None
    timed_out: bool = False
    interrupted: int | None = None
    finalized: bool = False


@dataclass(frozen=True)
class Artifact:
    """The run artifact paths written by the driver."""

    run_dir: str
    workspace: str
    support: str
    agent_dir: str
    metadata: str
    container_log: str
    inspect_json: str


@dataclass(frozen=True)
class Capture:
    """The final container artifacts captured before removal."""

    inspect_json: str
    container_log: str


@dataclass(frozen=True)
class SupportArtifact:
    """One copied support directory available to the agent."""

    name: str
    source_dir: str
    copied_dir: str


@dataclass(frozen=True)
class Meta:
    """The serialized metadata for one run."""

    run_id: str
    agent_name: str
    agent_args: tuple[str, ...]
    support_dirs: tuple[SupportArtifact, ...]
    image: str
    source_dir: str
    data_volume: str
    timeout_sec: int
    gpus: str | None
    passthrough: tuple[str, ...]
    explicit: tuple[Var, ...]
    docker_create: tuple[str, ...]
    artifacts: Artifact
    result: Result


def parse() -> argparse.Namespace:
    """Parse command line arguments for the agent driver."""

    parser = argparse.ArgumentParser(
        description=(
            "Run a benchmark agent inside the Docker image and persist the "
            "workspace, traces, and metadata under runs/. Pass "
            "BENCHMARK_SESSION_HOURS through --pass-env or --set-env to enable "
            "multi-hour agent session resume."
        )
    )
    parser.add_argument(
        "--agent",
        type=Path,
        required=True,
        help="Path to an agent directory on the host containing run.sh.",
    )
    parser.add_argument(
        "--agent-name",
        default=None,
        help="Logical agent name used in the run directory and metadata.",
    )
    parser.add_argument(
        "--agent-arg",
        action="append",
        default=[],
        help="Argument passed through to the agent run.sh entrypoint.",
    )
    parser.add_argument(
        "--support-dir",
        action="append",
        default=[],
        help="Named directory copied under /runner in NAME=PATH form.",
    )
    parser.add_argument(
        "--image",
        default=os.environ.get("BENCHMARK_IMAGE", DEFAULT_IMAGE),
        help="Docker image tag to run.",
    )
    parser.add_argument(
        "--record",
        default=os.environ.get("BENCHMARK_RECORD", DEFAULT_RECORD),
        help=(
            "Record directory name under --records-dir used when --source-dir "
            "is not set."
        ),
    )
    parser.add_argument(
        "--records-dir",
        type=Path,
        default=Path(os.environ.get("BENCHMARK_RECORDS_DIR", str(BASELINES))),
        help="Parent directory containing benchmark record directories.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Source directory copied into the per-run workspace. Overrides --record.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=RUNS,
        help="Parent directory that receives timestamped run outputs.",
    )
    parser.add_argument(
        "--data-volume",
        default=os.environ.get("BENCHMARK_DATA_VOLUME", DEFAULT_DATA_VOLUME),
        help="Docker volume mounted to /workspace/data inside the container.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=int(os.environ.get("BENCHMARK_TIMEOUT_SEC", DEFAULT_TIMEOUT_SEC)),
        help=(
            "Maximum wall clock time allowed for the agent container. Set this "
            "above BENCHMARK_SESSION_HOURS when multi-hour session resume is "
            "enabled."
        ),
    )
    parser.add_argument(
        "--gpus",
        default=os.environ.get("BENCHMARK_GPUS"),
        help="Optional Docker --gpus value.",
    )
    parser.add_argument(
        "--pass-env",
        action="append",
        default=[],
        help=(
            "Host environment variable name to pass through into the container. "
            "Use this for BENCHMARK_SESSION_HOURS when enabling multi-hour "
            "session resume."
        ),
    )
    parser.add_argument(
        "--set-env",
        action="append",
        default=[],
        help=(
            "Literal environment variable assignment in KEY=VALUE form. Use "
            "this for BENCHMARK_SESSION_HOURS=<hours> when enabling multi-hour "
            "session resume."
        ),
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional suffix appended to the timestamped run id.",
    )
    return parser.parse_args()


def unique(values: list[str]) -> tuple[str, ...]:
    """Keep values in order while removing duplicates."""

    kept: list[str] = []
    for value in values:
        if value not in kept:
            kept.append(value)
    return tuple(kept)


def slug(value: str) -> str:
    """Normalize a string for use in run and container names."""

    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return cleaned or "run"


def stamp() -> str:
    """Return a UTC timestamp suitable for run ids."""

    now = datetime.now(timezone.utc)
    return now.strftime("%Y%m%d_%H%M%S_%f")


def assign(items: list[str]) -> tuple[Var, ...]:
    """Parse literal KEY=VALUE assignments."""

    pairs: list[Var] = []
    for item in items:
        name, sep, value = item.partition("=")
        assert sep == "=", item
        assert name, item
        pairs.append(Var(name=name, value=value))
    return tuple(pairs)


def agent(args: argparse.Namespace) -> tuple[str, Path]:
    """Resolve the selected agent name and host directory."""

    path = args.agent.expanduser().resolve()
    assert path.is_dir(), path
    assert (path / "run.sh").is_file(), path / "run.sh"
    name = args.agent_name or path.stem
    return slug(name), path


def support(items: list[str]) -> tuple[Support, ...]:
    """Parse named support directories in ``NAME=PATH`` form."""

    dirs: list[Support] = []
    names: list[str] = []
    for item in items:
        name, sep, raw_path = item.partition("=")
        assert sep == "=", item
        assert name, item
        assert raw_path, item
        assert "/" not in name, item
        assert name not in {".", "..", "agent"}, item
        assert name not in names, item
        path = Path(raw_path).expanduser().resolve()
        assert path.is_dir(), path
        names.append(name)
        dirs.append(Support(name=name, path=path))
    return tuple(dirs)


def record(value: str) -> str:
    """Validate one record directory name."""

    name = value.strip()
    assert name, value
    assert "/" not in name, value
    assert name not in {".", ".."}, value
    return name


def source(args: argparse.Namespace) -> Path:
    """Resolve the selected source directory."""

    if args.source_dir is not None:
        return args.source_dir.expanduser().resolve()
    records_dir = args.records_dir.expanduser().resolve()
    assert records_dir.is_dir(), records_dir
    return records_dir / record(args.record)


def spec(args: argparse.Namespace) -> Spec:
    """Resolve the runtime specification from command line arguments."""

    agent_name, agent_dir = agent(args)
    source_dir = source(args)
    runs_dir = args.runs_dir.expanduser().resolve()
    assert source_dir.is_dir(), source_dir
    runs_dir.mkdir(parents=True, exist_ok=True)
    return Spec(
        agent_name=agent_name,
        agent_dir=agent_dir,
        agent_args=tuple(args.agent_arg),
        support_dirs=support(args.support_dir),
        image=args.image,
        source_dir=source_dir,
        runs_dir=runs_dir,
        data_volume=args.data_volume,
        timeout_sec=args.timeout_sec,
        gpus=args.gpus,
        passthrough=unique(args.pass_env),
        explicit=assign(args.set_env),
        run_name=args.run_name,
    )


def layout(spec: Spec) -> Paths:
    """Create the timestamped host layout for one run."""

    label = spec.agent_name
    if spec.run_name:
        label = f"{label}-{spec.run_name}"
    run_id = f"{stamp()}_{slug(label)}"
    run_dir = spec.runs_dir / run_id
    support_dir = run_dir / "support"
    workspace = run_dir / "workspace"
    return Paths(
        run_id=run_id,
        run_dir=run_dir,
        workspace=workspace,
        support=support_dir,
        agent_dir=support_dir / "agent",
        metadata=run_dir / "metadata.json",
        container_log=run_dir / "container.log",
        inspect_json=run_dir / "container.json",
    )


def stage(spec: Spec, paths: Paths) -> None:
    """Materialize the workspace and support files for one run."""

    paths.support.mkdir(parents=True, exist_ok=True)
    shutil.copytree(spec.source_dir, paths.workspace, symlinks=True)
    shutil.copytree(spec.agent_dir, paths.agent_dir)
    for item in spec.support_dirs:
        shutil.copytree(item.path, paths.support / item.name)
    (paths.workspace / "data").mkdir(parents=True, exist_ok=True)
    (paths.workspace / "home").mkdir(parents=True, exist_ok=True)
    (paths.workspace / "logs").mkdir(parents=True, exist_ok=True)
    (paths.workspace / "submissions").mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init", "-q", "-b", "main"],
        cwd=paths.workspace,
        check=True,
        text=True,
    )


def contract(spec: Spec, paths: Paths) -> tuple[Var, ...]:
    """Build the fixed container environment contract."""

    base = [
        Var("HOME", "/workspace/home"),
        Var("BENCHMARK_WORKSPACE", "/workspace"),
        Var("BENCHMARK_RUN_ID", paths.run_id),
        Var("BENCHMARK_AGENT", spec.agent_name),
        Var("BENCHMARK_AGENT_DIR", "/runner/agent"),
        Var("BENCHMARK_TIMEOUT_SEC", str(spec.timeout_sec)),
        Var("BENCHMARK_TRACE_DIR", "/workspace/home"),
        Var("BENCHMARK_LOG_DIR", "/workspace/logs"),
        Var("BENCHMARK_EXPERIMENT_DIR", "/workspace/experiments"),
        Var("BENCHMARK_SUBMISSION_DIR", "/workspace/submissions"),
        Var("BENCHMARK_EVENTS_PATH", "/workspace/agent_events.jsonl"),
        Var("BENCHMARK_FINAL_PATH", "/workspace/agent_final.txt"),
    ]
    base.extend(spec.explicit)
    return tuple(base)


def fail(command: list[str], completed: subprocess.CompletedProcess[str]) -> None:
    """Exit with a readable subprocess failure message."""

    parts = [
        f"command failed with exit code {completed.returncode}",
        shlex.join(command),
    ]
    stdout = completed.stdout.rstrip()
    stderr = completed.stderr.rstrip()
    if stdout:
        parts.append(f"stdout:\n{stdout}")
    if stderr:
        parts.append(f"stderr:\n{stderr}")
    raise SystemExit("\n\n".join(parts))


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a command and surface captured output on failure."""

    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    if completed.returncode != 0:
        fail(command, completed)
    return completed


def create(spec: Spec, paths: Paths) -> tuple[str, tuple[str, ...]]:
    """Create the docker container and return its name and create command."""

    container_name = f"nanogpt-bench-{slug(paths.run_id)}"
    env_vars = contract(spec, paths)
    command: list[str] = [
        "docker",
        "create",
        "--name",
        container_name,
        "--init",
        "--ipc=host",
        "--shm-size=16g",
        "--cap-add=SYS_PTRACE",
        "--security-opt",
        "seccomp=unconfined",
        "--workdir",
        "/workspace",
        "-v",
        f"{paths.workspace}:/workspace",
        "-v",
        f"{paths.support}:/runner",
        "-v",
        f"{spec.data_volume}:/workspace/data",
    ]
    if spec.gpus:
        command.extend(["--gpus", spec.gpus])
    for name in spec.passthrough:
        if name in os.environ:
            command.extend(["-e", name])
    for item in env_vars:
        command.extend(["-e", f"{item.name}={item.value}"])
    command.extend([spec.image, "bash", "/runner/agent/run.sh", *spec.agent_args])
    run(command)
    return container_name, tuple(command)


def start(container_name: str) -> None:
    """Start the named docker container."""

    run(["docker", "start", container_name])


def stream(container_name: str, path: Path) -> Relay:
    """Stream container logs to the terminal and append them to disk."""

    source = subprocess.Popen(
        ["docker", "logs", "-f", container_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert source.stdout is not None
    sink = subprocess.Popen(["tee", "-a", str(path)], stdin=source.stdout)
    source.stdout.close()
    return Relay(source=source, sink=sink)


def drain(relay: Relay) -> None:
    """Wait for the live log relay to finish draining."""

    relay.source.wait()
    relay.sink.wait()


def inspect(container_name: str) -> Container:
    """Return the current Docker state for the named container."""

    completed = subprocess.run(
        [
            "docker",
            "inspect",
            "-f",
            "{{.State.Running}} {{.State.ExitCode}}",
            container_name,
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        return Container(exists=False, running=False, exit_code=None)
    running_text, exit_code_text = completed.stdout.strip().split()
    return Container(
        exists=True,
        running=running_text == "true",
        exit_code=int(exit_code_text),
    )


def wait(container_name: str, timeout_sec: int) -> tuple[int | None, bool]:
    """Wait for the container to finish or time out."""

    deadline = time.monotonic() + timeout_sec
    while True:
        container = inspect(container_name)
        assert container.exists, container_name
        if not container.running:
            return container.exit_code, False
        if time.monotonic() >= deadline:
            return None, True
        time.sleep(POLL_SEC)


def interrupt(container_name: str) -> bool:
    """Forward SIGINT to the container and wait for it to stop."""

    completed = subprocess.run(
        ["docker", "kill", "--signal", "INT", container_name],
        check=False,
        text=True,
        capture_output=True,
    )
    assert completed.returncode in {0, 1}, completed.stderr
    deadline = time.monotonic() + INTERRUPT_GRACE_SEC
    while True:
        container = inspect(container_name)
        if not container.exists or not container.running:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(POLL_SEC)


def kill(container_name: str) -> None:
    """Force-stop the named docker container."""

    completed = subprocess.run(
        ["docker", "kill", container_name],
        check=False,
        text=True,
        capture_output=True,
    )
    assert completed.returncode in {0, 1}, completed.stderr
    while True:
        container = inspect(container_name)
        if not container.exists or not container.running:
            return
        time.sleep(POLL_SEC)


def collect(container_name: str) -> Capture:
    """Collect logs and inspect data from the container before cleanup."""

    inspect = run(["docker", "inspect", container_name])
    logs = subprocess.run(
        ["docker", "logs", container_name],
        check=False,
        text=True,
        capture_output=True,
    )
    return Capture(
        inspect_json=inspect.stdout,
        container_log=logs.stdout + logs.stderr,
    )


def write_capture(capture: Capture, paths: Paths) -> None:
    """Write the final container artifacts into the run directory."""

    paths.inspect_json.write_text(capture.inspect_json, encoding="utf-8")
    paths.container_log.write_text(capture.container_log, encoding="utf-8")


def remove(container_name: str) -> None:
    """Remove the container once its artifacts are collected."""

    run(["docker", "rm", container_name])


def meta(
    spec: Spec,
    paths: Paths,
    result: Result,
    create_cmd: tuple[str, ...],
) -> Meta:
    """Build the serialized metadata payload."""

    artifacts = Artifact(
        run_dir=str(paths.run_dir),
        workspace=str(paths.workspace),
        support=str(paths.support),
        agent_dir=str(paths.agent_dir),
        metadata=str(paths.metadata),
        container_log=str(paths.container_log),
        inspect_json=str(paths.inspect_json),
    )
    support_dirs = tuple(
        SupportArtifact(
            name=item.name,
            source_dir=str(item.path),
            copied_dir=str(paths.support / item.name),
        )
        for item in spec.support_dirs
    )
    return Meta(
        run_id=paths.run_id,
        agent_name=spec.agent_name,
        agent_args=spec.agent_args,
        support_dirs=support_dirs,
        image=spec.image,
        source_dir=str(spec.source_dir),
        data_volume=spec.data_volume,
        timeout_sec=spec.timeout_sec,
        gpus=spec.gpus,
        passthrough=spec.passthrough,
        explicit=spec.explicit,
        docker_create=create_cmd,
        artifacts=artifacts,
        result=result,
    )


def write(meta: Meta, path: Path) -> None:
    """Write metadata as JSON."""

    path.write_text(json.dumps(asdict(meta), indent=2), encoding="utf-8")


def trap(state: State) -> None:
    """Register signal handlers that exit through shared cleanup."""

    def handle(signum: int, _frame: FrameType | None) -> None:
        state.interrupted = 128 + signum
        raise SystemExit(state.interrupted)

    signal.signal(signal.SIGINT, handle)
    signal.signal(signal.SIGTERM, handle)


def finish(state: State) -> None:
    """Collect artifacts, clean up the container, and write metadata once."""

    if state.finalized:
        return
    state.finalized = True
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    clean_exit = True
    capture: Capture | None = None
    if state.container_name is not None:
        container = inspect(state.container_name)
        if state.interrupted is not None and container.exists and container.running:
            clean_exit = interrupt(state.container_name)
            container = inspect(state.container_name)
        if (state.timed_out or not clean_exit) and container.exists and container.running:
            kill(state.container_name)
            container = inspect(state.container_name)
        if state.exit_code is None and container.exists and not container.running:
            state.exit_code = container.exit_code
    if state.log_stream is not None:
        drain(state.log_stream)
    if state.container_name is not None:
        container = inspect(state.container_name)
        if state.exit_code is None and container.exists and not container.running:
            state.exit_code = container.exit_code
        if container.exists:
            capture = collect(state.container_name)
            remove(state.container_name)
    if capture is not None:
        write_capture(capture, state.paths)
    if state.started is None or state.started_at is None:
        return
    if state.elapsed_sec is None or state.ended_at is None:
        ended = time.perf_counter()
        state.ended_at = datetime.now(timezone.utc).isoformat()
        state.elapsed_sec = ended - state.started
    assert state.container_name is not None
    assert state.ended_at is not None
    assert state.elapsed_sec is not None
    write(
        meta(
            state.spec,
            state.paths,
            Result(
                container_name=state.container_name,
                started_at=state.started_at,
                ended_at=state.ended_at,
                elapsed_sec=state.elapsed_sec,
                exit_code=state.exit_code,
                timed_out=state.timed_out,
            ),
            state.create_cmd,
        ),
        state.paths.metadata,
    )


def main() -> None:
    """Run the selected benchmark agent."""

    args = parse()
    resolved = spec(args)
    paths = layout(resolved)
    state = State(spec=resolved, paths=paths)
    trap(state)
    atexit.register(finish, state)
    stage(resolved, paths)
    state.container_name, state.create_cmd = create(resolved, paths)
    start(state.container_name)
    state.started_at = datetime.now(timezone.utc).isoformat()
    state.started = time.perf_counter()
    state.log_stream = stream(state.container_name, paths.container_log)
    state.exit_code, state.timed_out = wait(state.container_name, resolved.timeout_sec)
    ended = time.perf_counter()
    state.ended_at = datetime.now(timezone.utc).isoformat()
    assert state.started is not None
    state.elapsed_sec = ended - state.started
    finish(state)
    print(paths.run_dir)


if __name__ == "__main__":
    main()
