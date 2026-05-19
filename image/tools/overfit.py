#!/usr/bin/env python3
"""Codebase serialization and Codex configuration shared by `submit` and `comparable`."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

DEFAULT_MODEL: Final = "gpt-5.4"
DEFAULT_REASONING: Final = "xhigh"
DEFAULT_AZURE_VERSION: Final = "2025-04-01-preview"

CODE_SUFFIXES: Final = frozenset({".py", ".sh", ".cpp", ".toml", ".yaml", ".cu"})


@dataclass(frozen=True)
class Var:
    """One environment variable assignment for the Codex subprocess."""

    name: str
    value: str


@dataclass(frozen=True)
class Codex:
    """Resolved Codex configuration and environment."""

    config: str
    env: tuple[Var, ...]


@dataclass(frozen=True)
class Codebase:
    """One serialized canonical codebase."""

    name: str
    root: Path
    text: str


@dataclass(frozen=True)
class Candidate:
    """One resolved candidate codebase."""

    name: str
    source: Path
    root: Path
    mode: str
    text: str


def path_arg(raw: str) -> Path:
    """Resolve and validate a directory argument."""

    path = Path(raw).expanduser().resolve()
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"directory does not exist: {path}")
    return path


def runnable(path: Path) -> bool:
    """Return whether a directory matches the runnable record layout."""

    return (path / "run.sh").is_file() and (path / "train_gpt.py").is_file()


def code_arg(raw: str) -> Path:
    """Resolve and validate a runnable code directory."""

    path = path_arg(raw)
    if not runnable(path):
        raise argparse.ArgumentTypeError(f"directory is not a runnable codebase: {path}")
    return path


def hidden(path: Path) -> bool:
    """Return whether a relative path should be skipped during serialization."""

    return any(part.startswith(".") for part in path.parts) or "__pycache__" in path.parts


def files(root: Path) -> tuple[Path, ...]:
    """Return the regular files that belong to a serialized codebase."""

    items: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if hidden(path.relative_to(root)):
            continue
        if path.suffix.lower() not in CODE_SUFFIXES:
            continue
        items.append(path)
    return tuple(items)


def serialize(root: Path) -> str:
    """Serialize a code directory into a deterministic text block."""

    entries: list[str] = []
    for path in files(root):
        rel = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8")
        entries.append(f"=== FILE {rel} ===\n{text}")
    if not entries:
        raise ValueError(f"no readable files found under {root}")
    return "\n\n".join(entries)


def workspaces(root: Path) -> tuple[Path, ...]:
    """Return the staged workspaces for a batch record folder."""

    items: list[Path] = []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        if not path.name.startswith("rep_"):
            continue
        workspace = path / "workspace"
        if workspace.is_dir():
            items.append(workspace)
    return tuple(items)


def resolve(root: Path) -> Candidate | None:
    """Resolve one immediate child into a candidate codebase."""

    batch = workspaces(root)
    if batch:
        texts = tuple(serialize(path) for path in batch)
        first = texts[0]
        for path, text in zip(batch[1:], texts[1:]):
            if text != first:
                raise ValueError(f"non-identical workspaces under {root}: {path}")
        return Candidate(
            name=root.name,
            source=root,
            root=batch[0],
            mode="batch",
            text=first,
        )
    if runnable(root):
        return Candidate(
            name=root.name,
            source=root,
            root=root,
            mode="direct",
            text=serialize(root),
        )
    return None


def children(root: Path) -> tuple[Candidate, ...]:
    """Resolve every comparable immediate child under the target root."""

    items: list[Candidate] = []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        if path.name.startswith("."):
            continue
        candidate = resolve(path)
        if candidate is not None:
            items.append(candidate)
    if not items:
        raise RuntimeError(f"no comparable candidate codebases found under {root}")
    return tuple(items)


def load(root: Path) -> Codebase:
    """Load and serialize one runnable code directory."""

    return Codebase(name=root.name, root=root, text=serialize(root))


def pick(*names: str) -> str:
    """Return the first non-empty environment variable value."""

    for name in names:
        value = (os.environ.get(name) or "").strip()
        if value:
            return value
    return ""


def openai(model: str, reasoning: str) -> Codex | None:
    """Resolve direct OpenAI-compatible Codex configuration."""

    openai_key = pick("OPENAI_API_KEY")
    codex_key = pick("CODEX_API_KEY")
    if not openai_key and not codex_key:
        return None
    config = ""
    if model:
        config += f'model = "{model}"\n'
    config += f'model_reasoning_effort = "{reasoning}"\n'
    env: list[Var] = []
    if openai_key:
        env.append(Var(name="OPENAI_API_KEY", value=openai_key))
    if codex_key:
        env.append(Var(name="CODEX_API_KEY", value=codex_key))
    return Codex(config=config, env=tuple(env))


def azure(reasoning: str) -> Codex | None:
    """Resolve Azure-backed Codex configuration."""

    azure_key = pick("AZURE_API_KEY", "AZURE_OPENAI_API_KEY")
    if not azure_key:
        return None
    azure_base = pick("AZURE_OPENAI_BASE_URL", "AZURE_API_BASE", "OPENAI_BASE_URL")
    if not azure_base:
        raise ValueError("Set AZURE_API_BASE or AZURE_OPENAI_BASE_URL for Azure Codex access.")
    azure_deployment = pick("AZURE_OPENAI_DEPLOYMENT")
    if not azure_deployment:
        raise ValueError("Set AZURE_OPENAI_DEPLOYMENT for Azure Codex access.")
    azure_version = pick("AZURE_API_VERSION") or DEFAULT_AZURE_VERSION
    root = azure_base.rstrip("/")
    if not root.endswith("/openai"):
        root = f"{root}/openai"
    env = (
        Var(name="AZURE_API_KEY", value=azure_key),
        Var(name="AZURE_OPENAI_API_KEY", value=azure_key),
    )
    return Codex(
        config=(
            f'model = "{azure_deployment}"\n'
            f'model_reasoning_effort = "{reasoning}"\n'
            'model_provider = "azure"\n'
            '[model_providers.azure]\n'
            'name = "Azure"\n'
            f'base_url = "{root}"\n'
            'env_key = "AZURE_API_KEY"\n'
            f'query_params = {{ api-version = "{azure_version}" }}\n'
            'wire_api = "responses"\n'
        ),
        env=env,
    )


def codex(model: str, reasoning: str) -> Codex:
    """Resolve the Codex configuration required for one comparison run."""

    model_name = model.strip()
    if not model_name:
        raise ValueError("model must not be empty")
    reasoning_name = reasoning.strip()
    if not reasoning_name:
        raise ValueError("reasoning must not be empty")
    direct = openai(model_name, reasoning_name)
    if direct is not None:
        return direct
    cloud = azure(reasoning_name)
    if cloud is not None:
        return cloud
    raise ValueError(
        "Set OPENAI_API_KEY, CODEX_API_KEY, or the Azure Codex environment variables before running this tool."
    )
