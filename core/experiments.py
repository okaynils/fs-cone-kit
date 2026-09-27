"""Durable, local experiment artifacts shared by training and evaluation."""

from __future__ import annotations

import importlib.metadata
import json
import math
import platform
import re
import subprocess
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import yaml


SCHEMA_VERSION = 1
_SENSITIVE_KEY = re.compile(r"(?:password|passwd|token|secret|api[_-]?key|credential)", re.I)


def _redact_uri(value: str) -> str:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return value
    if not parsed.scheme:
        return value
    netloc = parsed.netloc
    changed = False
    if parsed.username is not None:
        host = parsed.hostname or ""
        if parsed.port is not None:
            host = f"{host}:{parsed.port}"
        netloc = f"<redacted>@{host}"
        changed = True
    query = []
    for key, query_value in parse_qsl(parsed.query, keep_blank_values=True):
        if _SENSITIVE_KEY.search(key):
            query.append((key, "<redacted>"))
            changed = True
        else:
            query.append((key, query_value))
    if not changed:
        return value
    return urlunsplit((parsed.scheme, netloc, parsed.path, urlencode(query), parsed.fragment))


def redact_sensitive(value: Any, key: str = "") -> Any:
    """Redact credentials while retaining the shape of a resolved config."""
    if _SENSITIVE_KEY.search(key):
        return "<redacted>"
    if isinstance(value, dict):
        return {str(k): redact_sensitive(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_sensitive(item, key) for item in value]
    if isinstance(value, str):
        return _redact_uri(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    def normalize(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): normalize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(normalize(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _git_value(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], capture_output=True, check=True, text=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def _dependency_versions(project_root: Path) -> dict[str, str]:
    pyproject = project_root / "pyproject.toml"
    if not pyproject.exists():
        return {}
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    requirements = data.get("project", {}).get("dependencies", [])
    versions: dict[str, str] = {}
    for requirement in requirements:
        name = re.split(r"[<>=!~;\[ ]", requirement, maxsplit=1)[0]
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return dict(sorted(versions.items()))


def _environment_versions() -> dict[str, str]:
    versions = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            versions[name] = distribution.version
    return dict(sorted(versions.items(), key=lambda item: item[0].lower()))


def collect_run_metadata(project_root: Path, seed: int | None) -> dict[str, Any]:
    status = _git_value("status", "--porcelain")
    safe_command = []
    redact_next = False
    for argument in sys.argv:
        key, separator, _ = argument.partition("=")
        if redact_next or _SENSITIVE_KEY.search(key):
            safe_command.append(f"{key}=<redacted>" if separator else "<redacted>")
            redact_next = not separator and argument.startswith("--")
        else:
            safe_command.append(_redact_uri(argument))
            redact_next = False
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": safe_command,
        "seed": seed,
        "git": {
            "commit": _git_value("rev-parse", "HEAD"),
            "dirty": bool(status) if status is not None else None,
        },
        "dependencies": _dependency_versions(project_root),
        "environment": _environment_versions(),
        "platform": {
            "python": platform.python_version(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
    }


class ExperimentArtifacts:
    """Writes the stable files that make one experiment inspectable and repeatable."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.record_dir = self.root / "experiment"

    def start(
        self,
        resolved_config: dict[str, Any],
        dataset_info: dict[str, Any],
        project_root: Path,
    ) -> None:
        self.record_dir.mkdir(parents=True, exist_ok=True)
        safe_config = redact_sensitive(resolved_config)
        (self.record_dir / "config.yaml").write_text(
            yaml.safe_dump(safe_config, sort_keys=False), encoding="utf-8"
        )
        write_json(self.record_dir / "metadata.json", collect_run_metadata(project_root, safe_config.get("seed")))
        write_json(self.record_dir / "dataset.json", dataset_info)
        write_json(self.record_dir / "splits.json", dataset_info["manifest"])

    def finish_training(self, summary: dict[str, Any]) -> None:
        write_json(self.record_dir / "training.json", {"schema_version": SCHEMA_VERSION, **summary})


def relative_to_experiment(path: str | Path | None, experiment_root: Path) -> str | None:
    if path is None:
        return None
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(experiment_root.resolve()))
    except ValueError:
        return str(resolved)
