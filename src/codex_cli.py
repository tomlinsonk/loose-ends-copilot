#!/usr/bin/env python3
"""Shared non-interactive agent CLI execution helpers.

Two agent backends are supported: the OpenAI Codex CLI and the GitHub Copilot
CLI.  A process selects one with ``select_backend``; entry points expose this
as ``--backend``.  The library default remains Codex for compatibility.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import re
import signal
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
from typing import Callable, Mapping
import uuid

from validation import common as validation_common
from codex_transcripts import record_transcript


REASONING_EFFORTS = ("low", "medium", "high", "xhigh", "max", "ultra")
WEB_SEARCH_MODES = ("disabled", "indexed", "live")
BACKENDS = ("copilot", "codex")
DEFAULT_BACKEND = "copilot"
BACKEND_NAMES = {"copilot": "GitHub Copilot CLI", "codex": "Codex CLI"}
COPILOT_UNSUPPORTED_REASONING_EFFORTS = frozenset({"ultra"})
# Built-in Copilot tools that delegate to other agents, reach outside the
# per-run workspace, or load user-installed extensions.
COPILOT_EXCLUDED_TOOLS = (
    "task",
    "read_agent",
    "write_agent",
    "list_agents",
    "run_dynamic_workflow",
    "dynamic_workflows_manage",
    "session_store_sql",
    "skill",
    "fetch_copilot_cli_documentation",
    "semantic_code_search",
    "web_fetch",
)
# Live web search adds this unsandboxed MCP server so agents can read pages
# while shell commands stay offline; it refuses local and private addresses.
COPILOT_FETCH_SERVER = "loose-ends-fetch"
COPILOT_FETCH_SERVER_PATH = Path(__file__).resolve().with_name("copilot_fetch_mcp.py")
COPILOT_HOME_ENV = "COPILOT_HOME"
# Copilot may authenticate with these, so they stay in its own environment;
# --secret-env-vars strips them and other credentials from agent shells.
COPILOT_AUTH_ENV_VARS = ("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN")
COPILOT_SECRET_ENV_PATTERN = re.compile(
    r"TOKEN|SECRET|PASSWORD|PASSWD|PASSPHRASE|CREDENTIAL|API_?KEY|ACCESS_?KEY"
    r"|PRIVATE_?KEY|CONNECTION_?STRING|^COPILOT_GH_ACCOUNT_",
    re.IGNORECASE,
)
COPILOT_ACCOUNT_CONFIG_KEYS = ("lastLoggedInUser", "loggedInUsers")
COPILOT_READONLY_PATHS_ENV = "LOOSE_ENDS_COPILOT_READONLY_PATHS"
COPILOT_USAGE_EVENT = "loose_ends.copilot_usage"
# Copilot reports its session ID only at exit; this host-written event names
# the invocation from launch so usage accounting can track in-progress turns.
COPILOT_STARTED_EVENT = "loose_ends.copilot_started"
COPILOT_USAGE_FILENAME = "usage.json"
DEFAULT_MODEL = "gpt-6-astra"
DEFAULT_REASONING_EFFORT = "xhigh"
DEFAULT_VALIDATION_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent / "prompts" / "validate-output.md"
)
CODEX_LAUNCH_INTERVAL_SECONDS = 1.0
MAX_CODEX_START_ATTEMPTS = 3
CODEX_POLL_INTERVAL_SECONDS = 0.5
CODEX_STOP_GRACE_SECONDS = 10.0
WINDOWS_SANDBOX_ACL_FAILURE = "helper_unknown_error: apply deny-read acls"
WORKBENCH_DATABASE_ENV = "LOOSE_ENDS_WORKBENCH_DATABASE"
CODEX_ACCOUNT_CONFIG_KEYS = (
    "cli_auth_credentials_store",
    "forced_login_method",
    "forced_chatgpt_workspace_id",
    "sqlite_home",
    "model_provider",
    "openai_base_url",
    "chatgpt_base_url",
)
_CODEX_LAUNCH_LOCK = threading.Lock()
_WINDOWS_SANDBOX_PROBE_LOCK = threading.Lock()
_WINDOWS_ACL_NORMALIZE_LOCK = threading.Lock()
_next_codex_launch_at = 0.0
_windows_sandbox_probe_error: str | None = None
_windows_sandbox_probe_succeeded = False
_WINDOWS_RESERVED_DEVICE_NAMES = {
    "aux",
    "con",
    "nul",
    "prn",
    *(f"com{number}" for number in range(1, 10)),
    *(f"lpt{number}" for number in range(1, 10)),
}


class CodexError(RuntimeError):
    """A local agent invocation or its workspace could not be used."""


_active_backend = "codex"


def select_backend(backend: str) -> None:
    """Choose the agent CLI used by subsequent runs in this process."""
    global _active_backend
    if backend not in BACKENDS:
        raise CodexError("backend must be one of " + ", ".join(BACKENDS))
    _active_backend = backend


def active_backend() -> str:
    return _active_backend


def using_copilot() -> bool:
    return _active_backend == "copilot"


def backend_name() -> str:
    return BACKEND_NAMES[_active_backend]


def add_backend_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the agent-backend selector and the Copilot executable option."""
    parser.add_argument(
        "--backend",
        choices=BACKENDS,
        default=DEFAULT_BACKEND,
        help=f"agent CLI to run (default: {DEFAULT_BACKEND})",
    )
    parser.add_argument(
        "--copilot",
        default="copilot",
        help="GitHub Copilot CLI executable or command name (default: copilot)",
    )


def validate_backend_options(args: argparse.Namespace) -> None:
    """Reject Codex-only model options when the Copilot backend is selected."""
    if getattr(args, "backend", "codex") != "copilot":
        return
    for name, value in sorted(vars(args).items()):
        option = "--" + name.replace("_", "-")
        if name.endswith(("fast", "ultrafast")) and value is True:
            raise CodexError(
                f"{option} is a Codex service tier; the Copilot backend does not support it"
            )
        if (
            name.endswith("reasoning_effort")
            and value in COPILOT_UNSUPPORTED_REASONING_EFFORTS
        ):
            raise CodexError(
                f"{option} {value} is not supported by the Copilot backend"
            )


def agent_executable(args: argparse.Namespace) -> str:
    """Select the requested backend and resolve its CLI executable."""
    backend = getattr(args, "backend", "codex")
    select_backend(backend)
    validate_backend_options(args)
    command = args.copilot if backend == "copilot" else args.codex
    return resolve_codex_executable(command)


def backend_arguments(args: argparse.Namespace) -> list[str]:
    """Forward backend selection to a nested entry point."""
    backend = getattr(args, "backend", "codex")
    arguments = ["--backend", backend, "--codex", args.codex]
    if hasattr(args, "copilot"):
        arguments.extend(("--copilot", args.copilot))
    return arguments


def agent_credit_error(events_path: Path, log_path: Path | None = None) -> str | None:
    if using_copilot():
        return copilot_credit_error(events_path, log_path)
    return codex_credit_error(events_path)


COPILOT_CREDIT_PHRASES = (
    "quota exceeded",
    "out of credits",
    "insufficient credits",
    "usage limit",
    "premium request limit",
    "no remaining premium requests",
    "ai credit limit",
    "max ai credits",
)


def _copilot_credit_message(text: str) -> bool:
    folded = text.casefold()
    return any(phrase in folded for phrase in COPILOT_CREDIT_PHRASES)


def copilot_credit_error(events_path: Path, log_path: Path | None = None) -> str | None:
    """Recognize Copilot quota or credit exhaustion in error output."""
    try:
        with events_path.open(encoding="utf-8", errors="replace") as events:
            for line in events:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(event, dict):
                    continue
                kind = str(event.get("type", ""))
                if "error" not in kind and "fail" not in kind:
                    continue
                data = event.get("data")
                fields = [event.get("message"), event.get("error")]
                if isinstance(data, dict):
                    fields.extend(
                        (data.get("message"), data.get("error"), data.get("errorType"))
                    )
                for field in fields:
                    if isinstance(field, dict):
                        field = field.get("message")
                    if isinstance(field, str) and _copilot_credit_message(field):
                        return field
    except OSError:
        pass
    if log_path is not None:
        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            lines = []
        for line in lines:
            if "error" in line.casefold() and _copilot_credit_message(line):
                return line.strip()
    return None


def codex_credit_error(events_path: Path) -> str | None:
    """Recognize account exhaustion only in Codex error events."""
    with events_path.open(encoding="utf-8", errors="replace") as events:
        for line in events:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            if event.get("type") == "error":
                error = event
            elif event.get("type") == "turn.failed":
                error = event.get("error")
            else:
                continue
            if not isinstance(error, dict):
                continue
            message = str(error.get("message", ""))
            if error.get("code") in {"usage_limit_reached", "insufficient_quota"} or any(
                phrase in message.casefold()
                for phrase in ("hit your usage limit", "out of credits", "insufficient credits")
            ):
                return message or str(error["code"])
    return None


def pause_queue_for_credit_error(message: str) -> None:
    database = os.environ.get(WORKBENCH_DATABASE_ENV)
    if database:
        from workbench_store import WorkbenchStore

        WorkbenchStore(Path(database)).pause_for_codex_credits(message)


@dataclass(frozen=True)
class ModelOptions:
    model: str | None = None
    reasoning_effort: str | None = None
    fast: bool = False
    ultrafast: bool = False

    def __post_init__(self):
        if self.fast and self.ultrafast:
            raise ValueError("Fast and Ultrafast are mutually exclusive")

    @property
    def service_tier(self) -> str:
        return "ultrafast" if self.ultrafast else "fast" if self.fast else "standard"


@dataclass(frozen=True)
class OutputValidator:
    """One task-specific validator and its authoritative dynamic inputs."""

    source: Path
    validate: Callable[..., validation_common.ValidationReport]
    expectations: Mapping[str, object]


def validated_result(
    report: validation_common.ValidationReport,
) -> dict[str, object]:
    """Return a checked result without relying on assertions at call sites."""
    if not report.valid or report.result is None:
        raise CodexError("validator accepted no structured result")
    return report.result


def configure_utf8_stdio() -> None:
    """Keep research titles and author names printable on Windows consoles."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="backslashreplace")


def windowless_popen_options(
    *,
    new_process_group: bool = True,
) -> dict[str, object]:
    """Return isolated subprocess options without a visible Windows console."""
    if os.name != "nt":
        return {"start_new_session": True} if new_process_group else {}
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = subprocess.SW_HIDE
    creationflags = subprocess.CREATE_NO_WINDOW
    if new_process_group:
        creationflags |= subprocess.CREATE_NEW_PROCESS_GROUP
    return {
        "creationflags": creationflags,
        "startupinfo": startupinfo,
    }


def report_error(parser: argparse.ArgumentParser, error: BaseException) -> int:
    """Report a post-parse failure without printing irrelevant CLI usage."""
    print(f"{parser.prog}: error: {error}", file=sys.stderr)
    return 1


def positive_integer(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def positive_number(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return number


def add_prompt_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_template: Path,
    task: str,
    prefix: str = "",
) -> None:
    """Add consistent user-direction and low-level prompt-template flags."""
    option_prefix = f"{prefix}-" if prefix else ""
    destination_prefix = f"{prefix.replace('-', '_')}_" if prefix else ""
    parser.add_argument(
        f"--{option_prefix}prompt",
        dest=f"{destination_prefix}prompt",
        metavar="TEXT",
        help=f"additional instruction for the {task}",
    )
    parser.add_argument(
        f"--{option_prefix}prompt-template",
        dest=f"{destination_prefix}prompt_template",
        type=Path,
        default=default_template,
        metavar="FILE",
        help=(
            f"replace the complete low-level {task} prompt template "
            f"(default: {default_template})"
        ),
    )


def with_user_prompt(
    template: str,
    instruction: str | None,
    *,
    task: str,
    option_name: str = "--prompt",
) -> str:
    """Append an explicit user direction without replacing core safeguards."""
    if instruction is None:
        return template
    instruction = instruction.strip()
    if not instruction:
        raise CodexError(f"{option_name} must be nonempty")
    return (
        template.rstrip()
        + "\n\n# Additional user direction\n\n"
        + f"The user explicitly requested this direction for the {task}:\n\n"
        + f"<user_instruction>\n{instruction}\n</user_instruction>\n\n"
        + "Follow it throughout this run while preserving the task's output "
        + "contract, validation requirements, and mathematical accuracy.\n"
    )


def add_model_arguments(
    parser: argparse.ArgumentParser,
    *,
    prefix: str = "",
    default_reasoning_effort: str | None = None,
) -> None:
    """Add consistent model, reasoning, Fast mode, and executable options."""
    if (
        default_reasoning_effort is not None
        and default_reasoning_effort not in REASONING_EFFORTS
    ):
        raise ValueError(f"invalid default reasoning effort: {default_reasoning_effort}")
    option_prefix = f"{prefix}-" if prefix else ""
    destination_prefix = f"{prefix.replace('-', '_')}_" if prefix else ""
    model_default = None if prefix else DEFAULT_MODEL
    primary_reasoning_default = default_reasoning_effort or DEFAULT_REASONING_EFFORT
    reasoning_default = None if prefix else primary_reasoning_default
    model_default_help = (
        "inherit the primary run" if prefix else DEFAULT_MODEL
    )
    reasoning_default_help = (
        "inherit the primary run" if prefix else primary_reasoning_default
    )
    parser.add_argument(
        f"--{option_prefix}model",
        dest=f"{destination_prefix}model",
        default=model_default,
        metavar="MODEL",
        help=(
            "Codex model ID; for example, gpt-5.6-sol "
            f"(default: {model_default_help})"
        ),
    )
    parser.add_argument(
        f"--{option_prefix}reasoning-effort",
        dest=f"{destination_prefix}reasoning_effort",
        default=reasoning_default,
        choices=REASONING_EFFORTS,
        metavar="LEVEL",
        help=(
            "reasoning depth: low, medium, high, xhigh, max, or ultra; "
            f"extra-high is xhigh (default: {reasoning_default_help})"
        ),
    )
    speed = parser.add_mutually_exclusive_group()
    speed.add_argument(
        f"--{option_prefix}fast",
        dest=f"{destination_prefix}fast",
        action="store_true",
        help=(
            "request Codex Fast mode for this run; this uses more credits "
            + (
                "(default: inherit the primary run)"
                if prefix
                else "(default: use the CLI configuration)"
            )
        ),
    )
    speed.add_argument(
        f"--{option_prefix}ultrafast",
        dest=f"{destination_prefix}ultrafast",
        action="store_true",
        help="request Codex Ultrafast mode (higher cost; requires model/account support)",
    )


def add_web_search_argument(
    parser: argparse.ArgumentParser,
    *,
    default: str,
    prefix: str = "",
) -> None:
    """Add a scoped Codex first-party web-search option."""
    if default not in WEB_SEARCH_MODES:
        raise ValueError(f"invalid default web-search mode: {default}")
    option_prefix = f"{prefix}-" if prefix else ""
    destination_prefix = f"{prefix.replace('-', '_')}_" if prefix else ""
    parser.add_argument(
        f"--{option_prefix}web-search",
        dest=f"{destination_prefix}web_search",
        choices=WEB_SEARCH_MODES,
        default=None if prefix else default,
        metavar="MODE",
        help=(
            "Codex first-party web search: disabled, indexed, or live "
            f"(default: {'inherit the primary run' if prefix else default}); "
            "this does not enable shell network access or MCP/plugin apps"
        ),
    )


def model_options_from_args(
    args: argparse.Namespace,
    *,
    prefix: str = "",
) -> ModelOptions:
    destination_prefix = f"{prefix.replace('-', '_')}_" if prefix else ""
    return ModelOptions(
        model=getattr(args, f"{destination_prefix}model"),
        reasoning_effort=getattr(
            args,
            f"{destination_prefix}reasoning_effort",
        ),
        fast=getattr(args, f"{destination_prefix}fast"),
        ultrafast=getattr(args, f"{destination_prefix}ultrafast", False),
    )


def semantic_config_digest(
    prompt: str,
    schema_text: str,
    options: ModelOptions,
    *,
    web_search: str = "disabled",
    validation_source: Path | None = None,
) -> str:
    payload = {
        "fast": options.fast,
        "model": options.model,
        "prompt": prompt,
        "reasoning_effort": options.reasoning_effort,
        "schema": schema_text,
    }
    if options.ultrafast:
        payload["ultrafast"] = True
    # Preserve existing disabled-search digests for analysis and triage.
    if web_search != "disabled":
        payload["web_search"] = web_search
    if validation_source is not None:
        payload["validation"] = validation_code_digest(validation_source)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validation_code_digest(source: Path) -> str:
    digest = hashlib.sha256()
    for path in (
        Path(validation_common.__file__).resolve(),
        source.resolve(),
        DEFAULT_VALIDATION_PROMPT_PATH.resolve(),
    ):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def stage_output_validator(workspace: Path, validator: OutputValidator) -> Path:
    """Stage only the selected checker, shared primitives, and expectations."""
    directory = workspace / "validation"
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        Path(validation_common.__file__).resolve(),
        directory / "common.py",
    )
    shutil.copyfile(validator.source.resolve(), directory / "validate.py")
    (directory / "__init__.py").write_text("", encoding="utf-8")
    (directory / validation_common.EXPECTATIONS_FILENAME).write_text(
        json.dumps(
            validator.expectations,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return directory


def with_validation_instructions(prompt: str) -> str:
    """Append the shared, repository-owned validation prompt fragment."""
    try:
        instructions = DEFAULT_VALIDATION_PROMPT_PATH.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise CodexError(
            f"could not read validation prompt {DEFAULT_VALIDATION_PROMPT_PATH}: {exc}"
        ) from exc
    return prompt.rstrip() + "\n\n" + instructions.strip() + "\n"


def resolve_codex_executable(value: str) -> str:
    executable = shutil.which(value)
    if executable is None:
        raise CodexError(
            f"could not find the {backend_name()} executable {value!r} on PATH"
        )
    return executable


def read_codex_version(codex: str) -> str:
    try:
        completed = subprocess.run(
            [codex, "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise CodexError(f"could not run {codex!r}: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or f"exit status {completed.returncode}"
        raise CodexError(f"could not query the {backend_name()} version: {detail}")
    version = completed.stdout.strip()
    if using_copilot():
        # Copilot appends update hints after the version line.
        version = next(
            (line.strip().rstrip(".") for line in version.splitlines() if line.strip()),
            "",
        )
    return version


def is_windows_host() -> bool:
    return os.name == "nt" or sys.platform == "cygwin"


def codex_subprocess_environment() -> dict[str, str]:
    """Avoid Windows Store command aliases inaccessible to sandbox users."""
    environment = os.environ.copy()
    if not is_windows_host():
        return environment
    path = environment.get("PATH")
    if path is None:
        return environment
    entries = path.split(os.pathsep)
    environment["PATH"] = os.pathsep.join(
        entry
        for entry in entries
        if not entry.rstrip("\\/").replace("\\", "/").casefold().endswith(
            "/microsoft/windowsapps"
        )
    )
    return environment


def _run_local_command(command: list[str], description: str) -> str:
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, stderr = process.communicate()
    except OSError as exc:
        raise CodexError(f"{description}: {exc}") from exc
    if process.returncode != 0:
        detail = stderr.strip() or stdout.strip() or (
            f"exit status {process.returncode}"
        )
        raise CodexError(f"{description}: {detail}")
    return stdout


def path_for_codex(path: Path) -> str:
    """Return a path the Windows-native Codex CLI can understand."""
    resolved = path.resolve()
    if sys.platform != "cygwin":
        return str(resolved)
    try:
        process = subprocess.Popen(
            ["cygpath", "-w", str(resolved)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, stderr = process.communicate()
    except OSError as exc:
        raise CodexError(f"could not run cygpath for {resolved}: {exc}") from exc
    if process.returncode != 0:
        detail = stderr.strip() or f"exit status {process.returncode}"
        raise CodexError(f"could not convert Cygwin path {resolved}: {detail}")
    converted = stdout.strip()
    if not converted:
        raise CodexError(f"cygpath returned an empty path for {resolved}")
    return converted


@lru_cache(maxsize=1)
def windows_identity() -> str:
    domain = os.environ.get("USERDOMAIN", "").strip()
    username = os.environ.get("USERNAME", "").strip()
    if domain and username:
        return f"{domain}\\{username}"
    executable = shutil.which("whoami.exe")
    if executable is None:
        raise CodexError("could not find whoami.exe for Windows ACL setup")
    identity = _run_local_command(
        [executable],
        "could not determine the current Windows identity",
    ).strip()
    if not identity or "\\" not in identity:
        raise CodexError(
            f"whoami.exe returned an unexpected identity: {identity!r}"
        )
    return identity


@lru_cache(maxsize=1)
def windows_icacls() -> str:
    executable = shutil.which("icacls.exe") or shutil.which("icacls")
    if executable is None:
        raise CodexError("could not find icacls.exe for Windows ACL setup")
    return executable


def windows_icacls_for_sandbox() -> str:
    executable = windows_icacls()
    if sys.platform == "cygwin":
        return path_for_codex(Path(executable))
    return executable


def _windows_sandbox_probe_command(codex: str, workspace: Path) -> list[str]:
    windows_directory = os.environ.get("WINDIR", r"C:\Windows")
    executable = str(
        PureWindowsPath(windows_directory) / "System32" / "whoami.exe"
    )
    return [
        codex,
        "sandbox",
        "-c",
        'windows.sandbox="elevated"',
        "-P",
        ":workspace",
        "-C",
        path_for_codex(workspace),
        executable,
    ]


def _windows_sandbox_acl_state_path() -> Path:
    codex_home = os.environ.get("CODEX_HOME")
    if codex_home:
        path = Path(codex_home).expanduser()
    else:
        path = Path.home() / ".codex"
    return path / ".sandbox" / "deny_read_acl_state.json"


def windows_sandbox_acl_state_problem() -> str | None:
    """Describe a corrupt elevated-sandbox ACL state file, if present."""
    if not is_windows_host():
        return None
    path = _windows_sandbox_acl_state_path()
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        # A missing state file is valid: Codex creates it during setup.
        return None
    except OSError as exc:
        return f"could not read Codex sandbox ACL state {path}: {exc}"
    try:
        state = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        if raw and not raw.strip(b"\0 \t\r\n"):
            reason = "the file contains only NUL bytes"
        elif not raw:
            reason = "the file is empty"
        else:
            reason = f"invalid JSON ({exc})"
        return (
            f"Codex sandbox ACL state is corrupt at {path}: {reason}. "
            "Move that file aside and retry; Codex will regenerate it."
        )
    if not isinstance(state, dict) or not isinstance(
        state.get("principals"),
        dict,
    ):
        return (
            f"Codex sandbox ACL state is corrupt at {path}: unexpected "
            "JSON structure. Move that file aside and retry; Codex will "
            "regenerate it."
        )
    return None


def require_secure_windows_sandbox(codex: str, workspace: Path) -> None:
    """Fail before model use when the elevated Windows helper is broken."""
    global _windows_sandbox_probe_error
    global _windows_sandbox_probe_succeeded
    if not is_windows_host() or using_copilot():
        return
    with _WINDOWS_SANDBOX_PROBE_LOCK:
        if _windows_sandbox_probe_error is not None:
            raise CodexError(_windows_sandbox_probe_error)
        if _windows_sandbox_probe_succeeded:
            return
        state_problem = windows_sandbox_acl_state_problem()
        if state_problem is not None:
            _windows_sandbox_probe_error = (
                "secure elevated Windows sandbox preflight failed before "
                f"the agent was started: {state_problem}"
            )
            raise CodexError(_windows_sandbox_probe_error)
        command = _windows_sandbox_probe_command(codex, workspace)
        try:
            completed = subprocess.run(
                command,
                cwd=workspace,
                env=codex_subprocess_environment(),
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            detail = str(exc)
        else:
            if completed.returncode == 0:
                _windows_sandbox_probe_succeeded = True
                return
            detail = (completed.stderr or completed.stdout).strip()
            if not detail:
                detail = f"exit status {completed.returncode}"
        state_problem = windows_sandbox_acl_state_problem()
        if state_problem is not None:
            recovery = state_problem
        else:
            recovery = (
                "Stop other active Codex CLI batches and retry; if none are "
                "active, inspect ~/.codex/.sandbox/setup_error.json and the "
                "current sandbox log before restarting Codex or Windows."
            )
        _windows_sandbox_probe_error = (
            "secure elevated Windows sandbox preflight failed before the "
            f"agent was started: {detail}. {recovery}"
        )
        raise CodexError(_windows_sandbox_probe_error)


def note_windows_sandbox_failure(log_path: Path) -> None:
    """Prevent further model launches after an elevated-helper ACL failure."""
    global _windows_sandbox_probe_error
    global _windows_sandbox_probe_succeeded
    if not is_windows_host() or using_copilot():
        return
    try:
        log = log_path.read_text(encoding="utf-8", errors="replace").lower()
    except OSError:
        return
    if WINDOWS_SANDBOX_ACL_FAILURE not in log:
        return
    state_problem = windows_sandbox_acl_state_problem()
    with _WINDOWS_SANDBOX_PROBE_LOCK:
        _windows_sandbox_probe_succeeded = False
        if state_problem is not None:
            recovery = state_problem
        else:
            recovery = (
                "Stop other Codex CLI batches and inspect "
                "~/.codex/.sandbox/setup_error.json plus the current "
                "sandbox log."
            )
        _windows_sandbox_probe_error = (
            "the elevated Windows sandbox helper failed while applying "
            "deny-read ACLs. Further agents were not launched to avoid "
            f"wasting credits. {recovery}"
        )


def grant_workspace_owner_inheritance(workspace: Path) -> None:
    """Make future sandbox-created children readable by the invoking user."""
    if not is_windows_host() or using_copilot():
        return
    identity = windows_identity()
    _run_local_command(
        [
            windows_icacls(),
            path_for_codex(workspace),
            "/grant",
            f"{identity}:(OI)(CI)(F)",
        ],
        f"could not grant {identity} access to {workspace}",
    )


def grant_sandbox_read_access(path: Path) -> None:
    """Let the restricted Windows Codex sandbox read staged local inputs."""
    if not is_windows_host() or using_copilot():
        return
    domain = windows_identity().split("\\", 1)[0]
    sandbox_group = f"{domain}\\CodexSandboxUsers"
    staged_path = path_for_codex(path)
    _run_local_command(
        [
            windows_icacls(),
            staged_path,
            "/remove:d",
            sandbox_group,
            "/T",
            "/C",
        ],
        f"could not remove inherited sandbox read denials from {path}",
    )
    _run_local_command(
        [
            windows_icacls(),
            staged_path,
            "/grant",
            f"{sandbox_group}:(OI)(CI)(RX)",
            "/T",
            "/C",
        ],
        f"could not grant the Codex sandbox read access to {path}",
    )


def _is_windows_reserved_device_name(name: str) -> bool:
    """Recognize names that Win32 resolves as devices instead of files."""
    basename = name.rstrip(" .").split(".", 1)[0].rstrip(" ").casefold()
    return basename in _WINDOWS_RESERVED_DEVICE_NAMES


def workspace_is_user_accessible(workspace: Path) -> bool:
    """Return whether the invoking user can traverse and read a workspace."""
    try:
        def raise_walk_error(error: OSError) -> None:
            raise error

        windows_host = is_windows_host()
        for root, directories, filenames in os.walk(
            workspace,
            onerror=raise_walk_error,
        ):
            if windows_host:
                # Cygwin can create names such as NUL via shell redirection,
                # but Win32 cannot open them and no pipeline can consume them.
                directories[:] = [
                    name
                    for name in directories
                    if not _is_windows_reserved_device_name(name)
                ]
                filenames = [
                    name
                    for name in filenames
                    if not _is_windows_reserved_device_name(name)
                ]
            root_path = Path(root)
            for name in directories:
                (root_path / name).stat()
            for name in filenames:
                path = root_path / name
                path.stat()
                with path.open("rb") as source:
                    source.read(1)
    except OSError:
        return False
    return True


def normalize_workspace_access(workspace: Path, codex: str) -> None:
    """Remove sandbox-created deny ACLs, then grant recursive user access."""
    if not is_windows_host():
        return
    # Codex's elevated helper maintains shared ACL state under CODEX_HOME.
    # Serialize repair turns so concurrent solver/reviewer completions cannot
    # race while that state is being refreshed.
    with _WINDOWS_ACL_NORMALIZE_LOCK:
        if workspace_is_user_accessible(workspace):
            return
        if using_copilot():
            # Only Codex's elevated sandbox creates these ACLs, so repairing
            # data left by an earlier Codex run still needs the Codex CLI.
            repair_codex = shutil.which("codex")
            if repair_codex is None:
                raise CodexError(
                    f"{workspace} has restrictive ACLs from an earlier Codex "
                    "sandbox run; install the Codex CLI to repair them"
                )
            codex = repair_codex
        identity = windows_identity()
        command_prefix = [
            codex,
            "sandbox",
            "-P",
            ":workspace",
            "-C",
            path_for_codex(workspace),
            windows_icacls_for_sandbox(),
            ".",
        ]
        for action in (
            ["/remove:d", identity, "/T", "/C"],
            ["/grant", f"{identity}:(OI)(CI)(F)", "/T", "/C"],
        ):
            _run_local_command(
                [*command_prefix, *action],
                f"could not normalize sandbox-owned files in {workspace}",
            )
        if not workspace_is_user_accessible(workspace):
            raise CodexError(
                f"Windows ACL repair did not make {workspace} accessible"
            )


def wait_for_codex_launch_slot(interval: float) -> None:
    """Space process startups while allowing already-started runs to overlap."""
    if interval <= 0:
        return
    shared_gate = os.environ.get("LOOSE_ENDS_CODEX_LAUNCH_GATE")
    if shared_gate:
        _wait_for_shared_launch_slot(Path(shared_gate), interval)
        return
    global _next_codex_launch_at
    with _CODEX_LAUNCH_LOCK:
        now = time.monotonic()
        delay = max(0.0, _next_codex_launch_at - now)
        if delay:
            time.sleep(delay)
        _next_codex_launch_at = time.monotonic() + interval


def _wait_for_shared_launch_slot(path: Path, interval: float) -> None:
    """Coordinate the startup interval across workbench worker processes."""
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="ascii") as lock:
        if os.name == "nt":
            import msvcrt

            lock.seek(0, os.SEEK_END)
            if lock.tell() == 0:
                lock.write("0")
                lock.flush()
            while True:
                try:
                    lock.seek(0)
                    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(0.05)
            unlock = lambda: msvcrt.locking(
                lock.fileno(), msvcrt.LK_UNLCK, 1
            )
        else:
            import fcntl

            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            unlock = lambda: fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        try:
            lock.seek(0)
            raw = lock.read().strip()
            try:
                next_launch = float(raw)
            except ValueError:
                next_launch = 0.0
            delay = max(0.0, next_launch - time.time())
            if delay:
                time.sleep(delay)
            lock.seek(0)
            lock.truncate()
            lock.write(f"{time.time() + interval:.9f}")
            lock.flush()
            os.fsync(lock.fileno())
        finally:
            lock.seek(0)
            unlock()


def is_transient_startup_failure(
    completed: subprocess.CompletedProcess,
    events_path: Path,
    log_path: Path,
) -> bool:
    """Recognize the Windows startup race seen before a thread is created."""
    if completed.returncode == 0 or events_path.stat().st_size:
        return False
    try:
        log = log_path.read_text(encoding="utf-8").lower()
    except (OSError, UnicodeError):
        return False
    return (
        "the system cannot find the path specified" in log
        or "os error 3" in log
    )


def structured_turn_is_complete(events_path: Path, result_path: Path) -> bool:
    """Return whether the agent wrote both its final event and valid JSON result."""
    try:
        with result_path.open(encoding="utf-8") as source:
            json.load(source)
        with events_path.open(encoding="utf-8") as source:
            return any(
                _is_final_success_event(event)
                for line in source
                if line.strip()
                for event in (json.loads(line),)
            )
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False


def _is_final_success_event(event: object) -> bool:
    if not isinstance(event, dict):
        return False
    if event.get("type") == "turn.completed":
        return True
    # Copilot CLI ends each prompt-mode session with a single result event.
    return event.get("type") == "result" and event.get("exitCode") == 0


def _stop_codex_process(process: subprocess.Popen) -> None:
    """Ask a Codex process group to stop, then escalate if necessary."""
    if process.poll() is not None:
        return
    try:
        if os.name == "nt" and hasattr(signal, "CTRL_BREAK_EVENT"):
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            os.killpg(process.pid, signal.SIGINT)
    except (OSError, ProcessLookupError):
        pass
    try:
        process.wait(timeout=CODEX_STOP_GRACE_SECONDS)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        process.terminate()
    except OSError:
        pass
    try:
        process.wait(timeout=CODEX_STOP_GRACE_SECONDS)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        process.kill()
    except OSError:
        pass
    try:
        process.wait(timeout=CODEX_STOP_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        pass


def _run_codex_process(
    command: list[str],
    *,
    prompt: str,
    workspace: Path,
    environment: dict[str, str],
    events,
    log,
    events_path: Path,
    result_path: Path,
    timeout_seconds: float | None,
    completion_grace_seconds: float | None,
) -> tuple[subprocess.CompletedProcess, bool, bool]:
    """Run Codex while detecting completed turns with lingering tools."""
    popen_options: dict = {
        "cwd": workspace,
        "env": environment,
        "stdin": subprocess.PIPE,
        "stdout": events,
        "stderr": log,
        "text": True,
        "encoding": "utf-8",
    }
    popen_options.update(windowless_popen_options())
    process = subprocess.Popen(command, **popen_options)
    if process.stdin is None:
        _stop_codex_process(process)
        raise CodexError("could not open Codex prompt input")
    try:
        process.stdin.write(prompt)
    except BrokenPipeError:
        pass
    finally:
        try:
            process.stdin.close()
        except BrokenPipeError:
            pass
    started_at = time.monotonic()
    completed_at: float | None = None
    structured_result_complete = False
    timed_out = False
    try:
        while process.poll() is None:
            now = time.monotonic()
            if structured_turn_is_complete(events_path, result_path):
                if completed_at is None:
                    completed_at = now
                elif (
                    completion_grace_seconds is not None
                    and now - completed_at >= completion_grace_seconds
                ):
                    log.write(
                        "\nDriver: structured turn completed but Codex did "
                        "not exit within the grace period; stopping its "
                        "process group.\n"
                    )
                    log.flush()
                    structured_result_complete = True
                    _stop_codex_process(process)
                    break
            if (
                timeout_seconds is not None
                and completed_at is None
                and now - started_at >= timeout_seconds
            ):
                log.write(
                    f"\nDriver: Codex exceeded the {timeout_seconds:g}-second "
                    "wall-clock timeout; stopping its process group.\n"
                )
                log.flush()
                timed_out = True
                _stop_codex_process(process)
                break
            time.sleep(CODEX_POLL_INTERVAL_SECONDS)
    except BaseException:
        _stop_codex_process(process)
        raise
    if structured_turn_is_complete(events_path, result_path):
        # Some launchers, notably the Cygwin `codex` shell wrapper, can return
        # a nonzero status after the CLI has already fulfilled the structured
        # output contract.  The validated result and final event are the
        # authoritative success signal in that case.
        structured_result_complete = True
    return (
        subprocess.CompletedProcess(command, process.returncode),
        structured_result_complete,
        timed_out,
    )


def _toml_value(value: object) -> str:
    """Encode account/provider settings as TOML CLI override values."""
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(
            f"{_toml_value(key)} = {_toml_value(item)}"
            for key, item in value.items()
        ) + "}"
    raise CodexError("unsupported value type in Codex account/provider configuration")


def codex_home_path(environment: Mapping[str, str] | None = None) -> Path:
    """Resolve the account directory used by the native Codex CLI."""
    environment = os.environ if environment is None else environment
    home = environment.get("CODEX_HOME")
    if not home:
        # Cygwin Python can have a different home from the native Windows CLI.
        home = environment.get("USERPROFILE") if sys.platform == "cygwin" else None
    if home and sys.platform == "cygwin" and PureWindowsPath(home).is_absolute():
        home = _run_local_command(
            ["cygpath", "-u", home], "could not convert Codex home"
        ).strip()
    path = Path(home).expanduser() if home else Path.home() / ".codex"
    # USERPROFILE names the user's home, rather than CODEX_HOME itself.
    if not environment.get("CODEX_HOME") and home:
        path /= ".codex"
    return path.resolve()


def codex_home_warnings(environment: Mapping[str, str] | None = None) -> list[str]:
    """Check account selection without starting Codex or changing its files."""
    environment = os.environ if environment is None else environment
    warnings = []
    try:
        path = codex_home_path(environment)
        if not path.is_dir():
            warnings.append(
                f"Codex home is not an existing directory: {path}. "
                "Set CODEX_HOME in the task's Model and web-search settings "
                "or in the CLI environment to an existing authenticated account directory."
            )
    except (CodexError, OSError, ValueError) as exc:
        warnings.append(f"Could not inspect Codex home: {exc}")
    return warnings


def report_codex_home_warnings() -> None:
    warnings = copilot_environment_warnings() if using_copilot() else codex_home_warnings()
    for warning in warnings:
        print(f"Warning: {warning}", file=sys.stderr)


def copilot_environment_warnings(
    environment: Mapping[str, str] | None = None,
) -> list[str]:
    """Check prerequisites for sandboxed Copilot turns without starting one."""
    environment = codex_subprocess_environment() if environment is None else environment
    warnings = []
    if shutil.which("python", path=environment.get("PATH")) is None:
        warnings.append(
            "No regular Python interpreter is on PATH. Copilot's sandbox cannot "
            "start the Microsoft Store python.exe alias, and agents must run "
            "`python -m validation.validate`; install Python from python.org."
        )
    return warnings


def codex_account_config_overrides() -> list[str]:
    """Preserve account and connection settings while ignoring other user config."""
    config_path = codex_home_path() / "config.toml"
    try:
        with config_path.open("rb") as source:
            config = tomllib.load(source)
    except FileNotFoundError:
        config = {}
    except (OSError, ValueError) as exc:
        raise CodexError(f"could not read Codex configuration {config_path}: {exc}") from exc

    settings = {key: config[key] for key in CODEX_ACCOUNT_CONFIG_KEYS if key in config}
    for key, value in settings.items():
        if not isinstance(value, str):
            raise CodexError(f"Codex configuration {config_path}: {key} must be a string")

    def config_directory(value: str) -> str:
        if PureWindowsPath(value).is_absolute():
            return value
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = config_path.parent / path
        return path_for_codex(path)

    if "sqlite_home" in settings:
        settings["sqlite_home"] = config_directory(settings["sqlite_home"])
    elif sqlite_home := os.environ.get("CODEX_SQLITE_HOME"):
        # --ignore-user-config also bypasses the SQLite environment override.
        settings["sqlite_home"] = sqlite_home

    provider = settings.get("model_provider", "openai")
    providers = config.get("model_providers", {})
    if not isinstance(providers, dict):
        raise CodexError(f"Codex configuration {config_path}: model_providers must be a table")
    if provider in providers:
        definition = providers[provider]
        if not isinstance(definition, dict):
            raise CodexError(f"Codex configuration {config_path}: selected provider must be a table")
        auth = definition.get("auth")
        if isinstance(auth, dict) and isinstance(auth.get("cwd"), str):
            auth["cwd"] = config_directory(auth["cwd"])
        settings["model_providers"] = {provider: definition}

    overrides = []
    for key, value in settings.items():
        overrides.extend(("--config", f"{key}={_toml_value(value)}"))
    return overrides


def configure_backend(args: argparse.Namespace) -> None:
    """Apply an entry point's --backend choice before any agent work."""
    select_backend(getattr(args, "backend", "codex"))
    validate_backend_options(args)


def copilot_readonly_paths(environment: Mapping[str, str] | None = None) -> list[str]:
    """Return toolchain roots the sandboxed shell must read but never modify."""
    environment = codex_subprocess_environment() if environment is None else environment
    candidates: list[Path] = []
    for name in ("latexmk", "pdflatex", "kpsewhich"):
        executable = shutil.which(name, path=environment.get("PATH"))
        if executable is None:
            continue
        resolved = Path(executable).resolve()
        # TeX Live and MiKTeX install binaries as <root>/bin/<platform>/<tool>.
        if resolved.parent.parent.name.casefold() == "bin":
            root = resolved.parent.parent.parent
            candidates.append(root)
            local = root.parent / "texmf-local"
            if local.is_dir():
                candidates.append(local)
    extra = environment.get(COPILOT_READONLY_PATHS_ENV, "")
    candidates.extend(Path(value).expanduser() for value in extra.split(os.pathsep) if value)
    paths: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        value = path_for_codex(candidate)
        key = os.path.normcase(value)
        if key not in seen:
            seen.add(key)
            paths.append(value)
    return paths


def copilot_sandbox_settings(
    web_search: str = "disabled",
    *,
    readonly_paths: list[str] | None = None,
) -> dict[str, object]:
    """Build the per-run Copilot sandbox policy for shell commands."""
    policy: dict[str, object] = {
        "network": {"allowOutbound": False, "allowLocalNetwork": False},
    }
    readonly = copilot_readonly_paths() if readonly_paths is None else readonly_paths
    if readonly:
        policy["filesystem"] = {"readonlyPaths": readonly}
    sandbox: dict[str, object] = {
        "enabled": True,
        "allowBypass": False,
        "addCurrentWorkingDirectory": True,
        # Lets sandboxed commands run toolchains on PATH, such as Python.
        "allowDevToolAccess": True,
        "userPolicy": policy,
    }
    if web_search != "disabled":
        # GitHub's remote web-search MCP server needs egress; shell commands
        # remain offline because the network policy above still applies.
        sandbox["sandboxMcpServers"] = False
    return {"sandbox": sandbox}


def source_copilot_home(environment: Mapping[str, str] | None = None) -> Path:
    environment = os.environ if environment is None else environment
    home = environment.get(COPILOT_HOME_ENV)
    return Path(home).expanduser() if home else Path.home() / ".copilot"


def _read_jsonc_object(path: Path) -> dict:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return {}
    text = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("//")
    )
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def prepare_copilot_home(web_search: str = "disabled") -> Path:
    """Create an isolated COPILOT_HOME holding only the sandbox policy.

    This keeps user MCP servers, plugins, agents, and instructions out of the
    run while authentication still comes from the OS credential store.
    """
    home = Path(tempfile.mkdtemp(prefix="loose-ends-copilot-")).resolve()
    (home / "settings.json").write_text(
        json.dumps(copilot_sandbox_settings(web_search), indent=2) + "\n",
        encoding="utf-8",
    )
    config = _read_jsonc_object(source_copilot_home() / "config.json")
    account = {key: config[key] for key in COPILOT_ACCOUNT_CONFIG_KEYS if key in config}
    if account:
        (home / "config.json").write_text(
            json.dumps(account, indent=2) + "\n",
            encoding="utf-8",
        )
    return home


def copilot_secret_env_vars(environment: Mapping[str, str]) -> list[str]:
    """Name credential-like variables to hide from Copilot's agent shells."""
    names = {
        name
        for name in environment
        if COPILOT_SECRET_ENV_PATTERN.search(name) and "," not in name
    }
    names.update(COPILOT_AUTH_ENV_VARS)
    return sorted(names)


def copilot_fetch_mcp_config(workspace: Path) -> dict:
    """Describe the page-fetch MCP server for one live-web-search run."""
    return {
        "mcpServers": {
            COPILOT_FETCH_SERVER: {
                "type": "local",
                "command": sys.executable,
                "args": [
                    str(COPILOT_FETCH_SERVER_PATH),
                    "--workspace",
                    str(Path(workspace).resolve()),
                ],
                "tools": ["*"],
            }
        }
    }


def build_copilot_command(
    *,
    copilot: str,
    workspace: Path,
    options: ModelOptions,
    usage_path: Path,
    web_search: str = "disabled",
    environment: Mapping[str, str] | None = None,
) -> list[str]:
    if web_search not in WEB_SEARCH_MODES:
        raise CodexError(
            "web search must be one of " + ", ".join(WEB_SEARCH_MODES)
        )
    if options.fast or options.ultrafast:
        raise CodexError("the Copilot backend does not support Fast or Ultrafast mode")
    if options.reasoning_effort in COPILOT_UNSUPPORTED_REASONING_EFFORTS:
        raise CodexError(
            f"the Copilot backend does not support reasoning effort {options.reasoning_effort}"
        )
    command = [
        copilot,
        "--output-format",
        "json",
        "--stream",
        "off",
        "--allow-all-tools",
        "--no-ask-user",
        "--no-custom-instructions",
        "--no-auto-update",
        "--sandbox",
        "-C",
        path_for_codex(workspace),
        "--usage-output-file",
        path_for_codex(usage_path),
    ]
    if web_search == "disabled":
        command.append("--disable-builtin-mcps")
    else:
        # Copilot has a single web-search tool, so indexed and live match.
        command.extend(
            (
                "--disable-mcp-server",
                "githubiq",
                "--add-github-mcp-tool",
                "web_search",
            )
        )
    if web_search == "live":
        command.extend(
            ("--additional-mcp-config", json.dumps(copilot_fetch_mcp_config(workspace)))
        )
    if options.model is not None:
        command.extend(("--model", options.model))
    if options.reasoning_effort is not None:
        command.extend(("--reasoning-effort", options.reasoning_effort))
    environment = os.environ if environment is None else environment
    command.append("--secret-env-vars=" + ",".join(copilot_secret_env_vars(environment)))
    # Variadic, so it stays last; the prompt is read from standard input.
    command.extend(("--excluded-tools", *COPILOT_EXCLUDED_TOOLS))
    return command


def copilot_usage_event(
    usage_path: Path,
    events_path: Path,
    invocation_id: str | None = None,
) -> dict | None:
    """Summarize a Copilot usage file as one Codex-comparable usage event."""
    usage = _read_jsonc_object(usage_path)
    metrics = usage.get("modelMetrics")
    if not isinstance(metrics, dict) or not metrics:
        return None
    fields = {
        "input_tokens": "inputTokens",
        "cached_input_tokens": "cacheReadTokens",
        "cache_write_input_tokens": "cacheWriteTokens",
        "output_tokens": "outputTokens",
        "reasoning_output_tokens": "reasoningTokens",
    }
    totals = dict.fromkeys(fields, 0)
    models: dict[str, dict[str, int]] = {}
    for model, metric in metrics.items():
        counts = metric.get("usage") if isinstance(metric, dict) else None
        if not isinstance(counts, dict):
            continue
        values = {
            name: counts.get(source) if type(counts.get(source)) is int else 0
            for name, source in fields.items()
        }
        models[str(model)] = values
        for name, value in values.items():
            totals[name] += value
    session_id = None
    try:
        with events_path.open(encoding="utf-8", errors="replace") as events:
            for line in events:
                if '"result"' not in line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict) and event.get("type") == "result":
                    session_id = event.get("sessionId", session_id)
    except OSError:
        pass
    return {
        "type": COPILOT_USAGE_EVENT,
        "invocation_id": invocation_id,
        "session_id": session_id,
        "model": usage.get("currentModel") or next(iter(models), None),
        "models": models,
        "usage": totals,
        "nano_aiu": usage.get("totalNanoAiu"),
        "premium_requests": usage.get("totalPremiumRequestCost"),
    }


def _append_event(events_path: Path, event: dict) -> None:
    with events_path.open("a", encoding="utf-8") as events:
        events.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")


def append_copilot_usage_event(
    usage_path: Path,
    events_path: Path,
    invocation_id: str | None = None,
) -> None:
    event = copilot_usage_event(usage_path, events_path, invocation_id)
    if event is None:
        return
    _append_event(events_path, event)


def copilot_final_message(events_path: Path) -> str | None:
    """Return the last nonempty assistant message from Copilot JSONL output."""
    message = None
    try:
        with events_path.open(encoding="utf-8", errors="replace") as events:
            for line in events:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(event, dict) or event.get("type") != "assistant.message":
                    continue
                data = event.get("data")
                content = data.get("content") if isinstance(data, dict) else None
                if isinstance(content, str) and content.strip():
                    message = content
    except OSError:
        return None
    return message


def _json_object_from_message(message: str) -> object:
    text = message.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    return json.loads(text)


def with_copilot_result_instructions(
    prompt: str,
    *,
    workspace: Path,
    schema_path: Path,
    result_path: Path,
) -> str:
    """Replace Codex's --output-schema contract with explicit instructions."""
    schema = schema_path.resolve()
    schema_reference = (
        schema.relative_to(workspace).as_posix()
        if schema.is_relative_to(workspace)
        else path_for_codex(schema)
    )
    return (
        prompt.rstrip()
        + "\n\n# Final result\n\n"
        + f"Before finishing, write your final result as a single JSON object to "
        + f"`{result_path.relative_to(workspace).as_posix()}` in the working "
        + f"directory. It must conform exactly to the JSON Schema in "
        + f"`{schema_reference}`. Your final message must be the same JSON "
        + "object and nothing else.\n"
    )


def recover_copilot_result(events_path: Path, result_path: Path) -> None:
    """Fall back to the final assistant message when no result file exists."""
    try:
        with result_path.open(encoding="utf-8") as source:
            json.load(source)
        return
    except (OSError, UnicodeError, json.JSONDecodeError):
        pass
    message = copilot_final_message(events_path)
    if message is None:
        return
    try:
        value = _json_object_from_message(message)
    except json.JSONDecodeError:
        return
    result_path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def build_exec_command(
    *,
    codex: str,
    workspace: Path,
    prompt: str,
    schema_path: Path,
    result_path: Path,
    options: ModelOptions,
    web_search: str = "disabled",
    model_writes_result: bool = False,
) -> list[str]:
    if web_search not in WEB_SEARCH_MODES:
        raise CodexError(
            "web search must be one of " + ", ".join(WEB_SEARCH_MODES)
        )
    command = [
        codex,
        "exec",
        "--ephemeral",
        "--ignore-user-config",
        "--disable",
        "shell_snapshot",
        "--disable",
        "skill_mcp_dependency_install",
        "--config",
        f'web_search="{web_search}"',
        "--config",
        "apps._default.enabled=false",
        "--config",
        "agents.enabled=false",
        "--config",
        "sandbox_workspace_write.network_access=false",
        "--config",
        'windows.sandbox="elevated"',
        "--skip-git-repo-check",
        "--sandbox",
        "workspace-write",
        "--json",
        "--color",
        "never",
        "-C",
        path_for_codex(workspace),
    ]
    command.extend(codex_account_config_overrides())
    if not model_writes_result:
        command.extend(
            (
                "--output-schema",
                path_for_codex(schema_path),
                "-o",
                path_for_codex(result_path),
            )
        )
    if options.model is not None:
        command.extend(("--model", options.model))
    if options.reasoning_effort is not None:
        command.extend(
            (
                "--config",
                f'model_reasoning_effort="{options.reasoning_effort}"',
            )
        )
    if options.fast or options.ultrafast:
        command.extend(
            (
                "--config",
                "features.fast_mode=true",
                "--config",
                f'service_tier="{options.service_tier}"',
            )
        )
    # Windows has a comparatively small process command-line limit.  Prompts
    # can exceed it, especially for manuscript-writing agents, so keep the
    # prompt off argv and tell Codex to read it from standard input.
    command.append("-")
    return command


@record_transcript
def run_structured_codex(
    *,
    codex: str,
    workspace: Path,
    prompt: str,
    schema_path: Path,
    result_filename: str = "agent-result.json",
    events_filename: str = "events.jsonl",
    log_filename: str = "run.log",
    options: ModelOptions = ModelOptions(),
    web_search: str = "disabled",
    launch_interval: float = CODEX_LAUNCH_INTERVAL_SECONDS,
    timeout_seconds: float | None = None,
    completion_grace_seconds: float | None = None,
    model_writes_result: bool = False,
) -> Path:
    """Run one structured agent turn and return its final-response path."""
    arguments = dict(
        codex=codex,
        workspace=workspace,
        prompt=prompt,
        schema_path=schema_path,
        result_filename=result_filename,
        events_filename=events_filename,
        log_filename=log_filename,
        options=options,
        web_search=web_search,
        launch_interval=launch_interval,
        timeout_seconds=timeout_seconds,
        completion_grace_seconds=completion_grace_seconds,
        model_writes_result=model_writes_result,
    )
    if not using_copilot():
        return _run_structured_turn(**arguments, copilot_home=None)
    copilot_home = prepare_copilot_home(web_search)
    try:
        return _run_structured_turn(**arguments, copilot_home=copilot_home)
    finally:
        shutil.rmtree(copilot_home, ignore_errors=True)


def _run_structured_turn(
    *,
    codex: str,
    workspace: Path,
    prompt: str,
    schema_path: Path,
    result_filename: str,
    events_filename: str,
    log_filename: str,
    options: ModelOptions,
    web_search: str,
    launch_interval: float,
    timeout_seconds: float | None,
    completion_grace_seconds: float | None,
    model_writes_result: bool,
    copilot_home: Path | None,
) -> Path:
    workspace = workspace.resolve()
    grant_workspace_owner_inheritance(workspace)
    require_secure_windows_sandbox(codex, workspace)
    result_path = workspace / result_filename
    events_path = workspace / events_filename
    log_path = workspace / log_filename
    events_path.write_text("", encoding="utf-8")
    log_path.write_text("", encoding="utf-8")
    environment = codex_subprocess_environment()
    usage_path: Path | None = None
    invocation_id: str | None = None
    if copilot_home is not None:
        usage_path = copilot_home / COPILOT_USAGE_FILENAME
        invocation_id = uuid.uuid4().hex
        _append_event(
            events_path,
            {"type": COPILOT_STARTED_EVENT, "invocation_id": invocation_id},
        )
        if not model_writes_result:
            prompt = with_copilot_result_instructions(
                prompt,
                workspace=workspace,
                schema_path=schema_path,
                result_path=result_path,
            )
        command = build_copilot_command(
            copilot=codex,
            workspace=workspace,
            options=options,
            usage_path=usage_path,
            web_search=web_search,
            environment=environment,
        )
        environment[COPILOT_HOME_ENV] = str(copilot_home)
    else:
        command = build_exec_command(
            codex=codex,
            workspace=workspace,
            prompt=prompt,
            schema_path=schema_path,
            result_path=result_path,
            options=options,
            web_search=web_search,
            model_writes_result=model_writes_result,
        )
    agent = backend_name()

    completed: subprocess.CompletedProcess | None = None
    structured_result_complete = False
    timed_out = False
    credit_error: str | None = None
    for attempt in range(1, MAX_CODEX_START_ATTEMPTS + 1):
        wait_for_codex_launch_slot(launch_interval)
        try:
            with (
                events_path.open("a", encoding="utf-8") as events,
                log_path.open("a", encoding="utf-8") as log,
            ):
                if attempt > 1:
                    log.write(
                        f"\n--- {agent} startup retry {attempt}/"
                        f"{MAX_CODEX_START_ATTEMPTS} ---\n"
                    )
                    log.flush()
                if (
                    timeout_seconds is None
                    and completion_grace_seconds is None
                ):
                    completed = subprocess.run(
                        command,
                        cwd=workspace,
                        env=environment,
                        input=prompt,
                        stdout=events,
                        stderr=log,
                        text=True,
                        encoding="utf-8",
                        check=False,
                    )
                else:
                    (
                        completed,
                        structured_result_complete,
                        timed_out,
                    ) = _run_codex_process(
                        command,
                        prompt=prompt,
                        workspace=workspace,
                        environment=environment,
                        events=events,
                        log=log,
                        events_path=events_path,
                        result_path=result_path,
                        timeout_seconds=timeout_seconds,
                        completion_grace_seconds=completion_grace_seconds,
                    )
        except OSError as exc:
            if (
                attempt < MAX_CODEX_START_ATTEMPTS
                and getattr(exc, "winerror", None) in {2, 3}
            ):
                with log_path.open("a", encoding="utf-8") as log:
                    log.write(f"{agent} startup error: {exc}\n")
                continue
            raise CodexError(
                f"could not start {agent}; workspace preserved at "
                f"{workspace}: {exc}"
            ) from exc

        credit_error = agent_credit_error(events_path, log_path)
        if credit_error is not None:
            pause_queue_for_credit_error(credit_error)
            break
        if completed.returncode == 0 or structured_result_complete or timed_out:
            break
        if (
            attempt == MAX_CODEX_START_ATTEMPTS
            or not is_transient_startup_failure(
                completed,
                events_path,
                log_path,
            )
        ):
            break

    note_windows_sandbox_failure(log_path)
    if usage_path is not None:
        append_copilot_usage_event(usage_path, events_path, invocation_id)
        if not model_writes_result:
            recover_copilot_result(events_path, result_path)
    try:
        normalize_workspace_access(workspace, codex)
    except CodexError as exc:
        if not structured_turn_is_complete(events_path, result_path):
            raise
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\nDriver ACL recovery warning: {exc}\n")
    if credit_error is not None:
        raise CodexError(
            f"{agent} credits exhausted: {credit_error}; "
            f"workspace preserved at {workspace}"
        )
    if timed_out:
        raise CodexError(
            f"{agent} exceeded the {timeout_seconds:g}-second wall-clock "
            f"timeout; workspace preserved at {workspace}"
        )
    if (
        completed is None
        or (completed.returncode != 0 and not structured_result_complete)
    ):
        returncode = completed.returncode if completed is not None else "unknown"
        raise CodexError(
            f"{agent} exited with status {returncode}; "
            f"workspace preserved at {workspace}"
        )
    return result_path


def _merge_repair_logs(workspace: Path) -> None:
    pairs = (
        ("repair-events.jsonl", "events.jsonl", "repair turn events"),
        ("repair-run.log", "run.log", "repair turn log"),
    )
    for source_name, destination_name, heading in pairs:
        source = workspace / source_name
        if not source.is_file():
            continue
        contents = source.read_text(encoding="utf-8", errors="replace")
        with (workspace / destination_name).open("a", encoding="utf-8") as output:
            output.write(f"\n--- {heading} ---\n")
            output.write(contents)
        source.unlink()


def run_validated_codex(
    *,
    codex: str,
    workspace: Path,
    prompt: str,
    schema_path: Path,
    validator: OutputValidator,
    options: ModelOptions = ModelOptions(),
    web_search: str = "disabled",
    launch_interval: float = CODEX_LAUNCH_INTERVAL_SECONDS,
    timeout_seconds: float | None = None,
    completion_grace_seconds: float | None = None,
    repair_turns: int = 1,
) -> validation_common.ValidationReport:
    """Run Codex with in-turn validation and authoritative host rechecking."""
    workspace = workspace.resolve()
    try:
        result_schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CodexError(f"could not read result schema for validation: {exc}") from exc
    effective_validator = OutputValidator(
        validator.source,
        validator.validate,
        {**validator.expectations, "result_schema": result_schema},
    )
    validation_directory = stage_output_validator(
        workspace,
        effective_validator,
    )
    # The elevated Windows sandbox applies deny-read ACLs to host-created
    # workspace content unless it is explicitly staged for the sandbox user.
    # The model must be able to inspect and execute this validator because the
    # prompt requires an in-turn `python -m validation.validate` check.
    grant_sandbox_read_access(validation_directory)
    resolved_schema = schema_path.resolve()
    if resolved_schema.is_relative_to(workspace):
        # Review workflows generate a claim-constrained schema inside the
        # workspace.  Keep that authoritative schema inspectable as well.
        grant_sandbox_read_access(resolved_schema)
    run_structured_codex(
        codex=codex,
        workspace=workspace,
        prompt=with_validation_instructions(prompt),
        schema_path=schema_path,
        options=options,
        web_search=web_search,
        launch_interval=launch_interval,
        timeout_seconds=timeout_seconds,
        completion_grace_seconds=completion_grace_seconds,
        model_writes_result=True,
    )
    report = effective_validator.validate(
        workspace=workspace,
        expectations=effective_validator.expectations,
    )
    for _ in range(repair_turns):
        if report.valid or not report.repairable:
            break
        repair_prompt = (
            "The previous agent turn left generated output that failed the "
            "authoritative deterministic validator. Preserve all substantive "
            "work and staged inputs. Fix only the reported output-contract "
            "issues, run `python -m validation.validate` until it passes, and "
            "then return a short completion note.\n\n"
            "Validation issues:\n"
            + "\n".join(f"- {issue.render()}" for issue in report.issues)
        )
        run_structured_codex(
            codex=codex,
            workspace=workspace,
            prompt=repair_prompt,
            schema_path=schema_path,
            events_filename="repair-events.jsonl",
            log_filename="repair-run.log",
            options=options,
            web_search=web_search,
            launch_interval=launch_interval,
            timeout_seconds=timeout_seconds,
            completion_grace_seconds=completion_grace_seconds,
            model_writes_result=True,
        )
        _merge_repair_logs(workspace)
        report = effective_validator.validate(
            workspace=workspace,
            expectations=effective_validator.expectations,
        )
    if not report.valid:
        raise CodexError("generated output failed validation:\n" + report.failure_message())
    return report
