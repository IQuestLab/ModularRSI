"""Summarize solver trajectories and build Phase0 editor instructions.

Compact per-trial summary contains:
- task name + trial id
- reward
- exception type (if any) — `AgentTimeoutError`, `ContextLengthExceededError`, etc.
- # episodes, token totals
- last 2-3 step `message` snippets (where the agent was just before failure)
- detected failure signals (timeout / parse_error / max_iterations / stuck-in-loop)

Summaries are bounded so a reflection window fits in the editor context.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

_logger = logging.getLogger(__name__)


@dataclass
class TrialSummary:
    task_name: str
    trial_name: str
    reward: float | None
    exception_type: str | None
    exception_message: str | None
    n_episodes: int
    n_input_tokens: int
    n_output_tokens: int
    # Agent-execution wall-clock (from result.json agent_execution timestamps).
    # An efficiency signal for the editor: rule-based, hard to game.
    duration_sec: float | None = None
    last_step_messages: list[str] = field(default_factory=list)
    failure_signals: list[str] = field(default_factory=list)
    # Diagnostic fields extracted from the ATIF trajectory
    tool_call_counts: dict[str, int] = field(default_factory=dict)
    repeated_command_ratio: float = 0.0
    obs_lengths: list[int] = field(default_factory=list)
    context_compression_count: int = 0
    # Per-command-class breakdown — cracks open the single `bash_command` bucket
    # The solver exposes one tool, so parse terminal keystrokes for command counts.
    command_breakdown: dict = field(default_factory=dict)
    # Absolute path to this trial's output dir (result.json + agent/ live here).
    # The editor reads under it to investigate the raw trajectory itself.
    trial_dir: str | None = None
    # Which module variants the per-task Composer actually selected for THIS trial
    # (from trajectory agent.extra.bundle). Lets the editor see whether the variant
    # a past generation wrote was even called. None for trajectories that predate
    # bundle recording.
    bundle: dict[str, str] | None = None

    def to_markdown(
        self,
        max_msg_chars: int = 600,
        include_reward: bool = True,
        include_trace: bool = False,
    ) -> str:
        """A THIN index entry — header + one orientation line + WHERE to look.

        Deliberately NOT a digest: the editor investigates the raw trial files
        itself. A heavy pre-digest here biased the editor
        toward the agent_loop (its signals over-represent agent behaviour).

        ``include_reward=True`` (default) shows the ground-truth verdict — the
        SUPPORT-side training signal the reflecting editor is entitled to (the
        verifier's test files stay blocked; the scalar is all it gets).
        Pass ``include_reward=False`` for reward-blind consumers (the L4.5
        review gate judges mechanism, not score — keep it unleaked).

        ``include_trace=True`` appends the kernel MODULE TRACE (per-Protocol-call
        record of every active variant) so a lens investigator / the review gate
        can SEE whether a variant's declared mechanism actually fired at runtime.
        Reward-free (method-call summaries only) → safe for reward-blind readers.
        """
        head = f"### `{self.task_name}`"
        if include_reward:
            if self.reward is None:
                head += " — **NO-SCORE** (harness error, verifier never ran)"
            elif self.reward >= 1.0:
                head += f" — **PASSED** (reward {self.reward:g})"
            else:
                head += f" — **FAILED** (reward {self.reward:g})"
        if self.exception_type:
            head += f" — ended with exception `{self.exception_type}`"
        elif not include_reward:
            head += " — ran to completion (no crash)"
        lines = [head]
        # One compact orientation line (a hint of WHERE to look, NOT a verdict).
        bits = [f"{self.n_episodes} episodes"]
        if self.duration_sec:
            bits.append(f"agent time {self.duration_sec:.0f}s")
        if self.n_input_tokens or self.n_output_tokens:
            bits.append(
                f"tokens {self.n_input_tokens / 1000:.0f}k in"
                f"/{self.n_output_tokens / 1000:.1f}k out"
            )
        cb = self.command_breakdown or {}
        classes = cb.get("class_counts") or {}
        if classes:
            top = sorted(classes.items(), key=lambda x: -x[1])[:5]
            bits.append("cmds " + " ".join(f"{k}×{v}" for k, v in top))
            if cb.get("nav_work_ratio"):
                bits.append(f"nav/work {cb['nav_work_ratio']}")
        elif self.tool_call_counts:
            top = max(self.tool_call_counts.items(), key=lambda x: x[1])
            bits.append(f"top tool {top[0]}×{top[1]}")
        if self.obs_lengths:
            bits.append(f"obs max {max(self.obs_lengths)}c")
        if self.repeated_command_ratio > 0.05:
            bits.append(f"repeats {self.repeated_command_ratio:.0%}")
        bits.append(f"compress {self.context_compression_count}")
        if self.failure_signals:
            bits.append("signals: " + ",".join(self.failure_signals))
        lines.append("- " + " | ".join(bits))
        # Which variants actually RAN this task (per-task Composer choice). This is
        # the editor's only window into whether the variant it wrote was selected;
        # it is a fact about composition, not a behavioral verdict.
        if self.bundle:
            lines.append(
                "- active bundle: "
                + " ".join(f"{k}={v}" for k, v in self.bundle.items())
            )
        # Most-repeated raw command — surfaces "doing it the hard way".
        top_rep = (self.command_breakdown or {}).get("top_repeated") or []
        if top_rep and top_rep[0][1] >= 3:
            lines.append(f"- most-repeated: `{top_rep[0][0]}` ×{top_rep[0][1]}")
        if self.trial_dir:
            lines.append(f"- investigate: `{self.trial_dir}/`")
            lines.append(
                "  read `exception.txt` (if it crashed); "
                "`agent/episode-*/prompt.txt` (what the agent SAW) and "
                "`agent/episode-*/response.txt` (what it DID); "
                "`agent/terminus_2_modular.pane` (raw terminal). "
                "(`result.json` / `verifier/` are blocked.)"
            )
        if include_trace and self.trial_dir:
            block = _module_trace_block_for_trial(Path(str(self.trial_dir)))
            if block:
                lines.append("")
                lines.append(block)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Per-trial extraction
# ---------------------------------------------------------------------------


def _extract_tool_stats(
    trajectory: dict,
) -> tuple[dict[str, int], float, list[int], int]:
    """Extract diagnostic statistics from an ATIF trajectory.

    Returns:
        (tool_call_counts, repeated_command_ratio, obs_lengths, compression_count)
    """
    steps = trajectory.get("steps", [])
    fn_counts: Counter[str] = Counter()
    keystroke_seq: list[str] = []
    obs_lengths: list[int] = []
    compression_count = 0

    for s in steps:
        # Tool call function names and keystroke sequence
        for tc in s.get("tool_calls") or []:
            fn = (
                tc.get("function_name")
                or tc.get("function")
                or tc.get("name")
                or "unknown"
            )
            fn_counts[fn] += 1
            args = tc.get("arguments") or {}
            ks = args.get("keystrokes")
            if isinstance(ks, str) and ks.strip():
                keystroke_seq.append(ks.strip()[:300])

        # Observation lengths
        for r in (s.get("observation") or {}).get("results", []):
            content = r.get("content") or ""
            obs_lengths.append(len(content))

        # Context compression: system-source steps indicate a compression handoff
        if s.get("source") == "system":
            compression_count += 1

    # Fraction of keystrokes that are exact repeats of something seen before
    repeated_ratio = 0.0
    if len(keystroke_seq) > 1:
        seen: set[str] = set()
        repeats = 0
        for ks in keystroke_seq:
            if ks in seen:
                repeats += 1
            seen.add(ks)
        repeated_ratio = repeats / len(keystroke_seq)

    return dict(fn_counts), repeated_ratio, obs_lengths, compression_count


# Command classes — the solver's only tool is tmux `send_keys`, so every shell
# command collapses into one `bash_command` bucket and navigation overhead
# Aggregate tool counts hide repeated shell commands. Classifying keystrokes
# back into classes restores the behavior/efficiency signal the editor needs.
_NAV_CMDS = {
    "ls",
    "cat",
    "cd",
    "pwd",
    "find",
    "grep",
    "rg",
    "head",
    "tail",
    "less",
    "more",
    "file",
    "tree",
    "which",
    "wc",
    "stat",
    "realpath",
    "dirname",
    "basename",
    "type",
    "locate",
    "cut",
    "sort",
    "uniq",
    "diff",
    "nl",
    "xxd",
    "od",
    "readlink",
    "env",
}
_EDIT_CMDS = {
    "sed",
    "vim",
    "vi",
    "nano",
    "emacs",
    "patch",
    "tee",
    "touch",
    "mkdir",
    "cp",
    "mv",
    "rm",
    "chmod",
    "chown",
    "ln",
    "awk",
}
_RUN_CMDS = {
    "make",
    "cmake",
    "gcc",
    "g++",
    "clang",
    "clang++",
    "python",
    "python3",
    "pytest",
    "node",
    "npm",
    "npx",
    "yarn",
    "go",
    "cargo",
    "rustc",
    "bash",
    "sh",
    "zsh",
    "ruby",
    "perl",
    "java",
    "javac",
    "mvn",
    "gradle",
    "dotnet",
    "ctest",
    "ninja",
    "tox",
    "nox",
    "unittest",
}
_PKG_CMDS = {
    "pip",
    "pip3",
    "apt",
    "apt-get",
    "dpkg",
    "conda",
    "uv",
    "poetry",
    "yum",
    "brew",
    "gem",
    "cargo-install",
    "mamba",
}
_VCS_CMDS = {"git", "svn", "hg"}


def _classify_command(ks: str) -> str:
    """Map one tmux keystroke command to a coarse class.

    Classes: navigate / edit / build_run / vcs / pkg / other. Heredocs and
    redirected writes count as edits; `./x` counts as build_run.
    """
    s = ks.strip()
    if not s:
        return "other"
    head = s.split("\n", 1)[0]
    if "<<" in head:
        return "edit"  # heredoc write
    first = re.split(r"\|\||&&|[|;&]", head, maxsplit=1)[0].strip()
    tok = first.split()
    if not tok:
        return "other"
    cmd = tok[0]
    # strip leading env assignments: `FOO=bar cmd ...`
    while "=" in cmd and not cmd.startswith(("/", "./", "$")) and len(tok) > 1:
        tok = tok[1:]
        cmd = tok[0]
    if cmd.startswith("./"):
        return "build_run"
    cmd = cmd.split("/")[-1]
    if cmd in _VCS_CMDS:
        return "vcs"
    if cmd in _PKG_CMDS:
        return "pkg"
    if cmd in _RUN_CMDS:
        return "build_run"
    if cmd in _EDIT_CMDS:
        return "edit"
    if cmd in _NAV_CMDS:
        return "navigate"
    if cmd == "echo":
        return "edit" if ">" in head else "other"
    return "other"


def _command_breakdown(trajectory: dict) -> dict:
    """Decompose tmux keystrokes into command classes + repeat stats.

    Returns {class_counts, nav_work_ratio, top_repeated}. This is the
    behavior/efficiency signal `tool_call_counts` cannot give (single bucket).
    """
    classes: Counter[str] = Counter()
    exact: Counter[str] = Counter()
    for s in trajectory.get("steps", []):
        for tc in s.get("tool_calls") or []:
            ks = (tc.get("arguments") or {}).get("keystrokes")
            if not isinstance(ks, str) or not ks.strip():
                continue
            classes[_classify_command(ks)] += 1
            exact[ks.strip().split("\n", 1)[0][:80]] += 1
    nav = classes.get("navigate", 0)
    work = (
        classes.get("edit", 0)
        + classes.get("build_run", 0)
        + classes.get("vcs", 0)
        + classes.get("pkg", 0)
    )
    ratio = round(nav / work, 2) if work else float(nav)
    top_repeated = [(c, n) for c, n in exact.most_common(3) if n >= 2]
    return {
        "class_counts": dict(classes),
        "nav_work_ratio": ratio,
        "top_repeated": top_repeated,
    }


def _last_agent_messages(trajectory: dict, k: int = 3) -> list[str]:
    """Return last k 'agent' source messages (newest last)."""
    agent_msgs: list[str] = []
    for s in trajectory.get("steps", []):
        if s.get("source") == "agent":
            m = s.get("message") or ""
            if isinstance(m, list):
                m = " ".join(str(p) for p in m)
            agent_msgs.append(m)
    return agent_msgs[-k:]


_REPEAT_THRESHOLD = 3  # ≥3 identical keystrokes in a row → "stuck in loop"


def _detect_signals(result: dict, trajectory: dict | None) -> list[str]:
    signals: list[str] = []

    exc = (result.get("exception_info") or {}).get("exception_type")
    if exc == "AgentTimeoutError":
        signals.append("timeout")
    elif exc == "ContextLengthExceededError":
        signals.append("context_overflow")
    elif exc:
        signals.append(f"exc:{exc}")

    meta = (result.get("agent_result") or {}).get("metadata") or {}
    n_eps = meta.get("n_episodes")
    if n_eps is not None and n_eps >= 145:  # close to max_turns=150
        signals.append("max_iterations")

    if trajectory is None:
        return signals

    steps = trajectory.get("steps", [])
    # Count parse-error steps (agent steps whose `message` is raw JSON-looking)
    parse_errors = 0
    for s in steps:
        if s.get("source") != "agent":
            continue
        msg = s.get("message") or ""
        # Parse-error steps have raw LLM output as message; agent steps
        # built from successful parse have "Analysis: ...\nPlan: ..." format
        if (
            isinstance(msg, str)
            and msg.strip().startswith("{")
            and "analysis" in msg.lower()
        ):
            parse_errors += 1
    if parse_errors >= 3:
        signals.append(f"parse_errors({parse_errors})")

    # Stuck-in-loop: same keystrokes repeated
    keystroke_seq: list[str] = []
    for s in steps:
        for tc in s.get("tool_calls") or []:
            args = tc.get("arguments") or {}
            ks = args.get("keystrokes")
            if isinstance(ks, str) and ks.strip():
                keystroke_seq.append(ks.strip()[:200])
    # Walk and find longest consecutive run of same value
    if keystroke_seq:
        cur = 1
        run = 1
        for a, b in zip(keystroke_seq, keystroke_seq[1:]):
            if a == b:
                run += 1
                cur = max(cur, run)
            else:
                run = 1
        if cur >= _REPEAT_THRESHOLD:
            signals.append(f"stuck_loop({cur}x)")

    return signals


def _agent_duration_sec(result: dict) -> float | None:
    """Agent wall-clock from result.json's agent_execution timestamps.

    Falls back to the trial-level started/finished span (which additionally
    covers env setup + verification) when the phase record is absent.
    """
    from datetime import datetime

    def _span(rec: dict | None) -> float | None:
        rec = rec or {}
        s, e = rec.get("started_at"), rec.get("finished_at")
        if not (s and e):
            return None
        try:
            t0 = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
            t1 = datetime.fromisoformat(str(e).replace("Z", "+00:00"))
        except ValueError:
            return None
        return max(0.0, (t1 - t0).total_seconds())

    return _span(result.get("agent_execution")) or _span(result)


def summarize_trial(trial_dir: Path) -> TrialSummary | None:
    rj = trial_dir / "result.json"
    if not rj.exists():
        return None
    try:
        result = json.loads(rj.read_text())
    except Exception as exc:
        _logger.warning("could not parse %s: %s", rj, exc)
        return None

    task_name = result.get("task_name") or trial_dir.name.split("__")[0]
    trial_name = result.get("trial_name") or trial_dir.name
    reward = (result.get("verifier_result") or {}).get("rewards", {}).get("reward")
    ex_info = result.get("exception_info") or {}
    agent_result = result.get("agent_result") or {}
    meta = agent_result.get("metadata") or {}

    # Trajectory (may be absent if SUPPORTS_ATIF=False)
    trajectory = None
    tj = trial_dir / "agent" / "trajectory.json"
    if tj.exists():
        try:
            trajectory = json.loads(tj.read_text())
        except Exception:
            pass

    last_msgs: list[str] = []
    tool_call_counts: dict[str, int] = {}
    repeated_ratio = 0.0
    obs_lengths: list[int] = []
    compression_count = 0
    command_breakdown: dict = {}
    bundle: dict[str, str] | None = None
    if trajectory is not None:
        last_msgs = _last_agent_messages(trajectory, k=3)
        tool_call_counts, repeated_ratio, obs_lengths, compression_count = (
            _extract_tool_stats(trajectory)
        )
        command_breakdown = _command_breakdown(trajectory)
        b = ((trajectory.get("agent") or {}).get("extra") or {}).get("bundle")
        if isinstance(b, dict):
            bundle = {str(k): str(v) for k, v in b.items()}

    return TrialSummary(
        task_name=task_name,
        trial_name=trial_name,
        reward=reward,
        exception_type=ex_info.get("exception_type"),
        exception_message=ex_info.get("exception_message"),
        n_episodes=int(meta.get("n_episodes") or 0),
        n_input_tokens=int(agent_result.get("n_input_tokens") or 0),
        n_output_tokens=int(agent_result.get("n_output_tokens") or 0),
        duration_sec=_agent_duration_sec(result),
        last_step_messages=last_msgs,
        failure_signals=_detect_signals(result, trajectory),
        tool_call_counts=tool_call_counts,
        repeated_command_ratio=repeated_ratio,
        obs_lengths=obs_lengths,
        context_compression_count=compression_count,
        command_breakdown=command_breakdown,
        trial_dir=str(trial_dir),
        bundle=bundle,
    )


# ---------------------------------------------------------------------------
# Evolution log — per-gen "hypothesis → verification" chain (anti-flip-flop)
# ---------------------------------------------------------------------------

EVOLUTION_LOG_NAME = "evolution_log.jsonl"


def changed_module_files(old_modules: Path, new_modules: Path) -> list[str]:
    """Module files that differ between two modules/ dirs (new vs old)."""
    changed = _changed_py_files(old_modules, new_modules)
    # also note an editor-written active_bundle.json (enables a new module)
    if (Path(new_modules) / "active_bundle.json").exists() and not (
        Path(old_modules) / "active_bundle.json"
    ).exists():
        changed.append("active_bundle.json")
    return sorted(changed)


def _changed_py_files(old_root: Path, new_root: Path) -> list[str]:
    """`.py` files that differ between two trees (rel paths; `NEW:` prefix
    for files absent from the old tree). Dunder path parts are skipped."""

    def _hashes(d: Path) -> dict[str, str]:
        out: dict[str, str] = {}
        if not d.is_dir():
            return out
        for f in sorted(d.rglob("*.py")):
            if any(part.startswith("__") for part in f.parts):
                continue
            try:
                out[str(f.relative_to(d))] = hashlib.md5(f.read_bytes()).hexdigest()
            except Exception:
                pass
        return out

    old_h, new_h = _hashes(Path(old_root)), _hashes(Path(new_root))
    changed: list[str] = []
    for rel, h in new_h.items():
        if rel not in old_h:
            changed.append(f"NEW:{rel}")
        elif old_h[rel] != h:
            changed.append(rel)
    return sorted(changed)


def extract_editor_intent(trajectory_path: Path | None, max_chars: int = 1200) -> str:
    """The editor's stated intent for this generation — its own account of WHAT
    it changed and WHY.

    Anchored to the message on the step that emits ``<commit_patch/>``: that is
    where the editor states the change + rationale it is committing. We do NOT
    take the trajectory's *last* agent message — that is typically the
    post-commit wrap-up ("the patch has been committed…") or, on some sessions,
    empty, which is why the log used to record truncated boilerplate or nothing
    at all (notably for the change that actually got promoted).

    Fallbacks: the last substantive Analysis/Plan message; then the last
    non-empty agent message. Empty only if there is no trajectory.
    """
    if trajectory_path is None or not Path(trajectory_path).exists():
        return ""
    try:
        data = json.loads(Path(trajectory_path).read_text())
    except Exception:
        return ""

    steps = data.get("steps", [])

    def _msg(step) -> str:
        m = step.get("message") or ""
        if isinstance(m, list):
            m = " ".join(str(p) for p in m)
        m = m.strip() if isinstance(m, str) else ""
        if m:
            return m
        # Some reasoning models leave `message` empty — the
        # editor's stated rationale only exists in reasoning_content. Noisier
        # (chain-of-thought), but infinitely better than recording "" as the
        # generation's intent (which blinds the anti-flip-flop memory).
        rc = step.get("reasoning_content") or ""
        return rc.strip() if isinstance(rc, str) else ""

    def _emits_commit(step) -> bool:
        for tc in step.get("tool_calls") or []:
            ks = (tc.get("arguments") or {}).get("keystrokes") or ""
            if isinstance(ks, str) and '"action": "commit_patch"' in ks:
                return True
        return False

    # post-commit / completion wrap-ups carry no hypothesis — skip them when
    # falling back. Checked after stripping a leading "Analysis:"/"Plan:" label.
    _WRAPUP = (
        "patch committed",
        "patch has been committed",
        "has been committed",
        "committed successfully",
        "task complete",
        "task is complete",
        "no further actions",
    )

    def _body(text: str) -> str:
        low = text.lower()
        for label in ("analysis:", "plan:"):
            if low.startswith(label):
                return text[len(label) :].strip()
        return text

    chosen = ""
    # 1. message on the (last) commit_patch step — the editor's change + rationale.
    #    Require it to be rich enough; a thin "set task_complete" commit step
    #    carries no intent, so fall through to the diagnosis in that case.
    for s in steps:
        if s.get("source") == "agent" and _emits_commit(s):
            t = _msg(s)
            if len(t) >= 150 and not _body(t).lower().startswith(_WRAPUP):
                chosen = t
    # 2. fallback: the LONGEST substantive Analysis/Plan that isn't a post-commit
    #    wrap-up. Longest (not last) because the diagnosis/triage is the richest
    #    message; later short Plans are edit-mechanics chatter ("fix indentation,
    #    retry the edit") that overshoot the actual intent.
    if not chosen:
        for s in steps:
            if s.get("source") != "agent":
                continue
            t = _msg(s)
            if len(t) < 120 or len(t) <= len(chosen):
                continue
            body = _body(t).lower()
            if body.startswith(_WRAPUP):
                continue
            if "analysis" in t.lower() or "plan" in t.lower() or "change" in body:
                chosen = t
    # 3. last resort: last non-empty agent message
    if not chosen:
        for s in steps:
            if s.get("source") == "agent":
                t = _msg(s)
                if t:
                    chosen = t

    chosen = " ".join(chosen.split())  # flatten newlines/runs of whitespace
    if len(chosen) <= max_chars:
        return chosen
    return chosen[:max_chars].rsplit(" ", 1)[0] + " …"


def extract_variant_meta_blocks(trajectory_path: Path | None) -> str:
    """Concatenate the editor's RAW LLM responses (the ``episode-*/response.txt``
    siblings of the trajectory file) that carry ``<variant_meta>`` blocks, for
    ``archive.parse_variant_meta``.

    IMPORTANT: this reads ``response.txt``, NOT ``trajectory.json``. The
    trajectory's agent ``message`` is the *parsed* form (``<analysis>``/``<plan>``
    + actions); a ``<variant_meta>`` block is emitted OUTSIDE ``<analysis>`` and is
    stripped when the step is recorded, so it never survives in trajectory.json.
    The raw per-episode response.txt is the only place it remains intact (newlines
    preserved, untruncated — unlike ``extract_editor_intent``).
    """
    if trajectory_path is None:
        return ""
    ep_dir = Path(trajectory_path).parent
    if not ep_dir.is_dir():
        return ""
    parts: list[str] = []
    for rt in sorted(ep_dir.glob("episode-*/response.txt")):
        try:
            t = rt.read_text()
        except Exception:
            continue
        if "<variant_meta" in t:
            parts.append(t)
    return "\n\n".join(parts)


_KERNEL_META_RE = re.compile(
    r"<kernel_meta>\s*(.*?)\s*</kernel_meta>", re.DOTALL | re.IGNORECASE
)


def append_evolution_log(archive_root: Path, record: dict) -> None:
    """Append one generation's record to the persistent evolution log."""
    try:
        path = Path(archive_root) / EVOLUTION_LOG_NAME
        with path.open("a") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        _logger.warning("failed to append evolution log: %s", exc)


_BUNDLE_TYPE_ORDER = [
    "agent_loop",
    "observation",
    "context_mgmt",
    "tools",
    "verification",
]


def _render_module_trace(traj: dict, max_notable: int = 12) -> str:
    """Aggregate `agent.extra.module_trace` (kernel-recorded module calls)
    into a compact block for the analyzer/editor.

    Two parts: per-method call counts with one example line (so a mechanism
    that NEVER fires is visible as a missing/zero line — this is the
    dead-feature detector), plus up to `max_notable` notable events
    (exceptions, summarizations, completion signals). Old trajectories
    without the field render as "" (graceful degradation).
    """
    events = ((traj.get("agent") or {}).get("extra") or {}).get("module_trace")
    if not isinstance(events, list) or not events:
        return ""
    counts: dict[str, int] = {}
    examples: dict[str, str] = {}
    notable: list[str] = []
    for ev in events:
        if not isinstance(ev, dict):
            continue
        key = f"{ev.get('module')}.{ev.get('call')}"
        counts[key] = counts.get(key, 0) + 1
        s = str(ev.get("summary") or "")
        examples.setdefault(key, s)
        if len(notable) < max_notable and (
            "⚠" in s
            or "summarized" in s
            or "task_complete=True" in s
            or "PARSE-ERROR" in s
        ):
            notable.append(f"  ! {key} {s[:160]}")
    lines = [
        "MODULE TRACE (kernel-recorded; every Protocol call of each active "
        "variant — a declared mechanism that never appears here never fired):"
    ]
    for key in sorted(counts, key=lambda k: (-counts[k], k)):
        lines.append(f"  {key} ×{counts[key]} (e.g. {examples[key][:120]})")
    if notable:
        lines.append("  notable events:")
        lines.extend(notable)
    dropped = ((traj.get("agent") or {}).get("extra") or {}).get("module_trace_dropped")
    if dropped:
        lines.append(f"  (+{dropped} events dropped past cap)")
    return "\n".join(lines)


def _module_trace_block_for_trial(trial_dir: Path) -> str:
    """Load one trial's ATIF trace and render its kernel MODULE TRACE block.

    Reads ONLY `agent/trajectory.json` and renders method-call summaries (no
    reward, no verifier) → safe for the reward-blind review gate. Empty string
    when no trace was recorded (older gens, or tracing wrap disabled)."""
    try:
        tj = Path(trial_dir) / "agent" / "trajectory.json"
        if not tj.exists():
            return ""
        traj = json.loads(tj.read_text())
    except Exception:
        return ""
    return _render_module_trace(traj)


# ---------------------------------------------------------------------------
# Edit self-review — after the editor commits a change, a SECOND editor pass
# re-investigates the trajectories + the diff and judges whether the change is
# (a) effective for the diagnosed failures and (b) NOT overfit to a specific
# task. Pure reasoning (no task reward read) → no eval-signal leakage.
# ---------------------------------------------------------------------------


def build_diff_text(
    parent_modules: Path,
    staging_modules: Path,
    changed_files: list[str],
    max_chars: int = 8000,
) -> str:
    """Unified diff (parent → staging) for the editor-changed module files.

    `changed_files` is the output of `changed_module_files` (entries may carry
    a ``NEW:`` prefix; ``active_bundle.json`` is included). NEW files diff
    against empty. Total output is capped at `max_chars`.
    """
    parent_modules = Path(parent_modules)
    staging_modules = Path(staging_modules)
    parts: list[str] = []
    for entry in changed_files:
        rel = entry.replace("NEW:", "")
        old_path = parent_modules / rel
        new_path = staging_modules / rel
        old_txt = old_path.read_text() if old_path.exists() else ""
        new_txt = new_path.read_text() if new_path.exists() else ""
        diff = difflib.unified_diff(
            old_txt.splitlines(keepends=True),
            new_txt.splitlines(keepends=True),
            fromfile=f"a/{rel}",
            tofile=f"b/{rel}",
        )
        parts.append("".join(diff))
    text = "\n".join(p for p in parts if p.strip())
    if not text.strip():
        return "(no textual diff — files identical or only metadata changed)"
    if len(text) > max_chars:
        text = text[:max_chars] + "\n… [diff truncated]"
    return text


_REVIEW_PREAMBLE = """\
<review_gate_mode>true</review_gate_mode>

You are the terminus-2-modular **editor**, reviewing a change you just made to
the agent's `modules/` to fix failures from the batch below. The change is
ALREADY applied to the staging copy you can read. Your job now is NOT to edit —
it is to **decide whether this change should be KEPT or REJECTED**.

# The change you made (diff: parent → staging)

```diff
{diff_text}
```

Your stated intent for this change was:
{intent}

# The batch this change was meant to help (index — investigate the paths)

{trial_summaries}

# Architecture facts (judge against THESE, not assumptions)

- The module library is ADDITIVE with a PER-TASK composer: at runtime an LLM
  composer picks each task's bundle by reading every variant's DESCRIPTION.
  The batch trajectories above were produced BEFORE this change existed, so a
  NEW variant/file can NEVER appear in them — "no bundle selected it" /
  "it was not active in these trajectories" is true of EVERY new variant by
  construction and is NOT evidence of ineffectiveness. Do not reject on it.
- Helper tools in `tools/helpers/*.py` are AUTO-DISCOVERED and injected into
  FUTURE solver prompts (see `tools/tmux_full.py:_discover_helper_tools`). The
  prompt visible in old trajectories predates the change — a new helper being
  absent there is expected, not "dead code". A helper is dead code only if it
  is broken, shadowed, or its trigger cannot occur.

# How to review

1. **Re-investigate the real trajectories** with `<read_file>` / `<grep>` on the
   trial paths above (`agent/episode-*/prompt.txt` = what the agent SAW,
   `response.txt` = what it DID, the pane; `result.json` / `verifier/` are
   blocked — you deliberately see NO pass/fail rewards: judge the change's
   MECHANISM, not any score). Also `<read_file>` the changed module file(s) in
   staging to see the change in context.
2. Then judge the change on TWO axes:
   - **Effective?** The standard depends on the change kind:
     * **MODIFIED an existing module** — is there a concrete causal link
       between the touched code path and an actual failure you can see in the
       trajectories? A change that targets something the trajectories don't
       show as a problem is NOT effective.
     * **NEW variant / new helper (ADD)** — it cannot have run yet (see
       Architecture facts), so judge three things instead: (i) the failure
       MODE it targets is actually visible in the trajectories, (ii) its
       mechanism would plausibly change that outcome, and (iii) its
       DESCRIPTION would lead the composer to select it on tasks resembling
       those failures. Reject if any of the three fails — never for "it did
       not run in this batch".
   - **Overfit to a specific task?** Does the change special-case one task
     rather than fix a general weakness? Red flags: a literal task name in the
     code, branching on one task's specific output/filenames, a constant/limit
     that looks tuned to make exactly one task pass, narrow special-casing.

The per-trial **MODULE TRACE** (kernel-recorded, shown in the index above) is
your primary runtime evidence: it lists every Protocol-method call each active
variant actually made and what it returned. For a MODIFY, a change to a method
the trace shows never firing on the failing tasks is not effective. A mechanism
a variant DECLARES but that never appears in the trace never ran.

# Decide (be adversarial — default to scrutiny, but only REJECT on real evidence)

- **REJECT** if and only if: the change is **overfit** to a specific task, OR
  it fails the kind-appropriate effectiveness standard above (causal link for
  a MODIFY; the three-part test for an ADD). An unjustified or task-specific
  change is worse than no change — it risks silent regression.
- Otherwise **ACCEPT**.

# Output (REQUIRED — this ACTION is the ONLY thing the harness reads)

Do NOT edit any files. When you are done investigating, submit your decision as
ONE structured action — free prose (including any `VERDICT: ...` line) is NOT
read and will NOT be recorded. The template (fill it in; the `|` alternatives
below make the template itself unparseable, so quoting it back cannot count as
a submission):

  <review_verdict decision="accept|reject"
                  reject_class="proposal|implementation"
                  reason="one sentence why"
                  repair_brief="only for reject_class implementation"/>

Fill-in rules — the submission counts only when the action EXECUTES (quoting
the tag in your analysis records nothing):

- `decision="accept"`: omit `reject_class` and `repair_brief` entirely.
- `decision="reject"` requires `reject_class`:
  - `"proposal"` — the DIRECTION is wrong: the mechanism targets a failure
    that isn't real / isn't in this batch, or is overfit by design.
  - `"implementation"` — the direction is sound but THIS code is flawed (a
    provable bug, a regression, a broken invariant). Add `repair_brief`
    describing only the code defect — never redesign the proposal in it.
- `reason` is always required: one sentence, inside the attribute.
- Protocol: emit the action, the harness acknowledges it ("review verdict
  recorded"), and the session closes on your next response — just stop after
  the acknowledgement. Do not emit `<commit_patch/>` or `<task_complete>`.

PARTIAL bundles: when only SOME files of a multi-file change are bad, reject
with `reject_class="implementation"` and name the bad file(s) in
`repair_brief` — file-level DROP partial accept is disabled until the
proposal manifest exists (it could retire the wrong parent variant at
promotion). A bad hunk inside an otherwise-good file is the same case.

# You are READ-ONLY here — do not edit, do not commit

This session decides; it does not change anything. Do NOT `<edit_file>`,
`<create_file>` or `<commit_patch/>` — there is nothing to commit, and the
harness reads only your `<review_verdict/>` action. Submit it once, then stop
unless the two-phase gate requests its one confirmation; do not otherwise
restate it "to be safe" or narrate what the harness might do.
"""


def build_review_instruction(
    diff_text: str,
    intent: str,
    trial_summaries: list[TrialSummary],
    max_msg_chars: int = 600,
) -> str:
    """Build the instruction for the editor self-review pass.

    Reward-blind by design: the review gate is part of the ACCEPTANCE decision,
    so the per-trial ground-truth verdicts shown to the reflecting editor
    (``include_reward=True``) must NOT reach it — it judges whether the change's
    mechanism is effective and not overfit, never the score.
    """
    trial_md = "\n\n".join(
        s.to_markdown(
            max_msg_chars=max_msg_chars, include_reward=False, include_trace=True
        )
        for s in trial_summaries
    )
    return _REVIEW_PREAMBLE.format(
        diff_text=diff_text or "(empty)",
        intent=(intent.strip() or "(the editor did not state an intent)"),
        trial_summaries=trial_md,
    )


#: How a finding's verdict came to be. This is NOT decoration: `is_culprit:
#: False` alone cannot tell "the investigator cleared the module" from "we could
#: not read what the investigator said", and those two must never be filed as
#: the same observation. Only PARSE_OK and PARSE_SALVAGED carry a verdict.
PARSE_OK = "ok"
PARSE_SALVAGED = "salvaged"  # strict JSON failed; fields recovered key-by-key
PARSE_FAILED = "failed"  # no verdict exists — do not read `is_culprit`

#: What may be a field name. Anything else quoted is a value, not a key.
_KEY_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _top_level_keys(raw: str) -> list[tuple[int, int, str]]:
    """`(key_start, value_start, name)` for each `"key":` at object depth 1.

    A regex cannot do this. The block reaching salvage is one an investigator
    broke by quoting JSON inside its own prose, so the text genuinely contains
    a second `,"is_culprit": false,` — at a comma, in an object, spelled
    identically to the real field. Matching on the boundary character alone
    hands that impostor to the caller as a top-level verdict.

    So walk the block instead, tracking two things a regex has no notion of:
    whether we are inside a string, and how deep we are nested (a lens
    finding's ``evidence: [{"task": ...}]`` is not the block's own ``task``).

    This is still best-effort. Once an unescaped `"` has broken a string,
    *nothing* can know where that string was meant to end — which is why
    :func:`loads_finding_block` also keeps the FIRST value for a repeated key
    rather than the last.
    """
    out: list[tuple[int, int, str]] = []
    depth = 0
    i = 0
    n = len(raw)
    while i < n:
        char = raw[i]
        if char == '"':
            start = i
            i += 1
            while i < n and raw[i] != '"':
                i += 2 if raw[i] == "\\" else 1
            name = raw[start + 1 : i]
            i += 1  # step past the closing quote (or past the end)
            if depth == 1 and _KEY_NAME_RE.fullmatch(name):
                after = i
                while after < n and raw[after].isspace():
                    after += 1
                if after < n and raw[after] == ":":
                    out.append((start, after + 1, name))
            continue
        if char in "{[":
            depth += 1
        elif char in "}]":
            depth -= 1
        i += 1
    return out


_JSON_LITERALS = {"true": True, "false": False, "null": None}
_JSON_ESCAPES = {'"': '"', "\\": "\\", "/": "/", "n": "\n", "t": "\t", "r": "\r"}


def _coerce_json_value(text: str):
    """Best-effort read of ONE `"key":` value lifted out of a broken object."""
    text = text.strip().rstrip(",").strip()
    if text in _JSON_LITERALS:
        return _JSON_LITERALS[text]
    try:  # numbers, plus any string/array/object that is locally well-formed
        return json.loads(text)
    except Exception:
        pass
    if not text.startswith('"'):
        return text
    # The damage this exists for: an unescaped `"` ended the string early, so
    # `json.loads` refuses it. Take the whole span and unescape in one pass —
    # left to right, so `\\"` stays a backslash followed by a quote.
    body = text[1:-1] if text.endswith('"') and len(text) > 1 else text[1:]
    return re.sub(
        r"\\(.)", lambda m: _JSON_ESCAPES.get(m.group(1), m.group(0)), body, flags=re.S
    )


def loads_finding_block(raw: str) -> tuple[dict | None, str]:
    """Parse a finding block, returning ``(finding, parse_status)``.

    Strict JSON first. If that fails, recover key by key rather than throwing
    the whole block away: the defect that motivated this cost a *correct*
    culprit verdict plus a concrete fix, because one stray `"` deep inside a
    `suggested_change` string invalidated 8775 otherwise well-formed characters
    and `is_culprit` sat three lines above the damage. Salvage is a guess, so it
    is labelled as one; only a block we cannot read at all reports PARSE_FAILED.
    """
    try:
        finding = json.loads(raw)
    except Exception:
        pass
    else:
        return (
            (finding, PARSE_OK) if isinstance(finding, dict) else (None, PARSE_FAILED)
        )

    keys = _top_level_keys(raw)
    if not keys:
        return None, PARSE_FAILED
    salvaged: dict = {}
    for i, (_, value_start, name) in enumerate(keys):
        if name in salvaged:
            # First wins. A repeated key means the scanner lost the thread of
            # where a string ended, and the real fields sit at the top of the
            # block while the prose that broke it sits below.
            continue
        stop = keys[i + 1][0] if i + 1 < len(keys) else None
        chunk = (
            raw[value_start:stop] if stop else raw[value_start:].rstrip().rstrip("}")
        )
        salvaged[name] = _coerce_json_value(chunk)
    return salvaged, PARSE_SALVAGED


# ===========================================================================
# Contrastive diagnosis (per-task pass-vs-fail), locked to ONE module type.
# The investigator does NOT do blame-attribution ("is this module the problem").
# It runs a COUNTERFACTUAL: would CHANGING this module actually move this task
# fail→pass? A module can be involved in a failure and still be the wrong place
# to fix it — only a change that would improve the OUTCOME counts. If no such
# change exists, report no culprit and don't fix (avoids result-oriented, prompt-
# tweaking, marginal edits that regress downstream).
# ===========================================================================

# Each locked module type gets its own responsibility, strong-agent yardstick,
# and diagnostic probes. Observation and context management remain separate
# because seeing information and retaining it are different failure surfaces.
_MODULE_LENS: dict[str, dict] = {
    "agent_loop": {
        "covers": (
            "the think→act→observe control loop: how the agent plans, recovers "
            "from errors, and structures multi-step work."
        ),
        "yardstick": (
            "Strong agents plan / track TODOs for multi-stage work; on an error "
            "they READ it before retrying (never blind-retry the same command); "
            "they balance explore-vs-execute and stay aware of their progress.\n"
            "Ours is a fixed ReAct loop with none of that built in."
        ),
        "probes": (
            "Did a multi-stage task proceed with no plan? Did it hit the same "
            "failing command again and again? Get stuck unable to leave a state, "
            "or give up with budget left?"
        ),
    },
    "tools": {
        "covers": (
            "the agent's action surface: the tools it has (and whole capability "
            "classes it may lack)."
        ),
        "yardstick": (
            "Strong agents have structured file tools — Read with line numbers, "
            "exact-match atomic Edit, Write, Grep, Glob.\n"
            "Ours has a single tmux tool: every edit goes through raw shell "
            "(sed / heredoc), every inspection through cat/grep in a scrolling "
            "terminal."
        ),
        "probes": (
            "Did the agent burn steps hand-editing with sed/heredoc and get it "
            "wrong? Fight the terminal to inspect a file? Lack a tool that would "
            "have made the fix one step?"
        ),
    },
    "observation": {
        "covers": (
            "what the agent gets to SEE — how raw terminal output becomes the "
            "percept the model actually reads."
        ),
        "yardstick": (
            "Strong agents perceive reliably: errors and exit codes reach the "
            "model untruncated, key output stays legible, nothing important is "
            "silently cut.\n"
            "Ours feeds truncated terminal output as the only percept."
        ),
        "probes": (
            "Did truncation / noise swallow an error message, a path, an exit "
            "code, or part of the task spec? Did the agent act on a stale or "
            "garbled observation?"
        ),
    },
    "context_mgmt": {
        "covers": (
            "what the agent RETAINS across steps — when and what to compress as "
            "the conversation grows."
        ),
        "yardstick": (
            "Strong agents keep the facts that matter across many steps and "
            "compress with economy that never drops signal.\n"
            "Ours summarizes chat history when context grows, with no guard on "
            "what it throws away."
        ),
        "probes": (
            "Did a summary compress away an error / path / spec at a bad moment? "
            "Did the agent re-discover a fact it already had? Did context "
            "overflow force a lossy compression right when it needed that state?"
        ),
    },
    "verification": {
        "covers": (
            "when the agent decides it is DONE — too early, too late, or without "
            "checking."
        ),
        "yardstick": (
            "Strong agents finish on EVIDENCE: before declaring done they check "
            "artifacts exist, tests pass, output is complete; they neither stop "
            "early nor spin after the work is done.\n"
            "Ours does a two-phase self-assessment (the model saying it is "
            "confident) with no required check against reality."
        ),
        "probes": (
            "Did the agent declare done on a task that was actually FAILED "
            "(artifact missing, tests never run, output incomplete)? Did it loop "
            "long after the outcome was already decided?"
        ),
    },
}


def _module_lens(locked: str) -> dict:
    return _MODULE_LENS.get(
        locked, {"covers": f"the `{locked}` module.", "yardstick": "", "probes": ""}
    )


_CONTRAST_INVESTIGATION_PREAMBLE = """\
You are a READ-ONLY diagnostician deciding ONE thing: **would changing the
`{locked}` module actually CHANGE or IMPROVE this task's outcome?** You get the
SAME task solved on one roll and FAILED on another (same code, different
sampling).

This is NOT blame-attribution. Do not report "`{locked}` is involved" — a module
can play a part in a failure and still be the WRONG place to fix it. Report a
culprit ONLY if there is a concrete `{locked}` change that would plausibly move
THIS task from fail toward pass. If no `{locked}` change would help — even if
`{locked}` is "involved" — the answer is no culprit, and you do NOT fix.

# Your module: `{locked}`
{covers}

## Yardstick — so you can SEE a gap our agent can't see about itself
{yardstick}

Use it as a lens to look THROUGH, not a checklist to fill: report a gap ONLY when
the two rolls actually show it decided THIS task. Starter probes (suggestions,
not a form): {probes}

# The task: {task}
## PASSING roll{pass_origin}
{pass_block}

## FAILING roll
{fail_block}

You know each verdict — NOT the verifier's tests/expected outputs (blocked).
Investigate the raw rolls with `<read_file>` / `<grep>`:
`agent/episode-*/prompt.txt` = what the agent SAW, `response.txt` = what it DID,
`agent/terminus_2_modular.pane` = the raw terminal. The MODULE TRACE in each
index entry records every Protocol-method call each active variant made. The
`{locked}` module code you may propose changing is readable at `{locked}/`.

# How to decide (in order)
1. **Divergence (this is EVIDENCE, not yet a verdict)**: where did the pass and
   fail rolls stop behaving the same? What did the PASSING roll do there that the
   FAILING roll didn't (or did wrong)? Cite episodes/steps.
2. **The counterfactual (this IS the verdict)**: picture a concrete change to
   `{locked}` — the specific thing our `{locked}` lacks vs the yardstick above.
   Ask: *with that change in place, would the failing roll plausibly have reached
   what the passing roll reached — i.e. this task moves fail → pass (or clearly
   closer)?*
   - **YES, and it fits inside `{locked}`'s own code** → `"is_culprit": true`,
     `"fixable_now": true`; describe exactly that change.
   - **YES, but it needs more than `{locked}` alone** — a new Protocol method, a
     kernel / composer / library change so `{locked}` can even do it → this is a
     MODULE-ONLY run (the kernel is frozen this experiment), so `"is_culprit":
     true`, `"fixable_now": false`, and say in `suggested_change` exactly what
     architecture change it needs. Do NOT fake it inside `{locked}` alone — it
     becomes backlog for a kernel-evolution round.
   - **NO** — it was model luck, OR the real leverage is a DIFFERENT module, OR
     it is beyond reach → `"is_culprit": false`. If another module is where the
     fix belongs, name it in `other_module` (a lead for that module's own
     experiment). **Do NOT invent a marginal `{locked}` tweak just to have a
     finding: a change that would not move the outcome is a regression waiting to
     happen — no culprit is the correct, productive answer.**

# REQUIRED output — end your final message with exactly one block:

<contrast_finding>
{{
  "task": "{task}",
  "is_culprit": true or false,
  "locked_module": "{locked}",
{evidence_field}
  "would_change_outcome": "the counterfactual: what `{locked}` change, and why it would move THIS task fail→pass (or why nothing would — then is_culprit is false)",
  "fixable_now": true or false,
  "suggested_change": "the concrete NEW `{locked}` variant to build (only if is_culprit)",
  "other_module": "only if is_culprit=false and another module is where the real fix belongs"
}}
</contrast_finding>

With `"is_culprit": false`, only "task", "is_culprit" and an optional "note" /
"other_module" are needed. Then `<task_complete>true</task_complete>`. Do NOT
edit any file — you are diagnosis only.
"""


_DIVERGENCE_FIELD = (
    '  "divergence": "where/how the pass and fail rolls split, with step refs",'
)


def build_contrast_investigation_instruction(
    locked_module: str,
    task: str,
    failing: TrialSummary,
    passing: TrialSummary | None,
    *,
    pass_from_history: bool = False,
    max_msg_chars: int = 800,
) -> str:
    """Instruction for ONE read-only per-task contrast investigator.

    `passing` is the matched successful roll (same task). It may be None (a
    fixable-fail with no pass available this epoch) — then it is a single-sided
    diagnosis. `pass_from_history=True` labels a pass pulled from the archive /
    a past epoch rather than this batch.
    """
    fail_block = failing.to_markdown(
        max_msg_chars=max_msg_chars, include_reward=True, include_trace=True
    )
    if passing is not None:
        pass_block = passing.to_markdown(
            max_msg_chars=max_msg_chars, include_reward=True, include_trace=True
        )
        pass_origin = " (from an earlier generation)" if pass_from_history else ""
    else:
        pass_block = (
            "(no passing roll available this epoch — diagnose the failure alone: "
            "what in `%s` kept it from finishing?)" % locked_module
        )
        pass_origin = " — NONE this epoch"

    lens = _module_lens(locked_module)
    return _CONTRAST_INVESTIGATION_PREAMBLE.format(
        evidence_field=_DIVERGENCE_FIELD,
        locked=locked_module,
        covers=lens["covers"],
        yardstick=lens["yardstick"],
        probes=lens["probes"],
        task=task,
        pass_origin=pass_origin,
        pass_block=pass_block,
        fail_block=fail_block,
    )


_EFFICIENCY_INVESTIGATION_PREAMBLE = """\
You are a READ-ONLY diagnostician. This task PASSED — but wastefully (too many
turns / near the time budget / repeated or dead-end commands). Your job: find
WHERE the work was wasted, and whether the **`{locked}`** module (the only module
this experiment may change) could make the agent LEANER **without risking
correctness**.

# The task: {task} (passed, but wasteful)
{roll_block}

Investigate with `<read_file>` / `<grep>`: `agent/episode-*/prompt.txt` = what
the agent SAW, `response.txt` = what it DID, `agent/terminus_2_modular.pane` =
raw terminal. The MODULE TRACE shows every Protocol-method call. `{locked}/` is
readable.

# What to look for
1. The waste: repeated failed commands, re-reading the same files, thrashing,
   redundant verification, an over-long loop. Cite episodes/steps.
2. Attribution to `{locked}`: could a `{locked}` change have cut the waste (e.g.
   agent_loop: stop retrying a dead approach sooner; verification: don't re-check
   what's already proven; context_mgmt: keep the right state so it doesn't redo
   work)?
3. **Correctness first**: an efficiency change must NOT risk breaking tasks that
   currently pass. If the only way to trim is to cut a step that other tasks
   need, set `"is_culprit": false` and say so — a fragile speedup is a
   regression, not a win.

# REQUIRED output — end with exactly one block:

<contrast_finding>
{{
  "task": "{task}",
  "is_culprit": true or false,
  "locked_module": "{locked}",
  "divergence": "where the work was wasted, with step refs",
  "gap": "the `{locked}` behavior that, if leaner, would cut the waste safely",
  "fixable_now": true or false,
  "suggested_change": "a NEW `{locked}` variant that trims waste without risking correctness"
}}
</contrast_finding>

With `"is_culprit": false`, only "task", "is_culprit" and a "note" are needed.
Then `<task_complete>true</task_complete>`. Do NOT edit any file.
"""


def build_efficiency_investigation_instruction(
    locked_module: str,
    task: str,
    wasteful_roll: TrialSummary,
    *,
    max_msg_chars: int = 800,
) -> str:
    """Instruction for a per-task EFFICIENCY investigator (all_pass_wasteful):
    diagnose where a passing-but-wasteful roll burned effort and whether the
    locked module can trim it safely. Emits the same <contrast_finding> block as
    the correctness investigator, so parse_contrast_finding handles both."""
    roll_block = wasteful_roll.to_markdown(
        max_msg_chars=max_msg_chars, include_reward=True, include_trace=True
    )
    return _EFFICIENCY_INVESTIGATION_PREAMBLE.format(
        locked=locked_module, task=task, roll_block=roll_block
    )


_CONTRAST_FINDING_RE = re.compile(
    r"<contrast_finding>\s*(.*?)\s*</contrast_finding>", re.DOTALL | re.IGNORECASE
)


def parse_contrast_finding(trajectory_path: Path | None, task: str) -> dict:
    """Extract the last contrast finding, or return an explicit no-verdict."""
    null = {
        "task": task,
        "is_culprit": False,
        "note": "no finding parsed",
        "parse_status": PARSE_FAILED,
    }
    if trajectory_path is None:
        return null
    tp = Path(trajectory_path)

    def _last_block(texts: list[str]) -> str | None:
        found: str | None = None
        for text in texts:
            for m in _CONTRAST_FINDING_RE.finditer(text):
                found = m.group(1)
        return found

    episode_texts: list[str] = []
    ep_dir = tp.parent
    if ep_dir.is_dir():

        def _ep_num(p: Path) -> int:
            try:
                return int(p.parent.name.rsplit("-", 1)[-1])
            except ValueError:
                return 1 << 30

        for rt in sorted(ep_dir.glob("episode-*/response.txt"), key=_ep_num):
            try:
                episode_texts.append(rt.read_text())
            except Exception:
                continue
    raw = _last_block(episode_texts)

    if raw is None and tp.exists():
        try:
            data = json.loads(tp.read_text())
            step_texts: list[str] = []
            for s in data.get("steps", []):
                if s.get("source") != "agent":
                    continue
                for key in ("message", "reasoning_content"):
                    v = s.get(key) or ""
                    if isinstance(v, list):
                        v = " ".join(str(p) for p in v)
                    if v:
                        step_texts.append(str(v))
            raw = _last_block(step_texts)
        except Exception:
            pass
    if not raw:
        return null
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    finding, status = loads_finding_block(raw)
    if finding is None:
        _logger.warning(
            "contrast finding for `%s` was unreadable — filed as NO VERDICT, "
            "not as 'not the culprit'; first 300 chars: %s",
            task,
            raw[:300],
        )
        return {**null, "note": "finding block was unreadable", "raw": raw[:2000]}
    if status == PARSE_SALVAGED:
        verdict = finding.get("is_culprit")
        if not isinstance(verdict, bool):
            # A recovered fragment is not a verdict; coercing it with bool()
            # would promote any non-empty fragment to "culprit".
            _logger.warning(
                "contrast finding for `%s` salvaged a non-boolean is_culprit "
                "(%r) — filed as NO VERDICT, not as a culprit",
                task,
                verdict,
            )
            return {**null, "note": "salvaged verdict was not a boolean"}
        _logger.warning(
            "contrast finding for `%s` was not valid JSON; recovered %d "
            "field(s) key-by-key — verdict is_culprit=%s",
            task,
            len(finding),
            verdict,
        )
    finding.setdefault("task", task)
    finding["is_culprit"] = bool(finding.get("is_culprit"))
    # The backlog uses the module type as its clustering lens.
    finding.setdefault("lens", finding.get("locked_module", "contrast"))
    finding["parse_status"] = status
    return finding


_SANITY_REPAIR_PREAMBLE = """\
A change you just made to the agent's modules PASSED review (the idea was judged
sound) but then CRASHED at runtime on a fixed sanity task. Only a code BUG needs
fixing — keep the original intent and approach, do NOT redesign or remove the
change, do NOT weaken it to dodge the error.

# The change (diff vs the parent gen)
```diff
{diff}
```

# Where it crashed
{crashes}

# How to work
1. Read the traceback — it points at the line in YOUR new/edited code that threw.
   The "active variant" line tells you WHICH module file was running when it
   crashed (the per-task Composer selected it for that task).
2. Open that file (`<read_file>`) and fix the actual bug — a wrong attribute or
   arg name, a None access, a bad import, a signature mismatch. Stay in `modules/`.
3. Fix the BUG so the original behavior works; do NOT change the approach.
4. `<validate/>`, `<commit_patch/>`, `<task_complete>true</task_complete>`.
   If it genuinely cannot be fixed without changing the approach, commit with NO
   edits and say so — the candidate will be discarded.
"""


def build_sanity_repair_instruction(
    crashes: list[dict],
    diff_text: str,
    max_tb_chars: int = 2000,
    max_diff_chars: int = 6000,
) -> str:
    """Instruction to REPAIR a candidate that passed review but crashed in the
    sanity gate. ``crashes`` = list of dicts with task / exception_type /
    traceback / bundle (the variant that was active when it crashed)."""
    blocks: list[str] = []
    for c in crashes:
        tb = (c.get("traceback") or "").strip()
        if len(tb) > max_tb_chars:
            tb = tb[-max_tb_chars:]
        bundle = c.get("bundle") or {}
        active = (
            " ".join(f"{k}={v}" for k, v in bundle.items()) if bundle else "(unknown)"
        )
        blocks.append(
            f"## task `{c.get('task')}` — {c.get('exception_type') or 'crash'}\n"
            f"active variant: {active}\n"
            f"```\n{tb or '(no traceback captured)'}\n```"
        )
    d = (
        diff_text
        if len(diff_text) <= max_diff_chars
        else diff_text[:max_diff_chars] + "\n… [diff truncated]"
    )
    return _SANITY_REPAIR_PREAMBLE.format(diff=d, crashes="\n\n".join(blocks))


_SMOKE_REPAIR_PREAMBLE = """\
A change you just made to the agent's modules FAILED to even LOAD (the smoke
check: AST parse / import / library registration / Protocol conformance). The
modules cannot be used until everything loads cleanly. Fix the loading BUG — keep
the intent of your change, do NOT redesign or remove it.

# The change (diff vs the parent gen)
```diff
{diff}
```

# What failed to load
{failures}

# How to work
1. Read the failure(s) — they name the file and the error (ImportError, syntax
   error, missing register()/Protocol method, a relative import).
2. COMMON CAUSE: a new module file is loaded from a gen path with NO parent
   package, so **relative imports (`from . import x`, `from ..y import z`) fail**.
   Use ABSOLUTE imports (`from harbor.agents.terminus_2_modular... import ...`) or
   `importlib.import_module(...)`. Also ensure the file defines `register(library)`
   and its class satisfies the module Protocol.
3. Fix the bug so EVERY file under `modules/` loads. Stay inside `modules/`.
4. `<validate/>`, `<commit_patch/>`, `<task_complete>true</task_complete>`.
   If it genuinely cannot be fixed, commit with NO edits — it will be discarded.
"""


def build_smoke_repair_instruction(
    failures: list[str], diff_text: str, max_diff_chars: int = 6000
) -> str:
    """Instruction to REPAIR a candidate that failed the smoke (load) gate —
    import / syntax / registration / Protocol errors. ``failures`` =
    ``smoke.all_failures()``."""
    fl = "\n".join(f"- {f}" for f in failures) or "(no detail)"
    d = (
        diff_text
        if len(diff_text) <= max_diff_chars
        else diff_text[:max_diff_chars] + "\n… [diff truncated]"
    )
    return _SMOKE_REPAIR_PREAMBLE.format(diff=d, failures=fl)


_DROP_LINE_RE = re.compile(r"^[ \t>*-]*DROP:\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE)


def _parse_drop_list(raw: str) -> list[str]:
    """Split a `DROP:` line's payload into normalized module .py paths.

    Strips backticks/quotes and NEW:/MODIFIED: diff prefixes; keeps only tokens
    that look like a module file (`<...>.py`), so a quoted format placeholder
    (`<file>`) or prose ("none") yields nothing."""
    out: list[str] = []
    for tok in re.split(r"[,\s]+", raw.strip()):
        tok = tok.strip().strip("`'\"")
        for pfx in ("NEW:", "MODIFIED:", "MOD:", "M:"):
            if tok.upper().startswith(pfx):
                tok = tok[len(pfx) :]
        tok = tok.strip().lstrip("/")
        if tok.endswith(".py") and "/" in tok and ".." not in tok:
            out.append(tok)
    # de-dup, preserve order
    seen: set[str] = set()
    return [t for t in out if not (t in seen or seen.add(t))]


def parse_review_drop(trajectory_path: Path | None) -> list[str]:
    """Extract the review editor's DROP list (files to revert on ACCEPT_PARTIAL).

    Scans the reviewer's agent messages (message + reasoning, chronological) for
    a line beginning `DROP:` and returns the LAST such list — the reviewer's
    final decision. Only honored by the caller when the verdict is ACCEPT. Empty
    when absent / unreadable / no real .py path is named (so a format-spec quote
    of the `DROP:` line cannot cause a spurious revert)."""
    if trajectory_path is None or not Path(trajectory_path).exists():
        return []
    try:
        data = json.loads(Path(trajectory_path).read_text())
    except Exception:
        return []
    last: list[str] = []
    for s in data.get("steps", []):
        if s.get("source") != "agent":
            continue
        chunks: list[str] = []
        msg = s.get("message") or ""
        if isinstance(msg, list):
            msg = " ".join(str(p) for p in msg)
        if isinstance(msg, str) and msg.strip():
            chunks.append(msg)
        rc = s.get("reasoning_content") or ""
        if isinstance(rc, str) and rc.strip():
            chunks.append(rc)
        for text in chunks:
            for m in _DROP_LINE_RE.finditer(text):
                files = _parse_drop_list(m.group(1))
                if files:
                    last = files
    return last


def extract_review_reasoning(
    trajectory_path: Path | None, max_chars: int = 2500
) -> str:
    """The review editor's final conclusion text (last couple of agent messages).

    Human-readable context for logs/memory (for example, an editor memo). It is
    NEVER interpreted as a verdict — that is exclusively the structured
    `<review_verdict/>` action (self_evo/review_verdict.py). Returns "" if
    there is no trajectory to read.
    """
    if trajectory_path is None or not Path(trajectory_path).exists():
        return ""
    try:
        data = json.loads(Path(trajectory_path).read_text())
    except Exception:
        return ""
    msgs: list[str] = []
    for s in data.get("steps", []):
        if s.get("source") != "agent":
            continue
        m = s.get("message") or ""
        if isinstance(m, list):
            m = " ".join(str(p) for p in m)
        if isinstance(m, str) and m.strip():
            msgs.append(m.strip())
    if not msgs:
        # Reasoning models leave `message` empty — give the classifier the
        # tail of the last reasoning chain instead of nothing.
        for s in reversed(data.get("steps", [])):
            if s.get("source") != "agent":
                continue
            rc = s.get("reasoning_content") or ""
            if isinstance(rc, str) and rc.strip():
                return rc.strip()[-max_chars:]
        return ""
    return "\n\n".join(msgs[-2:])[-max_chars:]
