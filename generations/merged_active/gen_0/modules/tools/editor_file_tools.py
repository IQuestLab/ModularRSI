"""Editor tools module — for self-evo "editor" role.

This is the tools surface that terminus-2 uses when it's acting as an EDITOR
(modifying another generation's modules/) rather than a SOLVER (running
terminal-bench tasks).

Actions use the following XML grammar:

    <response>
      <analysis>...</analysis>
      <plan>...</plan>
      <actions>
        <read_file path="path/relative/to/staging">
        </read_file>

        <edit_file path="path/relative/to/staging">
          <old_string>exact string to find (must exist, must be unique)</old_string>
          <new_string>replacement string</new_string>
        </edit_file>

        <create_file path="path/relative/to/staging">
          <content>full file contents here</content>
        </create_file>

        <commit_patch/>
      </actions>
      <task_complete>false</task_complete>
    </response>

After `commit_patch`, the editor is signaling "I am done modifying; please run
smoke tests on staging now". The agent_loop translates this into a
task_complete signal so two-phase verification can wrap things up.

WRITE operations (edit_file / edit_lines / create_file) are restricted to
`ctx.state.staging_dir` — paths outside are rejected. READ operations
(read_file / grep) may additionally access (read-only) `ctx.state.trajectory_root`
when set — the batch's solver output dir, so the editor can investigate the raw
trajectories (episode prompt/response, terminal panes) itself — and
`ctx.state.skills_dir`, so it can read the BODY of a skill listed in
`<available_skills>`. The editor cannot delete files, write outside the staging
dir, or run shell commands.
"""

from __future__ import annotations

import re
from pathlib import Path

from harbor.agents.terminus_2_modular.protocols import (
    LLMResponseParseResult,
    ModuleCtx,
    ToolCall,
    ToolResult,
)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """\
You are an AI agent acting as an EDITOR for another agent's source code. \
You are looking at a snapshot of the agent's editable source tree and your \
job is to propose edits that will help it solve more tasks correctly on the \
benchmark.

Respond ONLY in XML with this structure:

<response>
<analysis>
What you've learned from the parent's trajectories / current code. \
What's the failure mode or missed opportunity?
</analysis>
<plan>
What change you'll make and why. Be specific about which file + region.
</plan>
<actions>
  <!-- Zero or more of these, in order. Recommended loop:
       locate (grep/glob) -> read -> edit -> VALIDATE -> fix -> commit_patch -->

  <!-- List files by glob pattern: -->
  <glob pattern="agent_loop/*.py"/>

  <!-- Search all module .py files. context=N shows N lines around each match.
       USE grep to confirm where something is DEFINED and CALLED before claiming
       it is unused — read_file truncates large files, so a call site may hide. -->
  <grep pattern="PARAMS_SCHEMA" context="3"/>

  <!-- The placeholders below show the ACTION SYNTAX only — they are NOT a
       suggested change. What to edit, and in which module, follows from your
       diagnosis of the actual trajectories. -->
  <read_file path="<type>/<impl>.py"/>
  <!-- Large files are truncated in the middle; read a specific line range: -->
  <read_file path="<type>/<impl>.py" start_line="380" end_line="440"/>

  <!-- Edit by unique string match (the strings are placeholders for whatever
       exact snippet your diagnosis calls for — NOT a suggested kind of edit): -->
  <edit_file path="<type>/<impl>.py">
    <old_string>...exact text to find...</old_string>
    <new_string>...replacement text...</new_string>
  </edit_file>

  <!-- OR edit by line numbers (more robust on big files; get the numbers from a
       line-numbered read_file). Replaces lines start_line..end_line inclusive: -->
  <edit_lines path="<type>/<impl>.py" start_line="405" end_line="406">
    <new_string>        ...replacement lines...</new_string>
  </edit_lines>

  <create_file path="<type>/<your_impl>.py">
    <content>... full file ...</content>
  </create_file>

  <!-- Self-check your staged changes (AST + import + library load + bundle
       config) WITHOUT committing. Run this after editing and BEFORE
       commit_patch; fix anything it reports, then re-validate. -->
  <validate/>
{self_test_grammar}
  <!-- Only after <validate/> passes — signal completion: -->
  <commit_patch/>
</actions>
<task_complete>false</task_complete>
</response>

Rules:

- All `path` attributes are RELATIVE to the staging directory; do NOT use \
absolute paths or `..`.
- `<edit_file>` requires `<old_string>` to occur EXACTLY ONCE in the file. \
If it doesn't exist, you get an error and should retry with more context. \
If it occurs multiple times, you must include more surrounding text to \
disambiguate.
- `<create_file>` only works if the file doesn't exist yet.
- `<read_file>` shows file content WITH line numbers; use it freely. Files over \
~32k chars are truncated in the MIDDLE — if you see a "[lines X-Y omitted]" \
marker, that region exists but is hidden; read it with the optional \
`start_line`/`end_line` attributes, or use `<grep>`.
- `<grep pattern="..." [path="..."] [context="N"]>` searches all `.py` files \
under staging, returns matching `path:line: text` (context=N shows N lines \
around each match, the match line marked with `>`). Use it to find where a \
symbol is DEFINED and where it is CALLED. NEVER claim code is "dead" / "never \
called" from a single `<read_file>` — large files are truncated, the call site \
may be off-screen. grep first.
- `<glob pattern="..."/>` lists files matching a glob (e.g. `agent_loop/*.py`, \
`**/*.py`). Use it to discover what modules exist.
- The `<available_skills>` block lists reference skills as name + description + \
`<location>`. The listing is only a summary — `<read_file path="<location>"/>` a \
relevant skill's SKILL.md for its full details before relying on it.
- `<edit_lines path="..." start_line="N" end_line="M">` replaces lines N..M \
(1-based, inclusive) with `<new_string>`. More robust than edit_file on big \
files — read with line numbers first to get N and M.
- `<validate/>` self-checks your staged changes (AST parse + import + library \
load + bundle config) WITHOUT committing, and returns PASS or the exact errors. \
ALWAYS run `<validate/>` after editing and fix everything it reports BEFORE \
`<commit_patch/>`. This is your inner loop: edit → validate → fix → re-validate \
→ commit. (The harness also re-runs these checks after you commit, but catching \
problems here saves a wasted generation.)
{kernel_rules}
- Emit `<commit_patch/>` only after `<validate/>` passes. After commit, the \
harness re-runs its gates on your patched staging (smoke + a review pass + a \
crash check).
- Set `<task_complete>true</task_complete>` only AFTER you emit \
`<commit_patch/>` and want the loop to end.

Task description (what to improve):
{instruction}

Initial state (output of `ls staging/`):
{terminal_state}
"""

# The Kernel-boundary rule line, per staging-root kind. Chosen at prompt time
# by _staging_root_kind — the flag never travels through module params.
_KERNEL_RULES_MODULES_ROOT = (
    "- You cannot delete files, rename files, or change the Kernel files "
    "(protocols.py, services.py, agent.py, library.py, composer/). These are "
    "off-limits."
)
_KERNEL_RULES_PACKAGE_ROOT = (
    "- This staging is a WHOLE-PACKAGE snapshot: you may edit "
    "`protocols.py`, `library.py`, `composer/`, `kernel/` (the task "
    "orchestration) and `modules/` — architecture-level changes (new module "
    "slots, new Protocol capabilities, orchestration restructuring) are in "
    "scope. If you change a Protocol's surface, you MUST migrate every "
    "affected module in the SAME commit and make `<validate/>` pass — its "
    "conformance layer instantiates every module against YOUR edited "
    "protocols and rejects the patch otherwise.\n"
    "- FROZEN regardless: `modules/tools/editor_file_tools.py` (the editor's "
    "own tooling), `modules/__init__.py`, and everything not in the snapshot "
    "(services.py, agent.py, the evaluation harness). You cannot delete or "
    "rename files."
)


def _self_test_grammar(ctx: ModuleCtx) -> str:
    """The `<run_task>` grammar block for the editor's system prompt — emitted
    only when self-test is enabled for this run (else empty; don't advertise a
    disabled tool)."""
    cfg = getattr(getattr(ctx, "state", None), "self_test", None)
    if cfg is None or not getattr(cfg, "enabled", False):
        return ""
    n = getattr(cfg, "max_self_tests", 0)
    return (
        "  <!-- OPTIONAL self-test (budget: %d run(s) this session). Runs your\n"
        "       CURRENT staged code on a FIXED simple task in a real sandbox and\n"
        "       reports whether it RAN or CRASHED (+ a folder to read: job.log,\n"
        "       traceback, terminal pane). It is a runtime smoke of YOUR code —\n"
        "       'does it still run?' — NOT a way to test a change's effect on a\n"
        "       task, so you do NOT pick the task. Use it AFTER editing and BEFORE\n"
        "       commit_patch; if it crashed, read the error, fix, and run again. -->\n"
        "  <run_task/>"
    ) % n


_RESP = re.compile(r"<response>(.*?)</response>", re.DOTALL | re.IGNORECASE)
_DONE = re.compile(
    r"<task_complete>\s*(true|false)\s*</task_complete>",
    re.IGNORECASE,
)
_ANALYSIS = re.compile(r"<analysis>(.*?)</analysis>", re.DOTALL | re.IGNORECASE)
_PLAN = re.compile(r"<plan>(.*?)</plan>", re.DOTALL | re.IGNORECASE)
_ACTIONS = re.compile(r"<actions>(.*?)</actions>", re.DOTALL | re.IGNORECASE)

# Robust action extraction (does NOT require an <actions> wrapper, and does NOT
# feed edit bodies — which contain code with < > & — through an XML parser).
# Container actions carry a body (old_string / new_string / content) read as
# RAW TEXT; self-closing actions carry only attributes.
_CONTAINER_RE = re.compile(
    r"<(edit_file|edit_lines|create_file)\b([^>]*)>(.*?)</\1>",
    re.DOTALL | re.IGNORECASE,
)
_SELFCLOSE_RE = re.compile(
    r"<(read_file|grep|glob|validate|commit_patch|archive|run_task)\b([^>]*?)/?>",
    re.IGNORECASE,
)
_ATTR_RE = re.compile(r'([\w-]+)\s*=\s*"([^"]*)"')


def _sub_text(inner: str, tag: str) -> str:
    """Extract <tag>...</tag> body from `inner` as raw text (DOTALL)."""
    m = re.search(rf"<{tag}\s*>(.*?)</{tag}>", inner, re.DOTALL | re.IGNORECASE)
    return m.group(1) if m else ""


# ---------------------------------------------------------------------------
# Off-limits paths — relative to staging_dir. TWO staging roots exist:
#
# - "modules" root: staging_dir IS the modules/ tree. Kernel files
#   are off-limits by top-level name; the watchman (this file) by exact path.
# - "package" root: staging_dir is a whole-package
#   snapshot (protocols.py / library.py / composer/ / kernel/ / modules/).
#   ALLOWLIST discipline: only those five top-level entries are touchable at
#   all (an unknown top-level name is rejected — additions to the snapshot
#   surface must be made deliberately, in staging.PACKAGE_SNAPSHOT_ITEMS and
#   here together), minus the exact-path freeze set.
#
# The freeze set is the evaluator boundary: this file is the gatekeeper that
# enforces these rules and the answer-key blocks — an editor allowed to edit
# it could rewrite its own guardrails. services.py / agent.py / self_evo are
# frozen structurally (never copied into a snapshot at all).
# ---------------------------------------------------------------------------

_OFF_LIMITS_REL = {
    "protocols.py",
    "services.py",
    "agent.py",
    "library.py",
    "__init__.py",
    "composer",  # the entire composer/ directory
}

# Exact-path off-limits: enforcement infrastructure that lives INSIDE the
# editable modules/ tree ("modules" root form).
_OFF_LIMITS_EXACT = {
    ("tools", "editor_file_tools.py"),
}

# Package-root allowlist + freeze. Keep the allowlist in sync with
# staging.PACKAGE_SNAPSHOT_ITEMS.
_PACKAGE_ALLOWED_TOP = {"protocols.py", "library.py", "composer", "kernel", "modules"}
_PACKAGE_FREEZE_EXACT = {
    ("modules", "tools", "editor_file_tools.py"),  # the watchman (this file)
    ("modules", "__init__.py"),  # mirrors the legacy modules-root __init__ freeze
}


def _staging_root_kind(staging_dir: Path) -> str:
    """ "package" when the staging root is a whole-package snapshot, else
    "modules". Safe: the classic smoke check forbids a protocols.py inside a
    modules tree, so the marker cannot be spoofed from a legacy staging."""
    return "package" if (Path(staging_dir) / "protocols.py").is_file() else "modules"


def _is_off_limits(rel_path: str, root_kind: str = "modules") -> bool:
    parts = Path(rel_path).parts
    if not parts:
        return True
    if root_kind == "package":
        if parts[0] not in _PACKAGE_ALLOWED_TOP:
            return True
        return tuple(parts) in _PACKAGE_FREEZE_EXACT
    return parts[0] in _OFF_LIMITS_REL or tuple(parts) in _OFF_LIMITS_EXACT


def _write_allowed(rel_path: str, locked_type: str | None, root_kind: str) -> bool:
    """Single-module ablation lock (self-evo): when ``locked_type`` is set, WRITE
    actions may only touch files under that one module type's directory —
    ``modules/<locked>/`` in a package root, ``<locked>/`` in a modules root.
    None/"" = no lock. This is WRITE-ONLY on purpose: reads/grep stay unrestricted
    so the editor can still study the other modules for context. Do NOT fold this
    into ``_is_off_limits`` (that one also gates reads).

    Independently of the lock, every module type's shared parent ``baseline.py``
    is read-only: variants must be ADDED or MERGED on top of it, never mutated in
    place. This is a structural invariant, NOT an ablation constraint, so it holds
    with the lock off too (a free/multi-module run must not rewrite the base
    either). Rationale: an in-place edit to the base is un-benched and
    un-rollbackable — the Composer has no old-vs-new pair to A/B, and cross-epoch
    confirm keys on ``(type, variant_name)``, which an in-place edit leaves
    unchanged, so a regression can never be attributed or superseded."""
    # Shared parent: refused ALWAYS, lock or no lock. Checked first on purpose.
    if Path(rel_path).name == "baseline.py":
        return False
    if not locked_type:
        return True
    # Solver helpers are the `tools` type's extension point but live in their own
    # top-level dir (`modules/tool_helper/`), so a tools-locked lineage must be
    # able to write there — otherwise it could add no new agent action at all.
    allowed = {locked_type}
    if locked_type == "tools":
        allowed.add("tool_helper")
    parts = Path(rel_path).parts
    if root_kind == "package":
        in_locked = len(parts) >= 2 and parts[0] == "modules" and parts[1] in allowed
    else:
        in_locked = len(parts) >= 1 and parts[0] in allowed
    return in_locked


def _parse_skill_frontmatter(content: str) -> dict[str, str] | None:
    """Parse YAML-like frontmatter at the start of a SKILL.md.

    Recognized keys: `name`, `description`. Other keys ignored. Returns None
    if frontmatter is missing or doesn't have both required keys.
    """
    if not content.startswith("---"):
        return None
    lines = content.splitlines()
    if len(lines) < 3:
        return None
    end = -1
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end = i
            break
    if end == -1:
        return None
    name: str | None = None
    description: str | None = None
    for raw in lines[1:end]:
        if ":" not in raw:
            continue
        k, _, v = raw.partition(":")
        k = k.strip().lower()
        v = v.strip()
        if k == "name":
            name = v
        elif k == "description":
            description = v
    if not name or not description:
        return None
    return {"name": name, "description": description}


def _safe_resolve(staging_dir: Path, rel_path: str) -> Path | None:
    """Resolve rel_path inside staging_dir; reject if it escapes.

    WRITE actions use this — they are always confined to staging_dir.
    """
    try:
        p = (staging_dir / rel_path).resolve()
    except Exception:
        return None
    try:
        p.relative_to(staging_dir.resolve())
    except ValueError:
        return None
    return p


# Files inside a solver trial dir that reveal the verifier's ANSWER KEY (test
# scripts, expected outputs, test stderr). The editor must NOT read these — they
# would let it code to the test (hard-code answers / learn the grader's quirks).
# The scalar pass/fail verdict itself is SUPPORT-side training signal and is
# already given in the reflection instruction (D11); what stays hidden is WHAT
# the grader checks, so the only way to flip a verdict is to fix behavior.
_REWARD_FILE_NAMES = {"result.json", "reward.txt", "reward.json", "ctrf.json"}


def _is_reward_file(p: Path) -> bool:
    """True if p is a reward/answer-bearing file in a trial dir (kept hidden from
    the editor). Covers result.json + everything under a `verifier/` dir
    (reward.txt, test-stdout.txt, ctrf.json, …)."""
    if p.name in _REWARD_FILE_NAMES:
        return True
    return "verifier" in p.parts


def _resolve_read(
    staging_dir: Path,
    traj_root: Path | None,
    path_str: str,
    skills_root: Path | None = None,
    self_test_root: Path | None = None,
) -> Path | None:
    """Resolve a READ path. Relative paths resolve under staging_dir (back-compat).
    Absolute paths are allowed ONLY if they live under staging_dir, traj_root
    (the read-only trajectory root), or skills_root (read-only SKILL.md docs).
    Returns None if disallowed/unsafe.

    Under traj_root, answer-key files (result.json, verifier/*) are REFUSED —
    the editor gets the scalar verdict in its instruction, never the grader's
    test content (no coding to the test).

    skills_root lets the editor read the BODY of a skill it sees listed in
    `<available_skills>` (the block only gives name/description/location); skills
    are read-only reference docs, no reward there.
    """
    if not path_str:
        return None
    cand = Path(path_str)
    if cand.is_absolute():
        try:
            rp = cand.resolve()
        except Exception:
            return None
        # staging is fully readable (module source — no reward there)
        try:
            rp.relative_to(staging_dir.resolve())
            return rp
        except ValueError:
            pass
        # skills root: read-only reference docs (SKILL.md bodies)
        if skills_root is not None:
            try:
                rp.relative_to(skills_root.resolve())
                return rp
            except ValueError:
                pass
        # trajectory root: readable EXCEPT reward/answer files (no eval leak)
        if traj_root is not None:
            try:
                rp.relative_to(traj_root.resolve())
                if not _is_reward_file(rp):
                    return rp
            except ValueError:
                pass
        # self-test root: the editor's own `<run_task>` rolls — same rule as
        # traj_root (behavior readable: pane/prompts/responses; grader files
        # blocked so the editor sees the verdict, not the answer key).
        if self_test_root is not None:
            try:
                rp.relative_to(self_test_root.resolve())
                if not _is_reward_file(rp):
                    return rp
            except ValueError:
                pass
        return None
    return _safe_resolve(staging_dir, path_str)


def _rel_for_display(
    p: Path,
    staging_dir: Path,
    traj_root: Path | None,
    skills_root: Path | None = None,
    self_test_root: Path | None = None,
) -> str:
    """Show p relative to whichever allowed root contains it; else absolute."""
    for root in (staging_dir, traj_root, skills_root, self_test_root):
        if root is None:
            continue
        try:
            return str(p.relative_to(root.resolve()))
        except ValueError:
            continue
    return str(p)


# ---------------------------------------------------------------------------
# EditorFileTools
# ---------------------------------------------------------------------------


class EditorFileTools:
    NAME = "editor_file_tools"
    NICHE = {"audience": "editor"}
    DESCRIPTION = (
        "Editor toolset for self-evo. Writes (edit_file / edit_lines / "
        "create_file) act on the staging modules dir; reads (read_file / grep) "
        "may also access the read-only trajectory_root (the batch's solver "
        "output) so you can investigate raw trajectories. commit_patch / "
        "validate too. No tmux, no shell."
    )
    PARAMS_SCHEMA: dict = {}

    def __init__(self, **_ignored):
        pass

    # ---------- Lifecycle ----------

    async def setup(self, ctx: ModuleCtx) -> None:
        if ctx.state.staging_dir is None:
            raise RuntimeError(
                "EditorFileTools requires ctx.state.staging_dir; got None. "
                "Pass `staging_dir=...` to Terminus2Modular(...)."
            )
        if not ctx.state.staging_dir.exists():
            raise FileNotFoundError(
                f"staging_dir does not exist: {ctx.state.staging_dir}"
            )

    async def teardown(self, ctx: ModuleCtx) -> None:
        return None

    async def build_skills_section(self, ctx: ModuleCtx) -> str | None:
        """Discover Agent Skills in ctx.state.skills_dir from LOCAL filesystem.

        Mirrors the XML format from Terminus2._build_skills_section but
        reads files directly (no env.exec; editor doesn't have a Docker env).
        """
        skills_dir = ctx.state.skills_dir
        if not skills_dir:
            return None
        skills_path = Path(skills_dir)
        if not skills_path.is_dir():
            return None

        entries: list[tuple[str, str, str]] = []
        for skill_md in sorted(skills_path.glob("*/SKILL.md")):
            try:
                content = skill_md.read_text()
            except Exception:
                continue
            fm = _parse_skill_frontmatter(content)
            if fm is None:
                continue
            # Emit an ABSOLUTE (resolved) location so the editor's read_file can
            # actually open it — a relative location would resolve under
            # staging_dir and 404. (Matches the skills_root branch in
            # _resolve_read, which compares resolved absolute paths.)
            entries.append((fm["name"], fm["description"], str(skill_md.resolve())))

        if not entries:
            return None

        from xml.etree.ElementTree import Element, SubElement, tostring

        root = Element("available_skills")
        for name, description, location in entries:
            skill_el = SubElement(root, "skill")
            SubElement(skill_el, "name").text = name
            SubElement(skill_el, "description").text = description
            SubElement(skill_el, "location").text = location
        return "\n\n" + tostring(root, encoding="unicode")

    # ---------- Prompt formatting ----------

    def format_initial_prompt(
        self,
        instruction: str,
        terminal_state: str,
        ctx: ModuleCtx,
    ) -> str:
        # `terminal_state` here is whatever observation.capture returned at
        # startup. For editor mode we replace it with a directory listing.
        listing = self._listing_for_prompt(ctx)
        sd = ctx.state.staging_dir
        kind = _staging_root_kind(sd) if sd is not None else "modules"
        return _SYSTEM_PROMPT_TEMPLATE.format(
            instruction=instruction,
            terminal_state=listing,
            kernel_rules=(
                _KERNEL_RULES_PACKAGE_ROOT
                if kind == "package"
                else _KERNEL_RULES_MODULES_ROOT
            ),
            self_test_grammar=_self_test_grammar(ctx),
        )

    def format_continuation_prompt(
        self,
        terminal_state: str,
        ctx: ModuleCtx,
    ) -> str:
        # Pass-through; the agent_loop already builds observation_text from
        # tool results (read_file output, edit results, etc.)
        return terminal_state

    def _listing_for_prompt(self, ctx: ModuleCtx) -> str:
        sd = ctx.state.staging_dir
        if sd is None or not sd.exists():
            return "(no staging dir)"
        try:
            lines: list[str] = []
            for p in sorted(sd.rglob("*")):
                if p.is_file():
                    rel = p.relative_to(sd)
                    lines.append(str(rel))
            if not lines:
                return "(empty staging dir)"
            if len(lines) > 200:
                lines = lines[:200] + [f"... ({len(lines) - 200} more)"]
            return "\n".join(lines)
        except Exception as exc:
            return f"(listing failed: {exc})"

    # ---------- LLM response parsing ----------

    def parse_llm_response(self, response: str) -> LLMResponseParseResult:
        m = _RESP.search(response)
        body = m.group(1) if m else response

        a = _ANALYSIS.search(body)
        p = _PLAN.search(body)
        done_m = _DONE.search(body)
        is_complete = done_m is not None and done_m.group(1).lower() == "true"
        analysis = a.group(1).strip() if a else ""
        plan = p.group(1).strip() if p else ""

        # Prefer the <actions> block when present, but DON'T require it — the
        # editor (esp. GLM) often emits bare <grep/>, <edit_file>… without the
        # wrapper, and the old strict parser silently dropped all of them.
        actions_m = _ACTIONS.search(body)
        action_text = actions_m.group(1) if actions_m else body
        commands = self._extract_actions(action_text)

        return LLMResponseParseResult(
            commands=commands,
            is_task_complete=is_complete,
            analysis=analysis,
            plan=plan,
            error="",
        )

    @staticmethod
    def _extract_actions(text: str) -> list[ToolCall]:
        """Pull tool actions out of `text` by regex (order-preserving).

        Robust to (a) missing <actions> wrapper and (b) code with < > & inside
        edit bodies — bodies are taken as raw text, never XML-parsed.
        """
        import json

        found: list[tuple[int, dict]] = []
        container_spans: list[tuple[int, int]] = []

        for m in _CONTAINER_RE.finditer(text):
            container_spans.append((m.start(), m.end()))
            tag = m.group(1).lower()
            attrs = dict(_ATTR_RE.findall(m.group(2)))
            inner = m.group(3)
            if tag == "edit_file":
                payload = {
                    "action": "edit_file",
                    "path": attrs.get("path", ""),
                    "old_string": _sub_text(inner, "old_string"),
                    "new_string": _sub_text(inner, "new_string"),
                }
            elif tag == "edit_lines":
                payload = {
                    "action": "edit_lines",
                    "path": attrs.get("path", ""),
                    "start_line": attrs.get("start_line"),
                    "end_line": attrs.get("end_line"),
                    "new_string": _sub_text(inner, "new_string"),
                }
            else:  # create_file
                payload = {
                    "action": "create_file",
                    "path": attrs.get("path", ""),
                    "content": _sub_text(inner, "content"),
                }
            found.append((m.start(), payload))

        for m in _SELFCLOSE_RE.finditer(text):
            pos = m.start()
            # Skip self-closing tags that fall inside a container body (e.g. a
            # literal <grep …> appearing in code being inserted).
            if any(s <= pos < e for s, e in container_spans):
                continue
            tag = m.group(1).lower()
            attrs = dict(_ATTR_RE.findall(m.group(2)))
            if tag == "read_file":
                payload = {"action": "read_file", "path": attrs.get("path", "")}
                if attrs.get("start_line"):
                    payload["start_line"] = attrs["start_line"]
                if attrs.get("end_line"):
                    payload["end_line"] = attrs["end_line"]
            elif tag == "grep":
                payload = {
                    "action": "grep",
                    "pattern": attrs.get("pattern", ""),
                    "path": attrs.get("path", ""),
                }
                if attrs.get("context"):
                    payload["context"] = attrs["context"]
            elif tag == "glob":
                payload = {"action": "glob", "pattern": attrs.get("pattern", "")}
            elif tag == "validate":
                payload = {"action": "validate"}
            elif tag == "archive":
                payload = {
                    "action": "archive",
                    "type": attrs.get("type", ""),
                    "lineage": attrs.get("lineage", ""),
                }
            elif tag == "run_task":
                payload = {"action": "run_task", "task": attrs.get("task", "")}
            else:  # commit_patch
                payload = {"action": "commit_patch"}
            found.append((pos, payload))

        found.sort(key=lambda x: x[0])
        return [ToolCall(keystrokes=json.dumps(p), duration_sec=0.0) for _, p in found]

    # ---------- Execution ----------

    async def execute(self, call: ToolCall, ctx: ModuleCtx) -> ToolResult:
        import json

        try:
            payload = json.loads(call.keystrokes)
        except Exception as exc:
            return ToolResult(success=False, output="", error=f"bad payload: {exc}")

        action = payload.get("action")
        sd = ctx.state.staging_dir
        if sd is None:
            return ToolResult(success=False, output="", error="no staging_dir on ctx")

        tr = ctx.state.trajectory_root
        sk = Path(ctx.state.skills_dir) if ctx.state.skills_dir else None
        st = getattr(ctx.state, "self_test_root", None)
        if action == "read_file":
            return self._do_read(
                sd,
                payload.get("path", ""),
                ctx,
                payload.get("start_line"),
                payload.get("end_line"),
                traj_root=tr,
                skills_root=sk,
                self_test_root=st,
            )
        if action == "grep":
            return self._do_grep(
                sd,
                payload.get("pattern", ""),
                payload.get("path", ""),
                ctx,
                payload.get("context"),
                traj_root=tr,
                skills_root=sk,
                self_test_root=st,
            )
        if action == "glob":
            return self._do_glob(sd, payload.get("pattern", ""), ctx)
        if action == "validate":
            return self._do_validate(sd, ctx)
        if action == "run_task":
            return await self._do_run_task(sd, payload.get("task", ""), ctx)
        if action == "archive":
            return self._do_archive(
                sd, payload.get("type", ""), payload.get("lineage", ""), ctx
            )
        if action == "edit_lines":
            return self._do_edit_lines(
                sd,
                payload.get("path", ""),
                payload.get("start_line"),
                payload.get("end_line"),
                payload.get("new_string", ""),
                ctx,
            )
        if action == "edit_file":
            return self._do_edit(
                sd,
                payload.get("path", ""),
                payload.get("old_string", ""),
                payload.get("new_string", ""),
                ctx,
            )
        if action == "create_file":
            return self._do_create(
                sd, payload.get("path", ""), payload.get("content", ""), ctx
            )
        if action == "commit_patch":
            # Mark the staging as ready; smoke tests run outside this module.
            ctx.shared.commit_patch_requested = True  # type: ignore[attr-defined]
            return ToolResult(
                success=True,
                output="commit_patch acknowledged; outer driver will run smoke tests.",
            )

        return ToolResult(success=False, output="", error=f"unknown action: {action!r}")

    # ---- self-test (`<run_task>`) ----

    async def _do_run_task(
        self, staging_dir: Path, task: str, ctx: ModuleCtx
    ) -> ToolResult:
        """Self-test: run the CURRENT staging on a FIXED simple/fast task in a
        real sandbox (path-mode) — a canned "does my code actually RUN?" smoke,
        NOT a way to test a change's effect on a chosen task. Returns a compact
        runtime digest + a pointer to the whole roll folder so the editor can read
        the errors (job.log, traceback, terminal pane) and fix until it runs.
        Bounded by SelfTestConfig (fixed smoke tasks + per-session budget)."""
        import asyncio

        cfg = getattr(ctx.state, "self_test", None)
        if cfg is None or not getattr(cfg, "enabled", False):
            return ToolResult(
                success=False,
                output="",
                error="self-test unavailable (disabled for this run)",
            )
        smoke = [t for t in (getattr(cfg, "tasks", None) or []) if t]
        if not smoke:
            return ToolResult(
                success=False,
                output="",
                error="self-test unavailable (no smoke tasks configured)",
            )
        count = getattr(ctx.shared, "self_test_count", 0)
        if count >= cfg.max_self_tests:
            return ToolResult(
                success=False,
                output="",
                error=(
                    f"self-test budget exhausted ({cfg.max_self_tests} used) — "
                    "make your best-judgement edit and <commit_patch/>."
                ),
            )
        # The editor does NOT choose the task — <run_task/> is a canned runtime
        # smoke on a FIXED simple task. An explicit task= is honored only if it's
        # already one of the configured smoke tasks; otherwise the set rotates.
        requested = (task or "").strip()
        task = requested if requested in smoke else smoke[count % len(smoke)]
        st_root = getattr(ctx.state, "self_test_root", None)
        if st_root is None:
            return ToolResult(
                success=False, output="", error="no self_test_root configured"
            )

        # Lazy imports: self_evo imports the modules package, not vice versa —
        # importing at module top would create a cycle.
        from harbor.agents.terminus_2_modular.self_evo.evo_driver import (
            _run_harbor_task,
        )
        from harbor.agents.terminus_2_modular.self_evo.online_evo import (
            _gen_run_kwargs,
            _harbor_results_to_summary,
            _is_code_break,
        )

        out_dir = Path(st_root) / f"{count}_{task}"
        out_dir.mkdir(parents=True, exist_ok=True)
        task_dir = (
            (Path(cfg.support_task_dir) / task)
            if cfg.support_task_dir is not None
            else None
        )
        run_kwargs = dict(
            task=task,
            **_gen_run_kwargs(Path(staging_dir)),
            model_name=cfg.model_name,
            api_base=cfg.api_base,
            api_key=cfg.api_key,
            output_dir=out_dir,
            max_turns=cfg.max_turns,
            timeout_sec=cfg.timeout_sec,
            environment=cfg.env,
            task_dir=task_dir,
            temperature=cfg.temperature,
            locked_module_type=getattr(ctx.state, "locked_module_type", None),
        )

        ctx.services.logger.info(
            "[self-test %d/%d] running %s on staging …",
            count + 1,
            cfg.max_self_tests,
            task,
        )
        # Charge the budget up-front so a hang/exception can't be retried for free.
        ctx.shared.self_test_count = count + 1
        pool = getattr(cfg, "e2b_pool", None)
        try:
            if pool is not None and hasattr(pool, "slot"):
                async with pool.slot() as s:
                    hr = await asyncio.to_thread(
                        _run_harbor_task,
                        e2b_key=getattr(s, "key", None),
                        e2b_token=getattr(s, "token", None),
                        **run_kwargs,
                    )
            else:
                hr = await asyncio.to_thread(_run_harbor_task, **run_kwargs)
        except Exception as exc:
            return ToolResult(
                success=False, output="", error=f"self-test roll errored: {exc}"
            )

        summary = _harbor_results_to_summary(out_dir)
        digest = self._format_self_test_result(hr, summary, out_dir, _is_code_break)
        return ToolResult(success=True, output=digest)

    @staticmethod
    def _format_self_test_result(hr, summary, out_dir: Path, is_code_break) -> str:
        """Compact digest of one self-test roll, geared at 'does my code RUN?'.
        Leads with the runtime verdict (ran / crashed / no-score) + the exception,
        then points at the WHOLE roll folder (job.log, exception.txt, agent
        trajectory + terminal pane) so the editor can read the tracebacks and fix.
        The grader's answer-key files (result.json, verifier/*) stay blocked;
        everything else in the folder is readable."""
        reward = hr.reward
        broke, why = is_code_break(hr, summary)
        if broke:
            runtime = f"CRASHED / did not run cleanly — {why}"
        elif reward is None:
            runtime = (
                f"ran, but NO SCORE (infra/harness error: {hr.error or 'unknown'})"
            )
        else:
            runtime = "ran end-to-end (no crash)"

        lines = [f"=== self-test: {hr.task} ===", f"runtime: {runtime}"]
        if reward is not None:
            if reward >= 1.0:
                lines.append(f"task outcome: PASSED (reward={reward})")
            elif reward > 0:
                lines.append(f"task outcome: PARTIAL (reward={reward})")
            else:
                lines.append(f"task outcome: FAILED (reward={reward})")
        if hr.error:
            lines.append(f"error: {hr.error}")
        target = str(out_dir)
        if summary is not None:
            if summary.exception_type:
                em = (summary.exception_message or "").strip()
                lines.append(
                    f"exception: {summary.exception_type}"
                    + (f": {em[:300]}" if em else "")
                )
            lines.append(f"episodes: {summary.n_episodes}")
            if summary.failure_signals:
                lines.append("failure signals: " + ", ".join(summary.failure_signals))
            msgs = summary.last_step_messages or []
            if msgs:
                lines.append("last agent messages:")
                for m in msgs[-3:]:
                    lines.append(f"  · {m[:400]}")
            if getattr(summary, "trial_dir", None):
                target = summary.trial_dir
        lines.append(
            "\nEVERYTHING this roll produced is readable under the roll folder — "
            "read job.log (orchestration errors), exception.txt (traceback), and "
            "agent/ (trajectory + terminal pane) to see what broke, then fix it:\n"
            f"  {out_dir}\n"
            f"  trial dir: {target}\n"
            f'  e.g. <read_file path="{target}/agent/terminus_2_modular.pane"/>'
        )
        lines.append(
            "Goal: get the code to RUN correctly. If it crashed, fix the crash and "
            "(budget permitting) <run_task/> again. Don't hard-code to a task's "
            "specific checks — aim for a general fix."
        )
        return "\n".join(lines)

    # ---- per-action helpers ----

    @staticmethod
    def _do_archive(
        staging_dir: Path, type_: str, lineage: str, ctx: ModuleCtx
    ) -> ToolResult:
        """Read-only niche/genealogy view (D4). Uses the run's archive.json when
        available (full lineage/addresses), else computes niche cells on-the-fly
        from the staging modules (niche only, no genealogy)."""
        from harbor.agents.terminus_2_modular import archive as _arch

        try:
            ap = ctx.state.archive_path
            if ap is not None and Path(ap).is_file():
                entries = _arch.load_archive(Path(ap).parent)
            else:
                from harbor.agents.terminus_2_modular.library import (
                    build_default_library,
                )

                lib = build_default_library(modules_root=staging_dir)
                entries = _arch.seed_from_library(lib)
            if lineage:
                out = _arch.render_lineage(entries, lineage)
            elif type_:
                out = _arch.render_type(entries, type_)
            else:
                types = sorted({e.type for e in entries})
                counts = ", ".join(
                    f"{t}({sum(1 for e in entries if e.type == t)})" for t in types
                )
                out = (
                    f"archive — types: {counts}\n"
                    'Query one: <archive type="observation"/> or '
                    '<archive lineage="<variant>"/>'
                )
            return ToolResult(success=True, output=out)
        except Exception as exc:
            return ToolResult(success=False, output="", error=f"archive: {exc}")

    @staticmethod
    def _do_read(
        staging_dir: Path,
        rel_path: str,
        ctx: ModuleCtx,
        start_line: str | int | None = None,
        end_line: str | int | None = None,
        traj_root: Path | None = None,
        skills_root: Path | None = None,
        self_test_root: Path | None = None,
    ) -> ToolResult:
        if _is_off_limits(rel_path, _staging_root_kind(staging_dir)):
            return ToolResult(
                success=False,
                output="",
                error=f"path is off-limits (frozen): {rel_path}",
            )
        p = _resolve_read(
            staging_dir,
            traj_root,
            rel_path,
            skills_root=skills_root,
            self_test_root=self_test_root,
        )
        if p is None:
            return ToolResult(
                success=False, output="", error=f"unsafe path: {rel_path}"
            )
        if not p.exists():
            return ToolResult(
                success=False, output="", error=f"file not found: {rel_path}"
            )
        try:
            lines = p.read_text().splitlines()
        except Exception as exc:
            return ToolResult(success=False, output="", error=str(exc))
        n = len(lines)

        def _numbered(lo: int, hi: int) -> str:
            # 1-based inclusive [lo, hi], each line prefixed with its number.
            return "\n".join(f"{i:>5}  {lines[i - 1]}" for i in range(lo, hi + 1))

        # Explicit line-range read.
        if start_line is not None or end_line is not None:
            try:
                s = int(start_line) if start_line is not None else 1
                e = int(end_line) if end_line is not None else n
            except (TypeError, ValueError):
                return ToolResult(
                    success=False,
                    output="",
                    error="start_line/end_line must be integers",
                )
            s = max(1, s)
            e = min(n, e)
            if s > e:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"empty range: start_line {s} > end_line {e}",
                )
            return ToolResult(
                success=True,
                output=f"=== {rel_path} (lines {s}-{e} of {n}) ===\n{_numbered(s, e)}",
            )

        # Full read, with line numbers. If too big, keep head + tail BY LINE
        # and report EXACTLY which lines were omitted, so the editor knows the
        # middle exists and can grep / range-read it (never silently drop a
        # call site like before).
        full = _numbered(1, n) if n else ""
        if len(full) <= 32_000:
            return ToolResult(
                success=True,
                output=f"=== {rel_path} ({n} lines) ===\n{full}",
            )

        head_hi = 0
        acc = 0
        for i in range(1, n + 1):
            acc += len(lines[i - 1]) + 7
            if acc > 14_000:
                break
            head_hi = i
        tail_lo = n + 1
        acc = 0
        for i in range(n, 0, -1):
            acc += len(lines[i - 1]) + 7
            if acc > 14_000:
                break
            tail_lo = i
        if tail_lo <= head_hi + 1:
            # Head and tail meet — whole file fits after all.
            return ToolResult(
                success=True,
                output=f"=== {rel_path} ({n} lines) ===\n{full}",
            )
        head = _numbered(1, head_hi)
        tail = _numbered(tail_lo, n)
        return ToolResult(
            success=True,
            output=(
                f"=== {rel_path} ({n} lines, TRUNCATED) ===\n{head}\n\n"
                f"... [lines {head_hi + 1}-{tail_lo - 1} omitted — read them with "
                f'`<read_file path="{rel_path}" start_line="{head_hi + 1}" '
                f'end_line="{tail_lo - 1}"/>`, or use <grep> to locate a symbol] ...'
                f"\n\n{tail}"
            ),
        )

    @staticmethod
    def _do_grep(
        staging_dir: Path,
        pattern: str,
        rel_path: str,
        ctx: ModuleCtx,
        context_arg=None,
        traj_root: Path | None = None,
        skills_root: Path | None = None,
        self_test_root: Path | None = None,
    ) -> ToolResult:
        if not pattern:
            return ToolResult(
                success=False, output="", error="grep requires a non-empty pattern"
            )
        try:
            ctx_lines = max(0, min(int(context_arg), 10)) if context_arg else 0
        except (TypeError, ValueError):
            ctx_lines = 0
        root_kind = _staging_root_kind(staging_dir)
        root = staging_dir
        if rel_path:
            if _is_off_limits(rel_path, root_kind):
                return ToolResult(
                    success=False,
                    output="",
                    error=f"path is off-limits (frozen): {rel_path}",
                )
            rp = _resolve_read(
                staging_dir,
                traj_root,
                rel_path,
                skills_root=skills_root,
                self_test_root=self_test_root,
            )
            if rp is None or not rp.exists():
                return ToolResult(
                    success=False, output="", error=f"path not found: {rel_path}"
                )
            root = rp
        try:
            rx = re.compile(pattern)
        except re.error:
            rx = None  # fall back to literal substring match
        # Inside staging: only .py modules. Inside the read-only trajectory
        # root: broaden to the text-ish files a trajectory actually contains
        # (json/txt/log/pane/...), so the editor can search what the agent saw.
        if root.is_file():
            files = [] if _is_reward_file(root) else [root]
        else:
            _under_staging = True
            try:
                root.resolve().relative_to(staging_dir.resolve())
            except ValueError:
                _under_staging = False
            if _under_staging:
                files = sorted(root.rglob("*.py"))
            else:
                _exts = {
                    ".py",
                    ".txt",
                    ".json",
                    ".log",
                    ".md",
                    ".pane",
                    ".toml",
                    ".sh",
                    ".cast",
                    ".out",
                    ".err",
                }
                # Exclude reward/answer files (result.json, verifier/*) — grepping
                # the trial dir must not surface the eval score / answer key.
                files = sorted(
                    f
                    for f in root.rglob("*")
                    if f.is_file() and f.suffix in _exts and not _is_reward_file(f)
                )
        blocks: list[str] = []
        nmatch = 0
        cap = 100
        for f in files:
            if not f.is_file():
                continue
            rel = _rel_for_display(
                f, staging_dir, traj_root, skills_root, self_test_root
            )
            if _is_off_limits(rel, root_kind):
                continue
            try:
                lines = f.read_text().splitlines()
            except Exception:
                continue
            for i, line in enumerate(lines, start=1):
                if rx.search(line) if rx else (pattern in line):
                    nmatch += 1
                    if ctx_lines == 0:
                        blocks.append(f"{rel}:{i}: {line.strip()[:200]}")
                    else:
                        lo = max(1, i - ctx_lines)
                        hi = min(len(lines), i + ctx_lines)
                        blocks.append(
                            "\n".join(
                                f"{rel}:{j}:{'>' if j == i else ' '} {lines[j - 1][:200]}"
                                for j in range(lo, hi + 1)
                            )
                        )
                    if nmatch >= cap:
                        break
            if nmatch >= cap:
                break
        if not blocks:
            return ToolResult(success=True, output=f"grep '{pattern}': no matches")
        sep = "\n--\n" if ctx_lines else "\n"
        capped = "\n... (capped at 100 matches)" if nmatch >= cap else ""
        return ToolResult(
            success=True,
            output=f"grep '{pattern}' ({nmatch} match line(s)):\n"
            + sep.join(blocks)
            + capped,
        )

    @staticmethod
    def _do_edit(
        staging_dir: Path,
        rel_path: str,
        old_string: str,
        new_string: str,
        ctx: ModuleCtx,
    ) -> ToolResult:
        if _is_off_limits(rel_path, _staging_root_kind(staging_dir)):
            return ToolResult(
                success=False,
                output="",
                error=f"path is off-limits (frozen): {rel_path}",
            )
        locked = getattr(getattr(ctx, "state", None), "locked_module_type", None)
        if not _write_allowed(rel_path, locked, _staging_root_kind(staging_dir)):
            return ToolResult(
                success=False,
                output="",
                error=f"module lock: refused {rel_path} — may only ADD/MERGE variants under {locked}/; baseline.py is read-only (add or merge a variant instead)",
            )
        p = _safe_resolve(staging_dir, rel_path)
        if p is None:
            return ToolResult(
                success=False, output="", error=f"unsafe path: {rel_path}"
            )
        if not p.exists():
            return ToolResult(
                success=False, output="", error=f"file not found: {rel_path}"
            )
        if not old_string:
            return ToolResult(
                success=False, output="", error="old_string must not be empty"
            )
        try:
            text = p.read_text()
        except Exception as exc:
            return ToolResult(success=False, output="", error=str(exc))
        occurrences = text.count(old_string)
        if occurrences == 0:
            return ToolResult(
                success=False,
                output="",
                error=(
                    f"old_string not found in {rel_path}. "
                    "Use read_file to see the current contents and retry "
                    "with surrounding context."
                ),
            )
        if occurrences > 1:
            return ToolResult(
                success=False,
                output="",
                error=(
                    f"old_string occurs {occurrences} times in {rel_path}; "
                    "include more surrounding context to make it unique."
                ),
            )
        new_text = text.replace(old_string, new_string, 1)
        try:
            p.write_text(new_text)
        except Exception as exc:
            return ToolResult(success=False, output="", error=str(exc))
        return ToolResult(
            success=True,
            output=f"edit_file ok: {rel_path} (-{len(old_string)} +{len(new_string)} chars)",
        )

    @staticmethod
    def _do_create(
        staging_dir: Path, rel_path: str, content: str, ctx: ModuleCtx
    ) -> ToolResult:
        if _is_off_limits(rel_path, _staging_root_kind(staging_dir)):
            return ToolResult(
                success=False,
                output="",
                error=f"path is off-limits (frozen): {rel_path}",
            )
        locked = getattr(getattr(ctx, "state", None), "locked_module_type", None)
        if not _write_allowed(rel_path, locked, _staging_root_kind(staging_dir)):
            return ToolResult(
                success=False,
                output="",
                error=f"module lock: refused {rel_path} — may only ADD/MERGE variants under {locked}/; baseline.py is read-only (add or merge a variant instead)",
            )
        p = _safe_resolve(staging_dir, rel_path)
        if p is None:
            return ToolResult(
                success=False, output="", error=f"unsafe path: {rel_path}"
            )
        if p.exists():
            return ToolResult(
                success=False,
                output="",
                error=f"file already exists: {rel_path} (use edit_file to modify)",
            )
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
        except Exception as exc:
            return ToolResult(success=False, output="", error=str(exc))
        return ToolResult(
            success=True,
            output=f"create_file ok: {rel_path} ({len(content)} chars)",
        )

    @staticmethod
    def _do_glob(staging_dir: Path, pattern: str, ctx: ModuleCtx) -> ToolResult:
        if not pattern:
            return ToolResult(
                success=False, output="", error="glob requires a non-empty pattern"
            )
        root_kind = _staging_root_kind(staging_dir)
        try:
            hits = sorted(
                str(p.relative_to(staging_dir))
                for p in staging_dir.glob(pattern)
                if p.is_file()
                and not _is_off_limits(str(p.relative_to(staging_dir)), root_kind)
            )
        except Exception as exc:
            return ToolResult(success=False, output="", error=f"bad glob: {exc}")
        if not hits:
            return ToolResult(success=True, output=f"glob '{pattern}': no files")
        return ToolResult(
            success=True,
            output=f"glob '{pattern}' ({len(hits)} files):\n" + "\n".join(hits[:200]),
        )

    @staticmethod
    def _do_validate(staging_dir: Path, ctx: ModuleCtx) -> ToolResult:
        """Self-check the staged modules WITHOUT committing — same battery the
        external smoke gate runs (AST + import + library load + bundle config).
        Lets the editor iterate to a clean state before commit_patch."""
        # Local import to avoid loading harbor at module import time.
        from harbor.agents.terminus_2_modular.self_evo.smoke_tests import (
            run_fast_smoke,
        )

        try:
            rep = run_fast_smoke(staging_dir)
        except Exception as exc:
            return ToolResult(
                success=False, output="", error=f"validate crashed: {exc}"
            )
        if rep.passed:
            notes = []
            for r in (
                rep.ast,
                rep.imports,
                rep.library,
                rep.bundle_config,
                rep.discovery,
                rep.static,
            ):
                if r is not None:
                    notes.extend(r.notes)
            note_str = ("\n  " + "\n  ".join(notes)) if notes else ""
            return ToolResult(
                success=True,
                output="validate PASSED (AST + import + library load + bundle "
                f"config + discovery contract + static name/attr audit all "
                f"OK).{note_str}",
            )
        fails = "\n  ".join(rep.all_failures())
        return ToolResult(
            success=True,  # the action ran fine; the staging just isn't clean yet
            output=f"validate FAILED — fix these before commit_patch:\n  {fails}",
        )

    @staticmethod
    def _do_edit_lines(
        staging_dir: Path,
        rel_path: str,
        start_line,
        end_line,
        new_string: str,
        ctx: ModuleCtx,
    ) -> ToolResult:
        """Replace lines [start_line, end_line] (1-based inclusive) with
        new_string. More robust than edit_file on large files — pair with a
        line-numbered read_file to get exact line numbers."""
        if _is_off_limits(rel_path, _staging_root_kind(staging_dir)):
            return ToolResult(
                success=False,
                output="",
                error=f"path is off-limits (frozen): {rel_path}",
            )
        locked = getattr(getattr(ctx, "state", None), "locked_module_type", None)
        if not _write_allowed(rel_path, locked, _staging_root_kind(staging_dir)):
            return ToolResult(
                success=False,
                output="",
                error=f"module lock: refused {rel_path} — may only ADD/MERGE variants under {locked}/; baseline.py is read-only (add or merge a variant instead)",
            )
        p = _safe_resolve(staging_dir, rel_path)
        if p is None:
            return ToolResult(
                success=False, output="", error=f"unsafe path: {rel_path}"
            )
        if not p.exists():
            return ToolResult(
                success=False, output="", error=f"file not found: {rel_path}"
            )
        try:
            s = int(start_line)
            e = int(end_line)
        except (TypeError, ValueError):
            return ToolResult(
                success=False,
                output="",
                error="edit_lines requires integer start_line and end_line",
            )
        try:
            lines = p.read_text().splitlines(keepends=True)
        except Exception as exc:
            return ToolResult(success=False, output="", error=str(exc))
        n = len(lines)
        if s < 1 or e > n or s > e:
            return ToolResult(
                success=False,
                output="",
                error=f"bad range: start_line={s} end_line={e}, file has {n} lines",
            )
        repl = (
            new_string
            if new_string.endswith("\n") or not new_string
            else new_string + "\n"
        )
        new_lines = lines[: s - 1] + [repl] + lines[e:]
        try:
            p.write_text("".join(new_lines))
        except Exception as exc:
            return ToolResult(success=False, output="", error=str(exc))
        return ToolResult(
            success=True,
            output=f"edit_lines ok: {rel_path} (replaced lines {s}-{e}, "
            f"{e - s + 1} → {repl.count(chr(10))} lines)",
        )


def register(library):
    library.register(
        type_="tools",
        name=EditorFileTools.NAME,
        factory=lambda params: EditorFileTools(**params),
        description=EditorFileTools.DESCRIPTION,
        params_schema=EditorFileTools.PARAMS_SCHEMA,
        niche=EditorFileTools.NICHE,
    )
