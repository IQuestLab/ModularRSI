"""Planning + guard loop — combines requirement tracking with completion-integrity
guards, stuck detection, read-only inspection detection, and background process
termination detection.

Merges PlanningChecklistLoop's requirement extraction/progress tracking with
CompletionIntegrityGuardLoop's evidence-gated termination, repeated-command
detection, read-only inspection guard, and re-orientation on repeated rejected
completions.

Key integration: whenever the model declares task_complete while any extracted
requirement is still pending (whether or not that step executed commands), the
declaration is rejected and the rejection message quotes the specific pending
requirements. This directly addresses the failure mode where the model declares
completion while silently dropping stated requirements.

Addresses failure modes where the agent:
- declares task_complete=true while the task's stated acceptance checks or
  extracted requirements are still unmet (with or without commands that step)
- repeats the same command 3+ times without adapting strategy
- spends multiple consecutive turns inspecting (ls, cat, man) without creating
  any task artifacts or running build/test commands
- declares completion without verifying all stated acceptance criteria
- runs a background process that gets stopped (SIGTTOU) or killed without
  noticing, then keeps trying to reach the dead process
"""

from __future__ import annotations

import re
from collections import deque
from typing import Any, ClassVar

from harbor.agents.terminus_2_modular.modules.agent_loop.baseline import (
    BaselineAgentLoop,
)
from harbor.agents.terminus_2_modular.protocols import (
    AgentLoopState,
    ModuleCtx,
    ToolResult,
    ToolSet,
)

# Regex-like patterns for read-only commands (no side effects on disk/process)
_READONLY_PREFIXES: tuple[str, ...] = (
    "ls",
    "cat",
    "which",
    "env",
    "find",
    "man",
    "grep",
    "head",
    "tail",
    "echo",
    "sleep",
    "pwd",
    "whoami",
    "id",
    "date",
    "uname",
    "pg_isready",
    "pg_lsclusters",
    "psql -c",
    "apt-cache",
    "dpkg -l",
    "dpkg -L",
    "apt-mark showhold",
    "help",
    "--help",
    "-h",
    "type",
)
_READONLY_ARTIFACT_MARKERS: tuple[str, ...] = (
    "CREATE TABLE",
    "Schema restore complete",
    "server started",
    "restore_schema.sh",
    "solution.sh",
    "neovim",
    "createdb",
)


class PlanningWithGuardLoop(BaselineAgentLoop):
    NAME = "planning_with_guard"
    NICHE = {"grounding": "planning", "drift": "stuck-detection", "parse": "baseline"}
    DESCRIPTION: ClassVar[str] = (
        "Planning checklist + completion-integrity guard with structural enforcement. "
        "Extracts requirements from the original instruction, tracks which are "
        "[DONE]/[TODO] via command content and tool output, and injects a 'Task "
        "Progress' section into the prompt each iteration. Also includes: (1) rejection "
        "of any completion declaration while requirements are still pending — the "
        "rejection quotes those pending requirements; (2) repeated-command detection "
        "with strategy-shift nudge, and BLOCKING of repeated commands after the nudge "
        "is ignored; (3) read-only-inspection guard with progress nudge; (4) re-orientation "
        "when the verifier rejects completions 2+ times; (5) empty completion detection "
        "with structural enforcement — after the threshold is exceeded, empty completion "
        "declarations are replaced with a [SYSTEM — COMMAND BLOCKED] message; (6) hard "
        "reset after 6+ cumulative empty completions (cumulative across the run, "
        "so alternating complete/not-complete declarations cannot evade the reset) — clears conversation history and "
        "restarts with a fresh prompt. (7) forced termination after 2 consecutive "
        "task_complete=true declarations (the two-phase confirmation gate) — when the "
        "model has confirmed completion twice and no pending requirements remain, the "
        "loop terminates regardless of the verifier's decision, preventing budget "
        "exhaustion when the work is actually complete. (8) background process "
        "termination detection — when terminal output shows a background process was "
        "stopped (SIGTTOU), killed, or terminated, injects a nudge suggesting the "
        "agent use nohup/setsid or run the process without backgrounding. Prefer for "
        "multi-requirement build/test tasks where the model tends to overlook items "
        "or declare completion prematurely."
    )
    PARAMS_SCHEMA: ClassVar[dict[str, str]] = {
        "max_checklist_items": (
            "int (default 20) — cap on number of extracted requirements to avoid "
            "prompt flooding with trivial items"
        ),
        "max_empty_completions": (
            "int (default 2) — consecutive zero-command task_complete steps "
            "before the re-orientation message is injected"
        ),
        "max_rejected_completions": (
            "int (default 2) — consecutive task_complete declarations rejected "
            "by the verifier before the re-orientation message is injected"
        ),
        "max_repeated_commands": (
            "int (default 3) — identical consecutive commands before a "
            "'you seem stuck' nudge is injected"
        ),
        "max_failed_commands": (
            "int (default 2) — consecutive command failures (non-zero exit, "
            "tool error) before an error-aware nudge is injected"
        ),
        "max_readonly_steps": (
            "int (default 3) — consecutive steps with only read-only commands "
            "before a 'stop inspecting, start implementing' nudge is injected"
        ),
        "_command_history_size": (
            "int (default 5) — how many recent command strings to track for "
            "repetition detection"
        ),
        "max_pending_completion_rejections": (
            "int (default 5) — consecutive task_complete declarations rejected "
            "because requirements are still pending before the loop falls back "
            "to the baseline two-phase gate (anti-hang safety valve)"
        ),
        "max_empty_completions_before_reset": (
            "int (default 6) — cumulative empty completions before the loop "
            "clears conversation history and restarts with a fresh prompt "
            "(hard reset to break stuck loops, uses cumulative total to "
            "prevent alternation loops from evading the safety valve)"
        ),
        "max_file_writes": (
            "int (default 3) — consecutive writes to the same file path "
            "before a 'file has been written X times' nudge is injected, "
            "to prevent the agent from overwriting a correct output file "
            "multiple times"
        ),
    }

    def __init__(
        self,
        max_iterations: int = 1_000_000,
        llm_call_kwargs: dict | None = None,
        raw_content: bool = False,
        max_checklist_items: int = 20,
        max_empty_completions: int = 2,
        max_rejected_completions: int = 2,
        max_repeated_commands: int = 3,
        max_failed_commands: int = 2,
        max_readonly_steps: int = 3,
        _command_history_size: int = 5,
        max_pending_completion_rejections: int = 5,
        max_empty_completions_before_reset: int = 6,
        max_file_writes: int = 3,
    ):
        super().__init__(
            max_iterations=max_iterations,
            llm_call_kwargs=llm_call_kwargs,
            raw_content=raw_content,
        )
        self.max_checklist_items = max(1, max_checklist_items)
        self.max_empty_completions = max(1, max_empty_completions)
        self.max_rejected_completions = max(1, max_rejected_completions)
        self.max_repeated_commands = max(2, max_repeated_commands)
        self.max_failed_commands = max(1, max_failed_commands)
        self.max_readonly_steps = max(2, max_readonly_steps)
        self._command_history_size = max(3, _command_history_size)
        self.max_pending_completion_rejections = max(
            1, max_pending_completion_rejections
        )
        self.max_empty_completions_before_reset = max(
            3, max_empty_completions_before_reset
        )
        self.max_file_writes = max(2, max_file_writes)
        self._pending_completion_rejections = 0
        self._pending_gate_disabled = False

        # Planning checklist state
        self._requirements: list[str] = []
        self._completed: set[int] = set()  # indices into _requirements
        self._original_instruction: str = ""

        # Completion guard state
        self._acceptance_checklist: str | None = None
        self._checklist_extracted = False
        self._consecutive_empty_completions = 0
        self._total_empty_completions = (
            0  # cumulative, never reset by task_complete=False steps
        )
        self._consecutive_rejected_completions = 0
        self._needs_reorientation = False
        self._needs_hard_reset = False

        # Repeated-command detection state
        self._command_history: deque[tuple[str, ...]] = deque(
            maxlen=self._command_history_size
        )
        self._consecutive_identical_commands = 0
        self._last_injected_repeat_nudge = False  # one nudge per stuck streak

        # Consecutive command failure state
        self._consecutive_failed_commands = 0
        self._last_injected_failure_nudge = False

        # Read-only inspection guard state
        self._consecutive_readonly_steps = 0
        self._last_injected_readonly_nudge = False

        # Hard reset state
        self._chat_ref: Any = None

        # File-write tracking state
        self._file_write_count: dict[str, int] = {}
        self._file_write_nudge_injected: set[str] = set()
        # Initialize before `_pre_llm_prompt`; reset branches run later in the loop.
        self._last_verifier_reason = None

    # ---------- Requirement extraction (from planning_checklist) ----------

    async def _query_llm(
        self,
        chat,
        prompt,
        ctx,
        context_mgmt,
        tools,
        original_instruction,
        logging_paths,
    ) -> Any:
        """Override: stash original instruction and extract requirements on first call.
        Also handles hard reset by clearing chat history and re-initializing prompt."""
        self._original_instruction = original_instruction
        if not self._requirements:
            self._requirements = self._extract_requirements(original_instruction)
        if not self._checklist_extracted:
            self._checklist_extracted = True
            self._acceptance_checklist = self._extract_acceptance_checks(
                original_instruction
            )
        # Handle hard reset: clear chat history and re-initialize prompt
        if self._needs_hard_reset:
            self._needs_hard_reset = False
            # Clear the conversation history
            chat.messages.clear()
            self._consecutive_empty_completions = 0
            self._total_empty_completions = 0
            self._consecutive_rejected_completions = 0
            self._consecutive_identical_commands = 0
            self._last_injected_repeat_nudge = False
            self._consecutive_failed_commands = 0
            self._last_injected_failure_nudge = False
            self._consecutive_readonly_steps = 0
            self._last_injected_readonly_nudge = False
            self._pending_completion_rejections = 0
            self._pending_gate_disabled = False
            self._command_history.clear()
            self._last_verifier_reason = None
            self._needs_reorientation = False
            # Reset requirements tracking so they get re-extracted
            self._requirements = []
            self._completed = set()
            self._checklist_extracted = False
            self._acceptance_checklist = None
            # Re-extract requirements from the fresh start
            self._requirements = self._extract_requirements(original_instruction)
            self._acceptance_checklist = self._extract_acceptance_checks(
                original_instruction
            )
            self._checklist_extracted = True
            # Create a fresh prompt with the original instruction + reset note
            prompt = (
                f"[SYSTEM — HARD RESET]\n"
                f"The previous agent session got stuck in a loop of declaring completion "
                f"prematurely. You are now restarting with a clean context.\n\n"
                f"Please work through the requirements carefully.\n\n"
                f"Original instruction:\n{original_instruction}"
            )
        return await super()._query_llm(
            chat=chat,
            prompt=prompt,
            ctx=ctx,
            context_mgmt=context_mgmt,
            tools=tools,
            original_instruction=original_instruction,
            logging_paths=logging_paths,
        )

    @staticmethod
    def _extract_requirements(instruction: str) -> list[str]:
        """Extract bullet-point requirements from the original instruction.

        Looks for common patterns: bullet lists (*, -, •), numbered lists, and
        'We will test your build by:' / 'Requirements:' sections. Returns a
        deduplicated list of requirement strings, capped at max_checklist_items."""
        requirements: list[str] = []
        seen: set[str] = set()

        # Common section markers that introduce requirements
        section_markers = [
            "requirements:",
            "we will test your build by:",
            "acceptance criteria:",
            "you must:",
            "the solution should:",
            "the solution must:",
            "please ensure:",
            "your task is to:",
            "steps:",
        ]

        # Try to find requirement sections
        lower = instruction.lower()
        for marker in section_markers:
            idx = lower.find(marker)
            if idx < 0:
                continue
            # Find the end of this section (next blank line or another marker-like line)
            start = idx + len(marker)
            end = start
            lines = instruction[start:].splitlines()
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    break
                # Check if this line starts a new section
                if any(
                    stripped.lower().startswith(m)
                    for m in ["hints:", "notes:", "context:", "original instruction:"]
                ):
                    break
                end += len(line) + 1  # +1 for newline
            section = instruction[start : start + end].strip()
            if section:
                for line in section.splitlines():
                    cleaned = line.strip().lstrip("-*•0123456789.)").strip()
                    if cleaned and len(cleaned) > 10 and cleaned not in seen:
                        seen.add(cleaned)
                        requirements.append(cleaned)
                break  # Only use the first matching section

        # If no section found, fall back to scanning for bullet points in the
        # whole doc. Only the FIRST contiguous top-level bullet block is treated
        # as actionable requirements: an intervening non-bullet line (or a
        # deeper-indented bullet) ends the block, so descriptive sub-sections
        # like "The DES variant uses:" with their informational sub-bullets are
        # not extracted as requirements. Ghost requirements that can never be
        # keyword-marked [DONE] otherwise block termination forever.
        if not requirements:
            block_started = False
            base_indent: int | None = None
            for line in instruction.splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                # Match bullet points: lines starting with -, *, •, or numbered items
                is_bullet = re.match(r"^[\s]*[-*\•]\s+", stripped) or re.match(
                    r"^[\s]*\d+[.)]\s+", stripped
                )
                if is_bullet:
                    indent = len(line) - len(line.lstrip())
                    if base_indent is None:
                        base_indent = indent
                    if indent > base_indent:
                        # Deeper-indented sub-bullet of an already-collected
                        # item — descriptive detail, not a new requirement.
                        break
                    block_started = True
                    cleaned = re.sub(r"^[\s]*[-*\•\d.)]+\s*", "", stripped).strip()
                    if cleaned and len(cleaned) > 10 and cleaned not in seen:
                        seen.add(cleaned)
                        requirements.append(cleaned)
                elif block_started:
                    # A contiguous bullet block ended; stop before reaching any
                    # descriptive sub-bullets further down the document.
                    break

        return requirements[:20]  # Hard cap

    @staticmethod
    def _extract_acceptance_checks(instruction: str) -> str | None:
        """Pull the concrete acceptance-criteria section from a build/test task's
        instruction. Known phrasing in benchmark tasks: 'We will test your build
        by:' followed by bullet checks, ending at 'Hints:'. Returns a formatted
        numbered checklist, or None when the task supplies no such section
        (callers then keep baseline completion behavior, so open-ended tasks are
        unaffected)."""
        lower = instruction.lower()
        marker = "we will test your build by:"
        idx = lower.find(marker)
        if idx < 0:
            return None
        start = idx + len(marker)
        end = lower.find("hints:", start)
        if end < 0:
            end = len(instruction)
        section = instruction[start:end].strip()
        if not section:
            return None
        lines = []
        for line in section.splitlines():
            cleaned = line.strip().lstrip("-*•").strip()
            if cleaned:
                lines.append(cleaned)
        if not lines:
            return None
        numbered = "\n".join(f"  {i}. {ln}" for i, ln in enumerate(lines, 1))
        return f"The task's stated acceptance checks:\n{numbered}"

    # ---------- Progress tracking (from planning_checklist) ----------

    def _update_completed(
        self,
        parse: Any,
        tool_results: list[ToolResult],
        observation_text: str = "",
    ) -> None:
        """Mark requirements as completed when the current step's commands, tool
        output, or terminal observation mention keywords related to that
        requirement.

        Uses simple keyword overlap: if a requirement's significant words (nouns,
        verbs, identifiers) appear in the command text, tool output, or captured
        terminal text, consider it addressed. Identifiers are extracted from
        backtick-quoted signatures (`vector_dot(a, b)` -> vector_dot) so that
        evidence like "vector_dot normal: 32.0" matches. This is heuristic — the
        goal is to nudge the model, not to perfectly adjudicate."""
        if not self._requirements:
            return

        # Build a set of significant words per requirement
        req_keywords: list[set[str]] = []
        for req in self._requirements:
            words = set()
            # Bare identifiers/function names: `vector_dot(a, b)` yields keyword
            # vector_dot instead of a punctuation-laden token that never appears
            # verbatim in terminal evidence. Also picks up identifiers embedded
            # in paths (CMakeLists.txt -> cmakelists).
            for ident in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", req):
                if len(ident) > 3:
                    words.add(ident.lower())
            # Whitespace-delimited tokens with surrounding punctuation stripped
            # keep version markers ("Python 3.11" -> 3.11, "C++17" -> c++17) and
            # other non-identifier tokens that pure identifier splitting loses.
            for token in req.split():
                cleaned = re.sub(r"^[\s\W]+|[\s\W]+$", "", token)
                if len(cleaned) > 3:
                    words.add(cleaned.lower())
            # Discard very common words
            words -= {
                "that",
                "this",
                "with",
                "from",
                "have",
                "been",
                "will",
                "must",
                "should",
                "your",
                "the",
                "and",
                "for",
                "not",
                "are",
                "has",
                "had",
                "but",
                "its",
                "all",
                "can",
                "you",
                "our",
                "their",
                "them",
            }
            req_keywords.append(words)

        # Gather evidence from commands, tool output, and the terminal
        # observation text. For tmux-based tools ToolResult.output is empty —
        # the terminal pane (observation text) is the only place command output
        # appears, so it must be included or requirements verified via `cat` /
        # `grep` can never be marked done.
        evidence_text = ""
        if parse and hasattr(parse, "commands") and parse.commands:
            for cmd in parse.commands:
                evidence_text += " " + getattr(cmd, "keystrokes", str(cmd))
        for tr in tool_results or []:
            if tr.output:
                evidence_text += " " + tr.output
            if tr.error:
                evidence_text += " " + tr.error
        if observation_text:
            evidence_text += " " + observation_text

        evidence_lower = evidence_text.lower()

        # Check each uncompleted requirement
        for idx, keywords in enumerate(req_keywords):
            if idx in self._completed:
                continue
            # A requirement is considered addressed if at least 2 of its
            # significant keywords appear in the evidence text, or a single
            # keyword if the requirement is short (≤3 significant words).
            if not keywords:
                continue
            matched = sum(1 for kw in keywords if kw in evidence_lower)
            threshold = 2 if len(keywords) > 3 else 1
            if matched >= threshold:
                self._completed.add(idx)
                # Real progress was made — reset the pending-completion safety
                # valve so the gate re-arms with the remaining requirements.
                self._pending_completion_rejections = 0
                self._pending_gate_disabled = False

    def _build_progress_section(self) -> str:
        """Build a '[TASK PROGRESS]' section listing completed and pending
        requirements."""
        if not self._requirements:
            return ""

        completed_items = []
        pending_items = []
        for i, req in enumerate(self._requirements):
            prefix = "[DONE]" if i in self._completed else "[TODO]"
            item = f"  {prefix} {req}"
            if i in self._completed:
                completed_items.append(item)
            else:
                pending_items.append(item)

        progress_section = "[TASK PROGRESS]\n"
        if pending_items:
            progress_section += (
                "Pending requirements:\n" + "\n".join(pending_items) + "\n"
            )
        if completed_items:
            progress_section += (
                "Completed requirements:\n" + "\n".join(completed_items) + "\n"
            )
        progress_section += "\n"
        return progress_section

    def _get_pending_requirements_text(self) -> str:
        """Return a formatted string of all pending (TODO) requirements."""
        if not self._requirements:
            return ""
        pending = []
        for i, req in enumerate(self._requirements):
            if i not in self._completed:
                pending.append(f"  • {req}")
        if not pending:
            return ""
        return "Pending requirements:\n" + "\n".join(pending)

    # ---------- Evidence-gated completion (from completion_integrity_guard, enhanced) ----------

    @staticmethod
    def _assemble_observation(obs_result: Any, tool_results: list[ToolResult]) -> str:
        """Same tool-output + terminal-pane assembly as the baseline's
        _build_observation_text, factored out for reuse."""
        tool_output_parts: list[str] = []
        for tr in tool_results:
            if tr.output:
                tool_output_parts.append(tr.output)
            elif tr.error:
                tool_output_parts.append(f"[tool error] {tr.error}")
        obs_pieces: list[str] = []
        if tool_output_parts:
            obs_pieces.append("\n\n".join(tool_output_parts))
        if obs_result.text:
            obs_pieces.append(obs_result.text)
        return "\n\n".join(obs_pieces)

    def _build_observation_text(
        self,
        *,
        parse: Any,
        obs_result: Any,
        tool_results: list[ToolResult],
        state: AgentLoopState,
        pending_completion: bool,
    ) -> tuple[str, bool]:
        """Evidence-gate completion whenever requirements are pending: a
        task_complete declaration that still has pending requirements never
        enters the two-phase confirmation (whether or not that step executed
        commands — commands alone don't demonstrate every stated requirement).
        Instead the model sees the pending requirements and is told to address
        them before completing.

        For tasks WITH explicit acceptance checks (from
        _extract_acceptance_checks), also quote those checks (a 0-command
        completion is always rejected for those tasks). For tasks WITHOUT
        pending requirements, fall through to the baseline two-phase gate.
        """
        if not parse.is_task_complete:
            return super()._build_observation_text(
                parse=parse,
                obs_result=obs_result,
                tool_results=tool_results,
                state=state,
                pending_completion=pending_completion,
            )

        # Check if there are pending requirements to show
        pending_text = self._get_pending_requirements_text()
        zero_cmd = len(parse.commands) == 0
        if pending_text or (self._acceptance_checklist and zero_cmd):
            # Anti-hang safety valve: after many rejections with no progress, fall
            # back to the baseline two-phase gate rather than looping forever on a
            # heuristic requirement that can't be keyword-marked.
            self._pending_completion_rejections += 1
            if (
                self._pending_gate_disabled
                or self._pending_completion_rejections
                > self.max_pending_completion_rejections
            ):
                self._pending_gate_disabled = True
                return super()._build_observation_text(
                    parse=parse,
                    obs_result=obs_result,
                    tool_results=tool_results,
                    state=state,
                    pending_completion=pending_completion,
                )
            state.consecutive_complete_signals = 0
            combined_obs_text = self._assemble_observation(obs_result, tool_results)

            prompt_parts = [
                "You declared the task complete, but the following requirements are "
                "still pending:\n\n",
            ]
            if pending_text:
                prompt_parts.append(pending_text)
                prompt_parts.append("\n")
            if self._acceptance_checklist:
                prompt_parts.append(self._acceptance_checklist)
                prompt_parts.append("\n\n")
            else:
                prompt_parts.append("\n")

            prompt_parts.append(
                "Address each pending requirement with real commands "
                "(compile/build/run/test as appropriate) and paste the actual output. "
                "Only after every requirement is demonstrably met should you emit "
                "task_complete=true again."
            )

            prompt = "".join(prompt_parts)
            if combined_obs_text:
                prompt += f"\n\nLatest terminal output:\n{combined_obs_text}"
            return prompt, pending_completion

        # Preserve the baseline two-phase gate after all requirements are met.
        self._pending_completion_rejections = 0
        return super()._build_observation_text(
            parse=parse,
            obs_result=obs_result,
            tool_results=tool_results,
            state=state,
            pending_completion=pending_completion,
        )

    @staticmethod
    def _is_readonly_command(keystrokes: str) -> bool:
        """Heuristic: is this command unlikely to create/modify files or run builds?"""
        stripped = keystrokes.strip().lower()
        # Anything that writes to a file or runs a script is productive
        if any(
            marker in stripped
            for marker in (
                "cat >",
                ">",
                "tee",
                "chmod",
                "touch",
                "python3",
                "bash ",
                "./",
            )
        ):
            return False
        # Check against known read-only prefixes
        for prefix in _READONLY_PREFIXES:
            if stripped.startswith(prefix):
                return True
        return False

    @staticmethod
    def _extract_written_file(keystrokes: str) -> str | None:
        """Extract the destination file path from a write command, or None if
        the command does not write to a file. Handles common patterns:
        - cat > /path/to/file
        - echo ... > /path/to/file
        - > /path/to/file
        - >> /path/to/file  (append — still a write)
        - tee /path/to/file
        - cp source /path/to/file
        - mv source /path/to/file
        """
        stripped = keystrokes.strip()
        # cat > /path/to/file
        m = re.match(r"^cat\s+>\s*(\S+)", stripped)
        if m:
            return m.group(1)
        # echo ... > /path/to/file  or  > /path/to/file
        m = re.search(r"(?:^|\s+)>\s*(\S+)", stripped)
        if m:
            return m.group(1)
        # tee /path/to/file
        m = re.search(r"(?:^|\s+)tee\s+(\S+)", stripped)
        if m:
            return m.group(1)
        # cp source /path/to/file  — return the last argument
        m = re.match(r"^cp\s+", stripped)
        if m:
            parts = stripped.split()
            if len(parts) >= 3:
                return parts[-1]
        # mv source /path/to/file  — return the last argument
        m = re.match(r"^mv\s+", stripped)
        if m:
            parts = stripped.split()
            if len(parts) >= 3:
                return parts[-1]
        return None

    @staticmethod
    def _has_artifact_evidence(observation_text: str) -> bool:
        """Check if terminal output shows evidence of artifact creation."""
        lower = observation_text.lower()
        for marker in _READONLY_ARTIFACT_MARKERS:
            if marker.lower() in lower:
                return True
        return False

    @staticmethod
    def _has_background_process_stop_signal(observation_text: str) -> bool:
        """Check if terminal output shows a background process was stopped or killed.

        Common patterns in asciinema/tmux terminal output:
        - '[1]+  Stopped' (SIGTTOU when backgrounding in a job-controlled shell)
        - 'Killed' (SIGKILL)
        - 'core dumped' (SIGABRT/SIGSEGV)
        - 'Terminated' (SIGTERM)
        - 'Aborted' (SIGABRT)
        """
        lower = observation_text.lower()
        # Check for the specific '[N]+  Stopped' pattern (SIGTTOU on background jobs)
        if re.search(r"\[\d+\]\+\s+Stopped", observation_text):
            return True
        # Check for job termination signals on their own lines
        stop_patterns = [
            r"^Killed$",
            r"core dumped",
            r"^Terminated$",
            r"^Aborted$",
            r"\[\d+\]\+?\s+Terminated",
            r"\[\d+\]\+?\s+Killed",
        ]
        for pattern in stop_patterns:
            if re.search(pattern, observation_text, re.MULTILINE):
                return True
        return False
        """Check if terminal output shows evidence of artifact creation."""
        lower = observation_text.lower()
        for marker in _READONLY_ARTIFACT_MARKERS:
            if marker.lower() in lower:
                return True
        return False

    async def _execute_commands(
        self, parse: Any, tools: ToolSet, ctx: ModuleCtx, *, iteration: int
    ) -> tuple[list[ToolResult], ToolResult | None]:
        """Override: block execution of repeated commands when the repeat nudge
        has already been injected. Returns a synthetic [COMMAND BLOCKED] result
        instead of executing the blocked command, which forces the LLM to try
        a different approach."""
        commands = parse.commands
        if commands and self._last_injected_repeat_nudge:
            cmd_keys = tuple(cmd.keystrokes.strip() for cmd in commands)
            # Check if this command matches the last repeated command
            if len(self._command_history) >= 2:
                last = self._command_history[-1]
                if cmd_keys == last:
                    # Block this repeated command — keep _last_injected_repeat_nudge=True
                    # so _shape_observation doesn't re-inject a duplicate nudge.
                    # The flag resets naturally at line 738-741 when command changes.
                    blocked_msg = (
                        f"[SYSTEM — COMMAND BLOCKED]\n"
                        f"You have already executed this command multiple times consecutively. "
                        f"Execution of '{cmd_keys[0] if len(cmd_keys) == 1 else str(cmd_keys)}' "
                        f"has been blocked to prevent an infinite loop.\n\n"
                        f"Read the error output from the previous run and try a different "
                        f"approach: different flags, a different tool, or fix the code.\n"
                    )
                    # ToolResult exposes only output, error and success.
                    tool_result = ToolResult(
                        output=blocked_msg,
                        error="",
                        success=True,
                    )
                    return [tool_result], tool_result

        return await super()._execute_commands(
            parse=parse, tools=tools, ctx=ctx, iteration=iteration
        )

    def _shape_observation(
        self,
        observation_text: str,
        *,
        parse: Any,
        tool_results: list[ToolResult],
        state: AgentLoopState,
        ctx: ModuleCtx,
    ) -> str:
        """Update progress tracking after each step's commands execute, then
        apply stuck-detection nudges (empty completions, repeated commands,
        read-only inspection). Structural enforcement: after the nudge threshold
        is exceeded, return a synthetic [SYSTEM — COMMAND BLOCKED] observation
        instead of just appending a warning."""
        # --- 0. Update progress tracking first ---
        self._update_completed(parse, tool_results, observation_text)

        # --- 1. Empty completion detection with structural enforcement ---
        if parse.is_task_complete and len(parse.commands) == 0:
            self._consecutive_empty_completions += 1
            self._total_empty_completions += 1  # cumulative, never reset
            # After max_empty_completions, switch from advisory to structural enforcement
            if self._consecutive_empty_completions >= self.max_empty_completions:
                self._needs_reorientation = True
                # Check if we need a hard reset — use cumulative total to break alternation loops
                if (
                    self._total_empty_completions
                    >= self.max_empty_completions_before_reset
                ):
                    self._needs_hard_reset = True
                    warning = (
                        f"\n\n[SYSTEM — HARD RESET TRIGGERED]\n"
                        f"You have declared the task complete without executing "
                        f"any commands {self._total_empty_completions}x total ("
                        f"{self._consecutive_empty_completions}x consecutively), "
                        f"despite repeated warnings. The conversation will be restarted "
                        f"with a fresh context to break this loop.\n"
                    )
                    return observation_text + warning
                # Structural enforcement: replace the observation with a blocked message
                # that includes the original observation text so the LLM still sees it
                blocked = (
                    f"[SYSTEM — COMMAND BLOCKED]\n"
                    f"You have declared the task complete without executing "
                    f"any commands ({self._consecutive_empty_completions}x consecutively, "
                    f"{self._total_empty_completions}x total). "
                    f"This empty completion declaration has been blocked. "
                    f"You MUST execute actual commands to complete the task. "
                    f"Do NOT declare completion again until you have done the work.\n\n"
                    f"{observation_text}"
                )
                return blocked
            # Below threshold: just append a warning
            warning = (
                f"\n\n[SYSTEM] You have declared the task complete without executing "
                f"any commands ({self._consecutive_empty_completions}x consecutively, "
                f"{self._total_empty_completions}x total). "
                f"Declaring completion without performing the work will be ignored. "
                f"Execute the commands needed to finish the task, then declare completion."
            )
            return observation_text + warning
        self._consecutive_empty_completions = 0
        # _total_empty_completions is NOT reset here — it's cumulative across the whole run

        # --- 2. Repeated-command detection ---
        commands = parse.commands
        if commands:
            cmd_keys = tuple(cmd.keystrokes.strip() for cmd in commands)
            if cmd_keys:
                self._command_history.append(cmd_keys)
                if len(self._command_history) >= 2:
                    last = self._command_history[-1]
                    count = 0
                    for entry in reversed(self._command_history):
                        if entry == last:
                            count += 1
                        else:
                            break
                    self._consecutive_identical_commands = count
                    if (
                        count >= self.max_repeated_commands
                        and not self._last_injected_repeat_nudge
                    ):
                        self._last_injected_repeat_nudge = True
                        self._consecutive_readonly_steps = 0  # reset read-only guard
                        nudge = (
                            f"\n\n[SYSTEM — STUCK LOOP DETECTED]\n"
                            f"You have issued the same command {count} times consecutively. "
                            f"Consider a different approach, flags, or tool.\n"
                            f"Repeat: {last[0] if len(last) == 1 else str(last)}\n\n"
                            f"If you repeat this command again, execution will be blocked and "
                            f"you will be forced to try a different approach."
                        )
                        return observation_text + nudge

        # Reset repeat nudge flag when a different command is seen
        if (
            len(self._command_history) >= 2
            and self._command_history[-1] != self._command_history[-2]
        ):
            self._last_injected_repeat_nudge = False
            self._consecutive_identical_commands = 0

        # --- 3. Read-only inspection guard ---
        if commands:
            all_readonly = all(
                self._is_readonly_command(cmd.keystrokes) for cmd in commands
            )
            no_artifact_evidence = not self._has_artifact_evidence(observation_text)
            if all_readonly and no_artifact_evidence:
                self._consecutive_readonly_steps += 1
                if (
                    self._consecutive_readonly_steps >= self.max_readonly_steps
                    and not self._last_injected_readonly_nudge
                ):
                    self._last_injected_readonly_nudge = True
                    self._consecutive_identical_commands = 0  # reset repeat guard
                    nudge = (
                        f"\n\n[SYSTEM — PROGRESS CHECK]\n"
                        f"You have spent {self._consecutive_readonly_steps} consecutive turns "
                        f"inspecting without creating or running any task files. This task "
                        f"requires you to WRITE and TEST the required scripts/commands. "
                        f"Stop inspecting and start implementing: write the file, run it, "
                        f"and verify it works. If a command errors, read the error, fix it, "
                        f"and re-run."
                    )
                    return observation_text + nudge
            else:
                self._consecutive_readonly_steps = 0
                self._last_injected_readonly_nudge = False

        # --- 4. Consecutive command failure detection ---
        if tool_results:
            # Check if ANY command this step failed (non-zero exit or tool error)
            any_failed = any(not tr.success for tr in tool_results)
            if any_failed:
                self._consecutive_failed_commands += 1
                if (
                    self._consecutive_failed_commands >= self.max_failed_commands
                    and not self._last_injected_failure_nudge
                ):
                    self._last_injected_failure_nudge = True
                    self._consecutive_identical_commands = 0  # reset repeat guard
                    nudge = (
                        f"\n\n[SYSTEM — COMMAND FAILURE DETECTED]\n"
                        f"You have had {self._consecutive_failed_commands} consecutive "
                        f"command failures. Read the error output carefully — it contains "
                        f"the exact reason the command failed. Try a different approach: "
                        f"fix the syntax, install missing dependencies, or use a different "
                        f"tool. Repeating the same failing command will not make it work."
                    )
                    return observation_text + nudge
            else:
                # A step with no failures resets the counter
                self._consecutive_failed_commands = 0
                self._last_injected_failure_nudge = False

        # --- 5. Background process stop detection ---
        if self._has_background_process_stop_signal(observation_text):
            nudge = (
                "\n\n[SYSTEM — BACKGROUND PROCESS STOPPED]\n"
                "The terminal shows that a background process was stopped or killed "
                "(SIGTTOU / SIGKILL / SIGTERM). This likely means the server or "
                "process you started in the background failed to start or was "
                "terminated. Instead of using `&` to background the process, try:\n"
                "  - Running it without backgrounding (in the foreground)\n"
                "  - Using `nohup` or `setsid` to keep it alive\n"
                "  - Using `tmux` or `screen` to run the process persistently\n\n"
                "Read the error message above and fix the underlying issue before "
                "re-running."
            )
            return observation_text + nudge

        # --- 6. File-write tracking (prevent overwriting correct output) ---
        if commands and tool_results:
            # Check if any command wrote to a file and succeeded
            all_successful = all(tr.success for tr in tool_results)
            if all_successful:
                for cmd in commands:
                    file_path = self._extract_written_file(cmd.keystrokes)
                    if file_path and file_path not in self._file_write_nudge_injected:
                        self._file_write_count[file_path] = (
                            self._file_write_count.get(file_path, 0) + 1
                        )
                        if self._file_write_count[file_path] >= self.max_file_writes:
                            self._file_write_nudge_injected.add(file_path)
                            nudge = (
                                f"\n\n[SYSTEM — FILE WRITE TRACKING]\n"
                                f"`{file_path}` has been written "
                                f"{self._file_write_count[file_path]} times this session. "
                                f"The output appears correct (no errors). "
                                f"If you are satisfied that the file is correct, "
                                f"consider declaring completion rather than overwriting it again."
                            )
                            return observation_text + nudge

        return observation_text

    # ---------- Prompt shaping: planning injection + re-orientation ----------

    def _pre_llm_prompt(
        self, prompt: str, *, iteration: int, state: AgentLoopState, ctx: ModuleCtx
    ) -> str:
        """Inject a 'Task Progress' section, verifier feedback, and/or a re-orientation block."""
        # First, inject any verifier rejection reason from the previous iteration
        if self._last_verifier_reason is not None:
            reason = self._last_verifier_reason
            self._last_verifier_reason = None  # consume it
            block = (
                "[SYSTEM — VERIFICATION RESULT]\n"
                "The task verifier reviewed your last completion declaration and "
                "rejected it for the following reason:\n"
                f"{reason}\n\n"
                "Re-read the original instruction below:\n"
                f"{self._original_instruction}\n\n"
                "Use this feedback to fix the specific issue. Do NOT simply re-declare "
                "completion — read the reason, determine what is still wrong, and "
                "continue working on it.\n\n"
            )
            prompt = block + prompt
        if self._needs_reorientation:
            self._needs_reorientation = False
            block = (
                "[SYSTEM — RE-ORIENTATION]\n"
                "You have declared the task complete multiple times, but the task is "
                "NOT finished. Stop declaring completion. Re-read the original "
                "instruction carefully, determine what is missing or still wrong, and "
                "continue working to fix it.\n\n"
                f"Original instruction: {self._original_instruction}\n\n"
            )
            prompt = block + prompt

        # Then inject task progress section (from planning_checklist)
        progress_section = self._build_progress_section()
        if progress_section:
            prompt = progress_section + prompt

        return prompt

    # ---------- Termination guard ----------

    def _should_continue(
        self,
        *,
        terminate: bool,
        reason: str,
        parse: Any,
        iteration: int,
        state: AgentLoopState,
        ctx: ModuleCtx,
    ) -> bool:
        # Evidence gate: any completion declared while requirements are pending is
        # never sufficient to stop the loop — with or without commands — because
        # commands alone don't demonstrate every stated requirement. For tasks
        # with an acceptance-checklist section only, keep rejecting 0-command
        # completions as before (commands at least show an attempt at the checks).
        if terminate and parse.is_task_complete and not self._pending_gate_disabled:
            has_pending = bool(self._get_pending_requirements_text())
            if has_pending or (self._acceptance_checklist and len(parse.commands) == 0):
                ctx.services.logger.warning(
                    "PlanningWithGuard: rejecting completion declaration with "
                    "pending requirements (iteration %d)",
                    iteration,
                )
                state.consecutive_complete_signals = 0
                return True  # keep looping

        # Forced termination: if the model has declared task_complete twice
        # consecutively (the two-phase confirmation gate is satisfied), and
        # there are no pending requirements (the evidence gate above did not
        # block), terminate regardless of what the verifier says.  This
        # prevents budget exhaustion when the work is actually complete but
        # the verifier is imperfect (e.g. returns False for an empty reason
        # string, or the verifier has a false-negative).
        #
        # The two-phase gate is a strong signal — the model has confirmed
        # completion twice in a row, which is the same bar the baseline uses
        # before it even consults the verifier.  If the work is truly
        # incomplete, the evidence gate above will have already reset
        # consecutive_complete_signals to 0 (because pending requirements
        # exist), so this check will not fire.
        if parse.is_task_complete and state.consecutive_complete_signals >= 2:
            ctx.services.logger.info(
                "PlanningWithGuard: forced termination after 2 consecutive "
                "task_complete declarations (iteration %d)",
                iteration,
            )
            return False

        # Track completions the verifier rejected, for re-orientation.
        if parse.is_task_complete:
            if not terminate:
                self._consecutive_rejected_completions += 1
                # Stash the verifier's reason so it can be surfaced to the LLM
                # on the next iteration via _pre_llm_prompt.
                self._last_verifier_reason = reason
                if (
                    self._consecutive_rejected_completions
                    >= self.max_rejected_completions
                ):
                    self._needs_reorientation = True
        else:
            self._consecutive_rejected_completions = 0
        return not terminate


def register(library):
    library.register(
        type_="agent_loop",
        name=PlanningWithGuardLoop.NAME,
        factory=lambda params: PlanningWithGuardLoop(
            max_iterations=int(params.get("max_iterations", 1_000_000)),
            llm_call_kwargs=params.get("llm_call_kwargs") or {},
            raw_content=bool(params.get("raw_content", False)),
            max_checklist_items=int(params.get("max_checklist_items", 20)),
            max_empty_completions=int(params.get("max_empty_completions", 2)),
            max_rejected_completions=int(params.get("max_rejected_completions", 2)),
            max_repeated_commands=int(params.get("max_repeated_commands", 3)),
            max_failed_commands=int(params.get("max_failed_commands", 2)),
            max_readonly_steps=int(params.get("max_readonly_steps", 3)),
            _command_history_size=int(params.get("_command_history_size", 5)),
            max_pending_completion_rejections=int(
                params.get("max_pending_completion_rejections", 5)
            ),
            max_empty_completions_before_reset=int(
                params.get("max_empty_completions_before_reset", 6)
            ),
            max_file_writes=int(params.get("max_file_writes", 3)),
        ),
        description=PlanningWithGuardLoop.DESCRIPTION,
        params_schema=PlanningWithGuardLoop.PARAMS_SCHEMA,
        niche=PlanningWithGuardLoop.NICHE,
    )
