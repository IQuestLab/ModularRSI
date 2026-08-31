---
name: failure-mode-checklist
description: How to diagnose WHY a parent gen failed from its trajectories — where to look and how to reason. Describes what each module is RESPONSIBLE for and what to check in the trace; it does not map a symptom to a module or prescribe a fix. Both the cause and the fix follow from YOUR reading of the actual evidence.
---

# Failure diagnosis aid

This skill helps you get from "a trial failed" to "which module, if any, is
responsible — and is it even worth changing." It is a DIAGNOSTIC aid only:
it points you at where to look and what to check. It deliberately gives **no
canned fixes and names no preferred module** — the right change (and whether to
change at all) follows from the specific evidence YOU read in the real
trajectory, not from a lookup table. The same symptom has different causes on
different tasks.

## What each module is RESPONSIBLE for — and what to check in the trace

Below is what each module OWNS, plus trace signals that bear on it. This is NOT
a symptom→module lookup: the same observation can come from any of several
modules, and one symptom does not imply one module. Use it to know what each
module governs, then attribute from the evidence YOU read.

| Module | The concern it owns | Signals in the raw trace that bear on it |
|---|---|---|
| `observation` | how raw terminal output becomes what the agent sees each step (capture timing, selection, truncation, layout) | per-step obs size; obs empty/cut-off/stale right after a command; identical consecutive obs; noise drowning the signal |
| `context_mgmt` | when / how the chat history is compressed or summarized | did compression ever fire? `context_compression_count` on long, high-token trials; was a needed prior result summarized away? |
| `tools` | the action surface: command grammar, response parser, execution/timing | which responses failed to parse and why; commands that mis-timed; a recurring multi-command routine done the hard way |
| `verification` | the termination decision — stop vs keep going | did it stop with work unfinished, or churn past done? what did it treat as "done"? |
| `agent_loop` | the think→act→observe control loop and its planning/retry strategy | many tool-calls with no progress; no structure on a long task — but check observation/context_mgmt FIRST (see reverse diagnostic) |

This table tells you what each module governs and what to read; it does NOT tell
you the cause or the fix — attribute those from the real trajectory. A single
observed symptom (e.g. "loops without finishing," "misses something the verifier
catches") is consistent with several of these modules; which one it is depends on
what the agent actually SAW, DID, and REMEMBERED, not on a table row.

## How to read trajectories

The trajectory you're shown contains `steps`, each with:

- `source`: `user` (initial prompt), `agent` (LLM response + tool calls),
  `system` (summarization handoff).
- `message`: structured `Analysis: ...\nPlan: ...` (or raw response if
  `raw_content=true`).
- `tool_calls`: list of `bash_command` / `mark_task_complete`.
- `observation.results[].content`: what fed back to the LLM as the next prompt.
- `metrics`: tokens used in this step.

Useful things to grep / scan:

- Final step's `observation.results[].content` — what did the LLM last see?
- Any step with `is_copied_context: true` — summarization happened nearby.
- Step messages containing "WARNINGS:" — parser warnings.
- Step messages containing "ERROR:" — parser errors.
- Big gap in `tool_calls` across steps — agent thinking but not acting.

## From diagnosis to a fix — directions, not recipes

Once the evidence points at a suspect module, the actual fix is yours to design
from the specific failure you saw. There are three shapes of change; the cause
decides which fits (adding a variant is the DEFAULT for any behavior change):

- **Add a new module variant** (the default) — create a fresh
  `modules/<type>/<name>.py`; NEVER overwrite the live impl. The old variant
  stays in the library, so you can't regress the tasks it handled; the Composer
  decides what goes live. Right whenever you change what a module DOES (any
  change to a module's behavior, in any of the five). See the `adding-a-variant`
  skill.
- **Build a new tool / capability** — right when the agent has no mechanism for a
  recurring situation at all (a transform it needs, a search/read tool, a check
  it never makes). No prompt-tuning creates a capability that isn't there. The
  `adding-a-solver-tool` skill is the cheap, bounded way to add one (a helper
  command, no grammar/parser change).
- **Tune in place** (`<edit_file>`) — ONLY for a bug fix or a param/threshold
  tweak inside a variant whose behavior you are NOT otherwise changing. If you're
  changing the approach, that's a new variant (above), not an in-place edit.

"The agent can already do it with raw shell" does NOT mean nothing is missing:
doing something repeatedly and clumsily IS a gap a tool can close. Equally, do
not invent a tool for something done once — match the shape to the cause.

Per-module concerns (what each owns — attribute from evidence, not from a
symptom trigger):

- **observation** — how terminal output is selected / truncated / laid out into
  what the agent sees.
- **context_mgmt** — when / how the chat history is compressed or summarized.
- **tools** — the action surface: command grammar, the response parser,
  execution and timing, and which commands exist at all.
- **verification** — the termination condition: when the loop is allowed to stop.
- **agent_loop** — the control loop's own strategy: retries, parse-error
  fallbacks, completion logic, and task-level planning. See the reverse
  diagnostic before editing it — the agent's narration over-represents the loop.

## Reverse diagnostic — when the problem is probably NOT the loop

If a failed trial shows `bash_command` × 30+ with no progress, AND
`context_compression_count` = 0, AND large average `obs_lengths`, don't assume
the loop's prompt is at fault just because the agent narrates its reasoning
there. Read what the agent SAW and REMEMBERED each turn before concluding: could
it even tell "I already tried this" from "I should try this again"? Attribute to
whatever the evidence shows — the loop is only one candidate.

In general: if the agent makes many tool calls on a simple task and still fails,
"add more text to the loop's prompt" is rarely the cause or the fix. Look at what
the agent saw and what it remembered first, then attribute from there.

## The ACT gate — decide whether to change at all, and verify before you do

Diagnosing a cause is NOT the same as having a fix. Before you edit anything,
pass all three checks — failing any means "no change" is the correct outcome:

1. **Is it module-attributable?** Some failures are the LLM simply reasoning
   wrong on a specific task — it had the right input, correct observations, and
   working tools, and still chose a wrong move/answer. Those are NOT fixable by
   editing modules. Commit with NO edits. Do not invent a marginal module tweak
   just to feel productive — an unjustified change only risks a regression and
   burns the round. (Committing no edits is a legitimate, expected outcome.)

2. **Can you name the causal link?** State it concretely from the evidence:
   "failure <X> is caused by <module>'s <behavior>; my change fixes it by <how>."
   If you can't fill that in, you haven't found the cause — keep investigating or
   change nothing.

3. **Is the lever you're about to pull actually binding?** Verify from the
   trajectory data that the param / threshold / path you're changing was actually
   active in the failures. Changing a lever the data shows was never exercised
   does literally nothing.

## Cautions

- One bad trajectory ≠ a module bug. Look for a pattern across multiple
  trajectories before claiming a module is the cause.
- **Match the fix to the cause, not to what's easiest to write.** A tunable flaw
  → tune it; a missing capability → build it. Reaching for "add one more line of
  prompt" because it is the smallest edit, when the cause is elsewhere, leaves
  the cause in place and the failure recurs.
- The generation history tells you what was tried and whether it stuck; use it to
  avoid repeating a falsified direction, not as an order to switch. It does not
  dictate which module to touch next — that follows from your diagnosis of the
  current failures.
