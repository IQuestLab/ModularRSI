---
name: editing-conventions
description: How to use the editor actions (glob / grep / read_file / edit_file / edit_lines / create_file / validate / commit_patch). Read this before emitting ANY action.
---

# Editing conventions

You have these actions (XML inside `<actions>...</actions>`):
`glob`, `grep`, `read_file`, `edit_file`, `edit_lines`, `create_file`,
`validate`, `commit_patch`.

## The inner loop (do this every reflection)

```
locate    →  grep / glob to find the relevant code
read      →  read_file (with line numbers; range-read big regions)
change    →  edit_file (unique string) or edit_lines (by line number) or create_file
VALIDATE  →  <validate/>  ← self-check BEFORE committing
fix       →  if validate reports errors, fix them and validate again
commit    →  <commit_patch/> only after validate PASSES
```

`<validate/>` is your safety net — it runs the same fast checks the harness
runs after you commit (AST, imports, library load, bundle config, discovery and
static contract). Catching a
mistake here costs one action; catching it after commit wastes a whole
generation. **Never `commit_patch` without a passing `validate` first.**

You have these actions. Use them as XML inside `<actions>...</actions>`.

## `<grep pattern="..." [path="..."]/>`

Search all `.py` files under staging for a string/regex. Returns matching
`path:line: text`. Optional `path` narrows the search to one file or subdir.

```xml
<grep pattern="PARAMS_SCHEMA"/>
```

- **Use grep to verify claims about the code before acting on them.** In
  particular, to decide whether something is "unused" / "dead" / "never
  called", grep for it — you will see BOTH its definition line and every call
  site. A single `read_file` is NOT enough for this, because large files are
  truncated in the middle (see below) and the call site may be off-screen.
- Returns up to 100 match lines.

## `<read_file path="..." [start_line="N"] [end_line="M"]/>`

Read a file's contents, shown **with line numbers**.

```xml
<read_file path="modules/<type>/<impl>.py"/>
<read_file path="modules/<type>/<impl>.py" start_line="380" end_line="440"/>
```

- `path` is RELATIVE to the staging directory.
- Always read a file BEFORE editing it — otherwise you don't know the exact
  content you need to match in `<edit_file>`.
- **Files over ~32k chars are TRUNCATED in the middle.** You'll see a
  `... [lines X-Y omitted — read them with start_line/end_line, or use <grep>] ...`
  marker. That middle region EXISTS — it is not empty. To see it, re-read with
  `start_line`/`end_line`, or `<grep>` for the symbol you care about.
  **Never conclude "this code is missing / never called" from a truncated
  read** — the part you didn't see is exactly where the call might be.
- Output is added to the next observation; you'll see it in the prompt.

## `<edit_file path="..."> ... </edit_file>`

Replace one occurrence of `old_string` with `new_string`. (The example below
uses placeholder names to show the replace SYNTAX only — it is not a suggested
change; what to edit follows from your diagnosis.)

```xml
<edit_file path="modules/<type>/<impl>.py">
  <old_string>SOME_THRESHOLD = 10</old_string>
  <new_string>SOME_THRESHOLD = 25</new_string>
</edit_file>
```

### Rules

1. **`old_string` must exist in the file** — otherwise the action fails with
   `old_string not found`.
2. **`old_string` must be UNIQUE in the file** — otherwise the action fails
   with `old_string occurs N times; include more context`. Add surrounding
   lines until the match is unique.
3. **Whitespace matters** — exact bytes, including indentation and trailing
   spaces.
4. Don't escape newlines / quotes inside the XML; the parser passes the raw
   text through.

### Common mistakes

- Trying to match a one-liner like `return None` that appears in 5 functions.
  → Include the surrounding function signature: 
  ```xml
  <old_string>    def setup(self):
          return None</old_string>
  ```
- Reading the file once, editing many times. Your second edit may invalidate
  the first edit's context. Re-read after major edits if unsure.

## `<create_file path="..."> ... </create_file>`

Create a new file. Fails if the file already exists (no overwrite). **This is the
PRIMARY way you evolve a module: add a new variant rather than overwriting the
live one (see the `adding-a-variant` skill).** (The example below uses placeholder
names to show the file STRUCTURE a new module needs — the Protocol methods + a
top-level `register()`. It is not a suggested design; what to build, and for which
module type, follows from your diagnosis.)

```xml
<create_file path="modules/<type>/<your_impl>.py">
  <content>
"""One-line summary of what this implementation does."""

from __future__ import annotations
# import ModuleCtx + the IO dataclasses your chosen Protocol uses
from harbor.agents.terminus_2_modular.protocols import ModuleCtx


class YourImpl:
    NAME = "your_impl"           # referenced by the Composer to select this variant
    DESCRIPTION = "..."          # one line — the Composer READS this to pick the variant
    PARAMS_SCHEMA = {}           # tunable params, or {}

    def __init__(self, **params):
        ...

    # implement the Protocol method(s) for <type>, matching the EXACT
    # signature from protocols.py (see the module-architecture skill).


def register(library):
    library.register(
        type_="<type>",          # MUST match the directory / Protocol
        name=YourImpl.NAME,
        factory=lambda params: YourImpl(**params),
        description=YourImpl.DESCRIPTION,
        params_schema=YourImpl.PARAMS_SCHEMA,
    )
  </content>
</create_file>
```

### Rules

- `path` must be under a valid `modules/<type>/` subdirectory.
- `path` cannot already exist.
- The file is auto-discovered next startup; you don't need to register it
  in `library.py`.
- Provide a `register(library)` function at module level so the library
  finds the new module.

## `<validate/>`

Self-check the staged changes WITHOUT committing — runs AST, imports, library
load, bundle config, discovery and static-contract checks (the same fast battery
the harness runs after commit) and returns PASS or the exact failures. **Run this after editing and
before `commit_patch`.** If it reports errors, fix them and validate again.
This is cheap (one action) and saves a wasted generation.

## `<commit_patch/>`

Signal that you're done editing. The harness then runs its gates on your staged
changes (a smoke battery + a review pass + a crash check). Emit this ONLY when:

- You've made all the edits you want.
- Each edit succeeded (no `error` in the previous observation).
- **`<validate/>` PASSED** on your latest state — do not commit on a failing
  or un-run validate.

After `<commit_patch/>`, also set `<task_complete>true</task_complete>` to
end the loop. The outer driver takes over from there.

## Iteration discipline

A typical session looks like:

```
turn 1: grep/glob for the symbols / failure-related code you suspect; read_file the file(s)
turn 2: if a read came back TRUNCATED, range-read or grep the region you need
        (e.g. to confirm whether a method is actually called) BEFORE concluding anything
turn 3: edit_file (unique string) or edit_lines (by line number) to make the change
turn 4: read_file the same region to verify the edit landed
turn 5: <validate/> — if it FAILS, fix and validate again until it PASSES
turn 6: commit_patch + task_complete=true
```

Budget your turns: don't read every file in the library — locate with
grep/glob, read only what you need, then change + validate + commit.

Don't try to do everything in one turn. Each turn produces an observation
that informs the next turn.

## When edits fail

If you get `old_string not found`:
- Re-read the file.
- Look for differences in indentation or surrounding text.
- Match a smaller, more specific snippet.

If you get `old_string occurs N times`:
- Expand the snippet to include uniquely-identifying surrounding text.

If you get `path is off-limits (Kernel)`:
- See the `kernel-boundaries` skill. You can't edit that file. Choose a
  different file in `modules/`.
