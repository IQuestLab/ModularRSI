"""Outer-loop self-evolution driver for terminus-2-modular.

Not part of the agent's runtime — runs OUTSIDE Harbor's job system. The driver:

1. Picks a parent generation (a `modules/` snapshot).
2. Copies it into a staging directory.
3. Invokes the Editor agent on staging with the task instruction.
4. Smoke-tests staging (AST diff vs Kernel; importable; passes hello-world).
5. On success: atomic mv staging → `gen_archive/gen_{N+1}/`.
   On failure: rm -rf staging, log reason, give editor another shot.

This package is intentionally separate from the modules themselves — modules
get evolved; the evo driver does NOT.
"""
