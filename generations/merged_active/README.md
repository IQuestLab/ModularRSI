# Merged active modular generation

This directory is a sanitized, runnable modular-agent generation assembled from
previously selected self-evolution variants. It contains implementation source
and archive metadata only—no benchmark results, trajectories, credentials,
runtime logs, or private Git history.

## Layout

```text
merged_active/
├── archive.json
├── PROVENANCE.json
└── gen_0/
    └── modules/
```

The `<run>/gen_0/modules` layout is intentional. `LLMComposer` resolves
`archive.json` from the run root and filters variants whose status is
`superseded` or `excluded`.

## Contents

The library registers 20 implementations:

- 3 agent loops
- 2 observation modules
- 2 context-management modules
- 5 tool modules
- 2 verification modules
- 6 solver helper tools

The release Composer additionally quarantines variants that are known to be
non-solver infrastructure or unsafe for solver selection. In this snapshot,
`editor_file_tools`, `passthrough`, and `combined_robust` are not offered to the
solver.

## Evaluate

From the release repository root, after installing the project and configuring
`.env`:

```bash
bash scripts/self_evolve.sh evaluate \
  "$PWD/generations/merged_active/gen_0/modules" \
  fix-git
```

Set `ENVIRONMENT=e2b` and `E2B_API_KEY` to use E2B. The default environment is
Docker. `llm_dynamic` is required to select evolved variants; `static` is a
useful control but intentionally returns the baseline bundle because this
snapshot has no `active_bundle.json` pin.

## Validation

The public copy passes the release fast-smoke battery: AST, isolated imports,
library load, bundle configuration, discovery contract, and static contract.
See `PROVENANCE.json` for source and sanitized tree hashes.
