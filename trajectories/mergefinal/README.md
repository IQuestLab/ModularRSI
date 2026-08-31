# ModularRSI evaluation trajectories

This directory contains three Terminal-Bench 2.0 evaluation repeats produced by
the merged modular harness.

| Run | Passed | Total | Accuracy |
|---|---:|---:|---:|
| `run-1` | 49 | 89 | 55.06% |
| `run-2` | 46 | 89 | 51.69% |
| `run-3` | 46 | 89 | 51.69% |

These are held-out evaluation artifacts. They are released for analysis and
reproducibility and are not used as evolution feedback.

Each run contains a summary result and 89 task directories. Each task keeps a
minimal configuration, score/token result, and the main ATIF trajectory. Raw
terminal recordings, per-episode requests, verifier logs, and intermediate
runtime files are excluded.

Model/provider identifiers, credentials, endpoints, host paths, personal
identifiers, exact timestamps, UUIDs, traceback paths, and cost fields have been
removed. Benchmark-internal sandbox paths such as `/app`, `/tmp`, and `/root`
are retained because they are part of the task interaction.

`manifest.json` records the byte length and SHA-256 digest of all 807 released
data files.
