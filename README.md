# ModularRSI

**ModularRSI studies generalizable Harness RSI: can an agent improve its
harness from independent execution experience and transfer the improvement to
unseen tasks, domains, and foundation models?**

Modern agent capability depends not only on the foundation model, but also on
the harness that controls interaction, observations, context, tools, and task
completion. ModularRSI decomposes this harness into five evolvable modules:

```text
agent_loop · observation · tools · context_mgmt · verification
```

It learns from a benchmark-independent evolution pool, contrasts successful and
failed trajectories, applies scoped module changes, and integrates only
candidates that pass validation gates. The study improves Terminal-Bench 2.0
accuracy from **47.57% to 52.43%** and evaluates transfer across unseen tasks,
domains, and models.

[Project article: ModularRSI — Toward Generalizable Harness RSI](https://app.notion.com/p/yizhilll/Blog-1-ModularRSI-Toward-Generalizable-Harness-RSI-3b1f36408b8c80bbadc7d7a41f6b0095)

[Released Terminal-Bench evaluation trajectories](trajectories/mergefinal/README.md)

The [released evolution data](evolution_data/README.md) contains 1,000 terminal
tasks and 1,000 software-engineering tasks, with a runnable 120-task training
view for each domain.

## How it works

```text
independent experience
        ↓
contrastive trajectory diagnosis
        ↓
module-level proposal and implementation
        ↓
review and runtime validation
        ↓
new immutable generation
        ↓
held-out evaluation
```

## Run

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
cp .env.example .env
```

```bash
# Evolve one module type
export SUPPORT_DATASET_DIR="$PWD/evolution_data/tb"
bash scripts/self_evolve.sh evolve tools

# Evaluate the included generation
bash scripts/self_evolve.sh evaluate
```

The included module library is under
[`generations/merged_active`](generations/merged_active/README.md). Evolution
runs are stored under `self_evo_runs/runs/<run-id>/`.

## License

Built on [Harbor](https://github.com/harbor-framework/harbor) and distributed
under its license. See [LICENSE](LICENSE).
