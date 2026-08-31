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

[Project article: ModularRSI — Toward Generalizable Harness RSI](https://recursive-self-improvement.notion.site/blog-1-modularrsi-toward-generalizable-harness-rsi)

[Released Terminal-Bench evaluation trajectories](trajectories/mergefinal/README.md)

The evolution dataset is released separately as `ModularRSI_2000_Instances`.
It contains 1,000 terminal tasks and 1,000 software-engineering tasks, with 120
training tasks marked in each domain manifest.

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
export SUPPORT_DATASET_DIR=/path/to/ModularRSI_2000_Instances/tb
bash scripts/self_evolve.sh evolve tools

# Evaluate the included generation
bash scripts/self_evolve.sh evaluate
```

The included module library is under
[`generations/merged_active`](generations/merged_active/README.md). Evolution
runs are stored under `self_evo_runs/runs/<run-id>/`.

## License

Original ModularRSI research contributions are available under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/deed.en) for
non-commercial use only. See
[LICENSE-MODULARRSI.md](LICENSE-MODULARRSI.md).

Harbor-derived code remains under the Apache License 2.0 in [LICENSE](LICENSE).
Third-party components and datasets retain their respective licenses.


```tex
@misc{modularRSI_blog_2026,
  title     = {Exploration of Harness RSI: Methodology, Generalization, and Empirical Foundations},
  url       = {https://recursive-self-improvement.notion.site/blog-1-modularrsi-toward-generalizable-harness-rsi?source=copy_link},
  publisher = {Notion},
  author    = { Wu, Siwei and Ren, Jincheng and Li, Yizhi and Li, Haau-Sing and
    Yang, Chengran and Gu, Weicheng and Zhang, Yuxuan and Yang, Jian and Batista-Navarro, Riza and
    Zhang, Chuanyi and Zhou, Ming and Dai, Bryan and Lin, Chenghua
  },
  year      = {2026},
  month     = {Aug}
}
```
