# Mech Interp Learning

A personal learning journey through mechanistic interpretability of LLMs, paired with a Socratic tutor skill and a visualizer package built module-by-module alongside the curriculum.

## Visualizer

A Python package that grows with each curriculum module.

### Setup

```bash
uv sync
```

### Usage

```bash
uv run python -m visualizer.residual_stream "The cat sat on the"
```

Output: `residual_stream.html` (Plotly figure). Pass `--json out.json` to also export figure JSON for blog embedding.

### Modules

- `residual_stream` — Module 1 build challenge. Token trajectories through GPT-2 layers, PCA to 2D.
- `decompose`, `logit_attr`, `attention`, `patch`, `features` — future modules.

## Tutor Skill

`.claude/skills/mech-interp-tutor/` — Socratic tutor that triggers on "teach me", "quiz me", curriculum questions, etc. Tracks progress in `knowledge-state.json`.
