# Mech Interp Learning

A project for learning mechanistic interpretability of LLMs through Socratic tutoring and hands-on building.

## Skills

- **mech-interp-tutor** — Socratic tutor for mech interp. Triggers on "teach me", "quiz me", "let's study", "tutor mode", "what should I learn next", "assess my knowledge", "mech interp", or any curriculum/progress discussion.

## Structure

```
mech-interp-learning/
├── .claude/
│   └── skills/
│       └── mech-interp-tutor/     # Socratic tutor skill
│           ├── SKILL.md
│           └── references/
│               ├── curriculum.md
│               └── knowledge-state-template.json
├── knowledge-state.json           # Learner progress tracking (persistent)
├── visualizer/                    # Python package built module-by-module
└── CLAUDE.md
```

## Knowledge Tracking

`knowledge-state.json` tracks module progress, strengths, gaps, and session logs. Updated after each substantive interaction.

## Visualizer Package

A Python package that grows with each curriculum module. Outputs Plotly figures (viewable locally, exportable as JSON for blog embedding).

### Tools

- TransformerLens for model access
- Plotly for interactive visualizations
- CircuitsVis for attention pattern visualization
- PyTorch for tensor operations

### Commands

```bash
uv run python -m visualizer.residual_stream "The cat sat on the"
uv run python -m visualizer.decompose "prompt" --position 5
uv run python -m visualizer.logit_attr "prompt" --target "token"
uv run python -m visualizer.attention "prompt" --layer 5 --head 1
uv run python -m visualizer.patch "clean prompt" "corrupted prompt" --target "token"
uv run python -m visualizer.features "passage of text" --top 10
```
