"""Residual stream visualizer.

Build challenge for Module 1. Loads GPT-2 small, extracts the residual stream
at every layer for a given prompt, projects to 2D via PCA, and renders a
Plotly figure showing how each token's representation moves through layers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_model(model_name: str = "gpt2"):
    """Load a HookedTransformer for the given model name.

    Returns the model. Should run on CPU by default; GPU optional.
    """
    raise NotImplementedError


def extract_residuals(model, prompt: str):
    """Run the prompt through the model and collect the residual stream
    at every layer.

    Returns:
        residuals: tensor of shape (n_layers + 1, n_tokens, d_model)
                   the +1 accounts for the embedding layer (layer 0)
        tokens:    list of decoded token strings, length n_tokens
    """
    raise NotImplementedError


def reduce_to_2d(residuals):
    """PCA-project residuals from d_model down to 2 dimensions.

    Input:  (n_layers, n_tokens, d_model)
    Output: (n_layers, n_tokens, 2)

    Decision to make: fit PCA on all (layer, token) points pooled together,
    or fit a separate PCA per layer? Pooled gives a consistent coordinate
    system across layers (better for visualizing trajectories).
    """
    raise NotImplementedError


def build_figure(coords_2d, tokens):
    """Build a Plotly figure showing each token's trajectory through layers.

    Input:
        coords_2d: (n_layers, n_tokens, 2)
        tokens:    list of token strings

    Output: plotly.graph_objects.Figure
    """
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", help="Text prompt to visualize.")
    parser.add_argument("--model", default="gpt2", help="HF model name (default: gpt2).")
    parser.add_argument("--out", type=Path, default=Path("residual_stream.html"),
                        help="Output HTML file for the Plotly figure.")
    parser.add_argument("--json", type=Path, default=None,
                        help="Optional path to dump figure JSON for blog embedding.")
    args = parser.parse_args()

    model = load_model(args.model)
    residuals, tokens = extract_residuals(model, args.prompt)
    coords_2d = reduce_to_2d(residuals)
    fig = build_figure(coords_2d, tokens)

    fig.write_html(args.out)
    print(f"Wrote {args.out}")

    if args.json is not None:
        args.json.write_text(json.dumps(fig.to_dict(), default=str))
        print(f"Wrote {args.json}")


if __name__ == "__main__":
    main()
