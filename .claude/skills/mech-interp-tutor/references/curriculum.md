# Mechanistic Interpretability Curriculum

This is a living document. Modules can be reordered, split, merged, or added based on the student's needs and interests. The sequence below is a suggested progression, not a mandate.

## Foundation Modules (1-3)
These must come first — everything else builds on them.

---

### Module 0: Prerequisite Calibration (Assessment Only)

**Purpose:** Map existing knowledge before starting. Not a teaching module — a diagnostic conversation.

**Topics to probe:**
- Neural networks: What is a neuron, a layer, a weight? What does training do? Can they explain backprop intuitively?
- Linear algebra: Matrix multiplication, dot products, what "projecting" means. Eigenvalues/vectors if they've seen them.
- PyTorch: Tensor manipulation, indexing, `.shape`, autograd basics
- Probability: Softmax, log-probs, what a distribution over tokens means
- Transformers (surface level): What do they think a transformer does? What have they heard about attention?

**Neuroscience bridge:** Their neuroscience background likely gives them strong intuitions about:
- Networks of units that activate in response to input (neurons → artificial neurons)
- Receptive fields (→ attention patterns)
- Sparse coding (→ superposition)
- Lesion studies (→ activation patching/ablation)
- Population coding (→ distributed representations)
Map these bridges early — they'll be used throughout.

**Outcome:** A clear picture of where to start, what to skip, what to emphasize, and which neuroscience analogies will be most useful.

---

### Module 1: Transformer Internals — The Architecture

**Core concepts:**
- Token embeddings and positional encodings — how text becomes vectors
- The residual stream as a communication bus (neuroscience: like a cortical workspace that different processing areas read from and write to)
- Attention mechanism step by step:
  - Query, Key, Value matrices — what each does and why
  - Attention pattern computation (QK^T / √d_k → softmax)
  - How attention heads read information from other positions
  - Multi-head attention — why multiple heads, what they each do
  - How the output (OV circuit) writes back to the residual stream
- MLP layers: what they compute, why they're different from attention
- LayerNorm: what it does and where it appears
- The unembedding — how the residual stream becomes next-token probabilities

**Key intuitions to build:**
- The residual stream is the "main highway" — attention and MLPs are "on-ramps and off-ramps"
- Each attention head can be thought of as an independent information-routing unit
- The QK circuit decides "where to look," the OV circuit decides "what to move"
- MLPs are thought to store factual associations (still debated)

**Socratic questions:**
- "If I remove all the attention heads and just keep the MLPs, what breaks? What still works?"
- "An attention head at layer 0 can only look at the input embeddings. What kind of useful work can it do?"
- "Why do we need both attention AND MLPs? What can one do that the other can't?"

**Build challenge:**
Add to your visualizer package:
1. A function that loads GPT-2 small using TransformerLens
2. A function that runs a prompt and extracts residual stream states at each layer
3. A visualization (Plotly) of the residual stream at each layer (dimensionality-reduced via PCA to 2D)
4. Shows how the residual stream changes from input to output
5. CLI entry point: `python -m visualizer.residual_stream "The cat sat on the"`
6. Export the Plotly figure as JSON for blog embedding

**Tools:** TransformerLens, Plotly or matplotlib, PCA from sklearn

---

### Module 2: The Residual Stream and Composition

**Core concepts:**
- The residual stream as a high-dimensional vector that accumulates information
- Decomposing the residual stream: each component (attention head, MLP) adds its contribution
- Composition: how heads in later layers can read what earlier heads wrote
  - Q-composition, K-composition, V-composition
- The "virtual weights" perspective — what the full QK and OV matrices look like
- Why this matters: it means we can trace information flow through the network

**Key intuitions:**
- The residual stream at any position is a sum of all the contributions so far
- This is like "message passing" — components write messages, later components read them
- Neuroscience bridge: like how cortical areas both receive input from and send output to a shared workspace (global workspace theory)

**Socratic questions:**
- "If attention head 3.2 writes something to the residual stream at position 7, which later components could potentially read it?"
- "What does it mean for two heads to 'compose'? Can you think of a concrete example?"
- "If we could surgically replace the residual stream at position 5 after layer 3 with the residual stream from a different input, what would we learn?"

**Build challenge:**
Add to your visualizer package:
1. For a given prompt, decompose the final residual stream at a chosen position into per-component contributions (each attention head + each MLP)
2. Visualize these contributions as a stacked bar chart or heatmap (Plotly)
3. Show which components contribute most to the final prediction
4. CLI: `python -m visualizer.decompose "prompt" --position 5`

---

### Module 3: Logit Attribution

**Core concepts:**
- The logit lens: projecting intermediate residual streams through the unembedding to see "what the model is thinking" at each layer
- Direct logit attribution: decomposing the final logit for a specific token into contributions from each component
- Why this works: the unembedding is a linear map, so we can linearly decompose contributions
- Logit difference: comparing logits for two tokens (e.g., correct vs incorrect next token)

**Key intuitions:**
- You can ask "how much did attention head 5.3 push toward predicting 'cat' vs 'dog'?" — and get a number
- This gives you a first coarse map of "which components matter for this prediction"
- It's the starting point for almost all mech interp investigations

**Socratic questions:**
- "If the logit attribution for head 7.1 is very negative for the correct token, what does that mean? Is that head 'bad'?"
- "Why is logit *difference* often more useful than raw logit attribution?"
- "What are the limitations of this approach? What can't it tell you?"

**Build challenge:**
Add to your visualizer package:
1. For a given prompt and target token, compute direct logit attribution for every component
2. Display a ranked bar chart: which components push most toward/against the correct next token
3. Add the logit lens view: at each layer, what token has the highest probability
4. CLI: `python -m visualizer.logit_attr "prompt" --target "token"`

---

## Technique Modules (4-7)
These build on the foundation. Order is flexible within this group.

---

### Module 4: Attention Patterns and Information Movement

**Core concepts:**
- Reading and interpreting attention patterns
- Common head types: previous-token heads, induction heads, name-movers, backup heads
- Induction heads in detail: how they implement in-context learning
  - The two-head induction circuit: previous-token head + induction head
  - Why this is considered a foundational circuit
- Using attention patterns as evidence (and their limitations — attention ≠ importance)

**Key intuitions:**
- Attention patterns are like "routing tables" — they show where information flows
- But they don't tell you *what* information moves — that's the OV circuit
- Induction heads are like learned copy-paste mechanisms
- Neuroscience bridge: induction heads are similar to sequence completion in hippocampal replay

**Build challenge:**
Add to your visualizer package:
1. Interactive attention pattern viewer (for any layer/head, show the attention matrix as a Plotly heatmap)
2. An induction head detector: automatically identify which heads show the induction pattern on repeated sequences
3. Visualize the two-head induction circuit
4. CLI: `python -m visualizer.attention "prompt" --layer 5 --head 1`

---

### Module 5: Activation Patching and Causal Tracing

**Core concepts:**
- The core idea: replace one component's activation with what it would have been on a different input, and measure the effect
- This is a causal intervention, not just a correlation
- Activation patching vs. ablation (zero, mean) — different questions, different answers
- Path patching: more surgical, traces specific causal pathways
- Neuroscience bridge: this is literally a lesion study / stimulation experiment on a neural network

**Key intuitions:**
- Patching answers "does this component causally matter for this behavior?"
- Mean ablation asks "what happens if this component contributes nothing special?"
- The difference matters: a component can be important (patching changes output) without being informative (its activation is close to the mean anyway)
- Path patching lets you say "this component matters *because of its effect on this downstream component*"

**Build challenge:**
Add to your visualizer package:
1. Implement activation patching: for each component, patch its activation from a corrupted prompt and measure the change in logit difference
2. Display results as a heatmap (layers × components)
3. Identify the most causally important components for a specific behavior
4. CLI: `python -m visualizer.patch "clean prompt" "corrupted prompt" --target "token"`

---

### Module 6: Sparse Autoencoders and Features

**Core concepts:**
- The problem of superposition: networks represent more features than they have dimensions
- Why individual neurons are often uninterpretable (polysemanticity)
- Sparse autoencoders (SAEs) as a tool to decompose activations into interpretable features
- Dictionary learning: the mathematical framework
- Neuroscience bridge: this is directly analogous to sparse coding in visual cortex (Olshausen & Field)

**Key intuitions:**
- Superposition is an efficient compression trick — but it makes interpretation hard
- SAEs try to "undo" the compression by finding the real features
- A feature is a direction in activation space that corresponds to a human-interpretable concept
- This is cutting-edge and rapidly evolving — tools and methods are still being developed

**Build challenge:**
Add to your visualizer package:
1. Load a pre-trained SAE (from Anthropic's or the open-source ones)
2. Run text through the model and SAE, find which features activate
3. Visualize the top-activating features for a given input, with their descriptions
4. Show how features activate across a passage of text
5. CLI: `python -m visualizer.features "passage of text" --top 10`

---

### Module 7: Circuits — Putting It All Together

**Core concepts:**
- What is a circuit: a subgraph of the model that implements a specific behavior
- The full workflow: identify behavior → logit attribution → attention patterns → activation patching → trace the circuit
- Known circuits: IOI (indirect object identification), induction, greater-than
- Criteria for a good circuit explanation: faithfulness, completeness, minimality
- Automated circuit discovery (ACDC and related methods)

**Key intuitions:**
- A circuit is a human-interpretable story about how the model computes something
- Finding one requires combining all the techniques from previous modules
- It's detective work — form hypotheses, test them, revise
- Neuroscience bridge: this is like mapping a neural pathway for a specific behavior (like the fear circuit through the amygdala)

**Build challenge:**
The capstone project:
1. Pick a simple behavior in GPT-2 small (or replicate a known one)
2. Use all the tools built so far to trace the circuit responsible
3. Document the circuit with your visualizer: show the components, the information flow, and the causal evidence
4. Write a blog post with embedded interactive visualizations that someone else could follow
5. CLI: `python -m visualizer.circuit "prompt" --behavior "description"` (this one is aspirational — the real work is the investigation)

---

## Extension Modules (add as needed)

### Module E1: RLHF and Post-Training
When the student is curious about how training shapes model behavior beyond pretraining. Connects to their existing RL/DQN knowledge.

### Module E2: Developmental Interpretability
How circuits form during training. Grokking. Phase transitions. Connects to neuroscience of development.

### Module E3: Causal Abstraction
The formal framework for relating high-level algorithms to neural network computations. More mathematical, introduce when the student is ready.

### Module E4: Scaling and Practical Considerations
How mech interp techniques change when you go from GPT-2 small to production-scale models. What works, what breaks, what's unknown.

### Module E5: Reinforcement Learning from Human Feedback (RLHF) Internals
Deep dive into RLHF mechanics, reward modeling, and how post-training interventions change model internals. Bridges the student's RL knowledge to LLM training.

---

## Resources Reference

**Primary tools:**
- TransformerLens: `pip install transformer-lens`
- CircuitsVis: `pip install circuitsvis` (for attention pattern visualization in notebooks)
- Plotly: for custom interactive visualizations
- PyTorch: underlying tensor operations

**Key readings (introduce as relevant, don't front-load):**
- Neel Nanda's "A Comprehensive Mechanistic Interpretability Explainer"
- "A Mathematical Framework for Transformer Circuits" (Elhage et al.)
- "Toy Models of Superposition" (Elhage et al.)
- "In-context Learning and Induction Heads" (Olsson et al.)
- "Interpretability in the Wild" (Wang et al. — the IOI paper)
- "Scaling Monosemanticity" (Anthropic)
- ARENA Mechanistic Interpretability tutorials (Callum McDougall)

**The student's environment:**
- Python (primary language), structured as a package with reusable functions/classes
- CLI/IPython/scripts for exploration (not notebooks — the student prefers command-line workflows)
- A growing visualizer package that accumulates tools module by module
- Plotly for interactive visualizations (exported as JSON for blog embedding)
- Next.js blog where visualizations are embedded using `react-plotly.js` or CircuitsVis React components
- Claude Code for implementation help (but Claude should guide, not solve)

**Visualization architecture:**
- Python does the compute: load model, run analysis, extract tensors
- Outputs are Plotly figures (viewable locally via browser, exportable as JSON) and/or raw data JSON
- Blog consumes JSON artifacts via `react-plotly.js` or custom React components
- CircuitsVis components (attention patterns, colored tokens) can be used directly in React since they're React under the hood
- Each build challenge should produce both a CLI-runnable tool AND exportable artifacts for the blog
