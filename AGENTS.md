# ARC-AGI-3 AmadeusZero Agent - Project Knowledge & Roadmap

## 🧠 Core Philosophy
This project uses a reinforcement learning (RL) approach combined with Behavioral Cloning (BC) to solve ARC-AGI tasks through a sequential action space (clicking coordinates and tools), rather than using LLMs or pure program synthesis.

## 🏗️ Architecture History

### AmadeusZero V1 (amadeuszero.py)
* **Design:** 4-layer CNN (shared backbone) -> MaxPool -> LSTM -> Action Head (5 discrete) & Spatial Head (64x64 grid).
* **Flaws Identified & Preserved:**
  * **Memory/Speed Bottleneck:** The CNN output flattened to 65,536 features, causing the LSTM to have ~135 Million parameters. This caused extreme memory usage and slow training.
  * **Reward Math Bug:** Behavioral Cloning applied a discount factor (`gamma=0.997`) continuously across the entire human demonstration (e.g., 340 steps). This degraded target rewards for early levels down to ~0.36, causing the model to underfit human data. **(Fixed in V1)**.
  * **Probability Distortion:** Coordinate probabilities were divided by 4096 and flattened with action probabilities, artificially suppressing coordinate selection to ~16%.
* **Status:** Kept intact to preserve the existing `.pth` checkpoint compatibility.

### AmadeusZero V2 (amadeuszero_v2.py)
* **Optimized LSTM:** Inserted a 1x1 Convolution (`nn.Conv2d(256, 32, kernel_size=1)`) and increased MaxPool to 8x8. This reduced the LSTM input to 2,048 features, dropping the parameter count to ~5M. Training is exponentially faster.
* **Hierarchical Action Head:** Predicts a categorical type (1-6). If type 6 is chosen, it subsequently samples from a spatial coordinate heatmap. This fixes the mathematical probability distortion.
* **Loss Function:** Replaced isolated Binary Cross Entropy (BCE) with categorical Cross-Entropy. This provides a negative gradient to unselected actions, stabilizing RL drift.

## 🚀 The Path Forward: Why MuZero and Graph Exploration?
The user correctly identified that a pure "Reactive" CNN+LSTM cannot reason about unseen ARC puzzles. It can perfectly memorize human demonstrations (BC), but cannot perform deductive reasoning.

**The MuZero Transition:**
MuZero (which mastered Atari and Go) is highly relevant because it combines **Deep RL with Monte Carlo Tree Search (MCTS)**.
For AmadeusZero to truly solve unseen ARC tasks, it must evolve into a MuZero-style architecture:
1. **Representation Network:** Convert the ARC grid into a latent state.
2. **Dynamics Network:** Given a latent state and an action, predict the *next* latent state and the reward (i.e., simulate clicks in its head without interacting with the real environment).
3. **Prediction Network:** Evaluate the value of the latent state and the prior probabilities of actions.
4. **MCTS:** Use the above networks to plan out a sequence of clicks (imagination) before actually executing a move on the real ARC grid.

### 🗺️ The "Graph Explorer" Paradigm
Based on recent ARC-AGI-3 findings ([Graph-Based Exploration for ARC-AGI-3 Interactive Reasoning Tasks - arXiv:2512.24156v1](https://arxiv.org/html/2512.24156v1)), frontier LLMs and pure Deep RL struggle massively with the 96,000 step limits and sparse rewards.
The state-of-the-art solution involves:
1. Treating the game as a **deterministic graph of states**.
2. **Hashing** every visual state visited.
3. Tracking which actions have been attempted from each state hash.
4. **Forcing exploration of new frontiers** by actively masking/blocking MCTS or Random choices from selecting actions that have already been tested in the current state.

Our roadmap includes integrating a rigorous `tested_actions` Graph Explorer layer directly into AmadeusZero's `choose_action` loop to prevent RL stagnation and infinite animation-farming loops.

## 🛑 Agent Instructions / Rules
1. **Never alter `amadeuszero.py` structurally.** It must remain compatible with existing `_model.pth` checkpoints. All major structural enhancements belong in `v2` or subsequent versions.
2. **Reward Math:** Ensure episodic reward discounting resets *per level* (`score` change), not continuously across entire multi-level JSONL recordings.
3. **Action Space:** Always treat discrete UI actions (1-5) and spatial coordinates (6) as a hierarchical or properly masked sequence, never as a flat probability distribution divided by grid size.
