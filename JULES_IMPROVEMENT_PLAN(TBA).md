# ARC-AGI-3 Agent Improvement Plan (TBA)

Based on the observation of ~10% GPU utilization during local training and the review of the current `AmadeusZero` model (`my_agent.py`), the following architectural improvements are proposed to increase model capacity and better utilize GPU resources.

## 1. Upgrade from LSTM to a Transformer/Self-Attention Backbone (Highly Recommended)
LSTMs must process sequences inherently step-by-step, which creates an execution bottleneck that starves modern highly-parallel GPUs (leading to the 10% usage).
**Suggestion**: Replace the LSTM with a small Transformer Encoder or a Multi-Head Attention mechanism. Transformers allow parallel processing of the entire sequence of frames in a batch during experience replay, which will massively spike your GPU utilization and likely improve the agent's ability to correlate past and future events.

## 2. Deepen the CNN with Residual Blocks (ResNet)
The current 4-layer CNN (32->64->128->256) is very shallow. With a 64x64 grid, it captures basic features but lacks the depth to build complex hierarchical representations of the environment (like long-range shape relationships).
**Suggestion**: Upgrade the CNN backbone to use Residual Blocks (e.g., a mini-ResNet18 structure). This allows you to increase the network depth from 4 layers to 10-18 layers without vanishing gradients. This increases the FLOPs (using more of the idle GPU) and will yield much richer spatial representations.

## 3. Increase Batch Size & Experience Replay Frequency
Currently, the `batch_size = 64` and `train_frequency = 5` (trains once every 5 steps). The current model is so small that a batch of 64 processes instantly, leaving the GPU idle while the environment simulates the next 5 steps.
**Suggestion**: Double or quadruple the batch size (e.g., 128 or 256) and train more frequently (e.g., every 1 or 2 steps). Combining this with Prioritized Experience Replay (PER) instead of random sampling will ensure the GPU spends more time doing high-value backpropagation.

## 4. Vectorized Environments (Parallelizing the Game)
The biggest bottleneck in offline RL is often waiting for the game simulator to step forward on the CPU.
**Suggestion**: If the ARC environment allows it, run multiple environment instances in parallel (e.g., 4-8 parallel games). The agent then predicts actions in a batched forward pass for all 8 environments simultaneously, which completely fills the GPU pipeline.

---
*Note: These changes are marked as TBA (To Be Announced / To Be Added) to prevent burning up "CUDA-chan" prematurely.*
