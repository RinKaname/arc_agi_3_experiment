# =====================================================================
# AmadeusZero – CNN-based action learning agent
# Source:
# Authors: Chakra (Lead), Gerry Weber (Adviser) — CUDA MUSUME
# GANBATTE RTX3060-CHAN!!! 就算 Kernel Panic，我的心对你也是 Thread-safe 的！
# =====================================================================
import atexit
import hashlib
import logging
import os
import random
import threading
import time
import traceback
from collections import deque
from datetime import datetime
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agents.agent import Agent

from arcengine import FrameData, GameAction, GameState


# --- Inlined from utils.py ---

def setup_experiment_directory(base_output_dir='runs'):
    """Create directories for outputs and logging."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join(base_output_dir, timestamp)
    os.makedirs(base_dir, exist_ok=True)
    log_file = os.path.join(base_dir, 'logs.log')
    print(f"Experiment directory created: {base_dir}")
    return base_dir, log_file


def get_environment_directory(base_dir, game_id):
    """Get or create environment-specific directory for a game_id."""
    env_dir = os.path.join(base_dir, game_id)
    os.makedirs(env_dir, exist_ok=True)
    return env_dir


def setup_logging_for_experiment(log_file_path):
    """Update logging configuration to use the experiment directory's log file."""
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            root_logger.removeHandler(handler)
            handler.close()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_file_path, mode="w")
    file_handler.setLevel(root_logger.level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)


# --- ActionModel ResNet ---

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class DynamicsNetwork(nn.Module):
    """The Imagination Engine: Predicts the next hidden state and immediate reward given a state and action."""
    def __init__(self, hidden_channels=128, num_action_planes=7):
        super().__init__()
        # 128 (hidden state) + 7 (action planes) = 135
        self.conv_in = nn.Conv2d(hidden_channels + num_action_planes, hidden_channels, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm2d(hidden_channels)

        self.res_blocks = nn.Sequential(
            ResidualBlock(hidden_channels),
            ResidualBlock(hidden_channels),
            ResidualBlock(hidden_channels)
        )

        # Reward head
        self.pool = nn.MaxPool2d(4, 4) # 64x64 -> 16x16
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_channels * 16 * 16, 256),
            nn.ReLU(),
            nn.Linear(256, 1) # Predicts scalar reward
        )

    def forward(self, hidden_state, action_planes):
        # Concatenate spatial hidden state and action planes
        x = torch.cat([hidden_state, action_planes], dim=1)

        # Process through ResNet to get next hidden state
        x = F.relu(self.bn_in(self.conv_in(x)))
        next_hidden_state = self.res_blocks(x)

        # Predict expected immediate reward
        pooled = self.pool(next_hidden_state)
        flattened = pooled.view(pooled.size(0), -1)
        reward = self.reward_head(flattened).squeeze(-1)

        return next_hidden_state, reward

def encode_action_to_spatial_planes(action_idx, grid_size=64, device='cpu'):
    """
    Converts a flat action index (0 to 5 for buttons, 5 + coord for clicks)
    into a 7-channel spatial representation [batch, 7, H, W] for the Dynamics network.
    """
    batch_size = action_idx.shape[0]
    # Initialize blank canvas: 7 channels of 64x64
    planes = torch.zeros(batch_size, 7, grid_size, grid_size, device=device)

    for i in range(batch_size):
        a = action_idx[i].item()
        if a < 5:
            # Action 1-5 (Global actions like MOVE UP). Fill the whole grid for that channel.
            planes[i, a, :, :] = 1.0
        else:
            # Action 6 (Coordinate click).
            coord = a - 5
            y = coord // grid_size
            x = coord % grid_size
            # Only put a '1' exactly at the clicked coordinate on the 7th plane (index 6)
            planes[i, 6, y, x] = 1.0

    return planes

class ActionModel(nn.Module):
    """CNN that predicts which actions will result in new frames with shared conv backbone."""

    def __init__(self, input_channels=16, grid_size=64):
        super().__init__()
        self.grid_size = grid_size
        self.num_action_types = 6  # ACTION1-ACTION5, plus ACTION6

        # 1. Representation Network (Initial Observation -> Hidden State)
        self.rep_conv_in = nn.Conv2d(input_channels + 2, 64, kernel_size=3, padding=1)
        self.rep_bn_in = nn.BatchNorm2d(64)
        self.rep_res_blocks = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64)
        )
        self.rep_conv_out = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # 2. Dynamics Network (Hidden State + Action -> Next Hidden State + Reward)
        self.dynamics_network = DynamicsNetwork(hidden_channels=128, num_action_planes=7)

        # 3. Prediction Network (Hidden State -> Policy + Value)
        self.pred_pool = nn.MaxPool2d(4, 4) # 64x64 -> 16x16
        self.pred_fc_proj = nn.Linear(128 * 16 * 16, 512)
        self.dropout = nn.Dropout(0.2)

        self.action_head = nn.Linear(512, self.num_action_types)

        self.value_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.coord_conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.coord_conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.coord_conv3 = nn.Conv2d(32, 1, kernel_size=3, padding=1)

        # Create the spatial meshgrid priors
        y_coords = torch.linspace(-1, 1, grid_size).view(-1, 1).repeat(1, grid_size)
        x_coords = torch.linspace(-1, 1, grid_size).view(1, -1).repeat(grid_size, 1)
        # Combine into [2, H, W]
        self.register_buffer('meshgrid', torch.stack([y_coords, x_coords], dim=0))

    def representation(self, x):
        """Initial inference step mapping observation to hidden state."""
        batch_size, seq_len, c, h, w = x.size()
        x = x.view(batch_size * seq_len, c, h, w)

        # Append spatial meshgrid to the input
        mesh = self.meshgrid.unsqueeze(0).repeat(batch_size * seq_len, 1, 1, 1)
        x = torch.cat([x, mesh], dim=1)

        x = F.relu(self.rep_bn_in(self.rep_conv_in(x)))
        res_features = self.rep_res_blocks(x)
        hidden_state = F.relu(self.rep_conv_out(res_features))
        return hidden_state, batch_size, seq_len

    def prediction(self, hidden_state, batch_size, seq_len):
        """Maps hidden state to policy (action probabilities) and value."""
        pooled = self.pred_pool(hidden_state)
        flattened = pooled.view(batch_size * seq_len, -1)
        common_features = F.relu(self.pred_fc_proj(flattened))
        common_features = self.dropout(common_features)

        action_logits = self.action_head(common_features)

        state_values = self.value_head(common_features)
        state_values = state_values.view(batch_size, seq_len)

        coord_features = F.relu(self.coord_conv1(hidden_state))
        coord_features = F.relu(self.coord_conv2(coord_features))
        coord_logits = self.coord_conv3(coord_features)
        coord_logits = coord_logits.view(batch_size * seq_len, -1)

        combined_logits = torch.cat([action_logits, coord_logits], dim=1)
        combined_logits = combined_logits.view(batch_size, seq_len, -1)

        return combined_logits, state_values

    def forward(self, x, hidden_state=None):
        """
        Standard forward pass for Behavioral Cloning and Actor-Critic RL.
        (MCTS will use .representation, .dynamics_network, and .prediction directly).
        """
        hidden_state, batch_size, seq_len = self.representation(x)
        combined_logits, state_values = self.prediction(hidden_state, batch_size, seq_len)
        return combined_logits, state_values, None


class MCTSNode:
    def __init__(self, prior: float):
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.children: dict[int, MCTSNode] = {}
        self.hidden_state = None  # Tensor of shape [1, 128, H, W]
        self.reward = 0.0         # Predicted immediate reward to reach this node

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class MCTS:
    def __init__(self, action_model: nn.Module, num_simulations: int = 100, c_puct: float = 1.5, discount: float = 0.99):
        self.action_model = action_model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.discount = discount

    def run(self, root_observation: torch.Tensor, available_actions: list[int] = None, tested_actions: set[int] = None) -> tuple[int, np.ndarray]:
        """
        Run MCTS from root observation.
        Returns:
            - Selected action index
            - Visit counts/probabilities distribution over the full 4101-action space
        """
        self.action_model.eval()
        device = next(self.action_model.parameters()).device
        grid_size = self.action_model.grid_size

        # 1. Representation step for root
        with torch.no_grad():
            # root_observation shape: [1, 1, C, H, W]
            hidden_state, batch_size, seq_len = self.action_model.representation(root_observation)
            # hidden_state shape: [1, 128, H, W]
            combined_logits, state_values = self.action_model.prediction(hidden_state, batch_size, seq_len)
            combined_logits = combined_logits.squeeze(0).squeeze(0)  # [4102]

        # 2. Get root policy distribution
        action_logits = combined_logits[:6]
        coord_logits = combined_logits[6:]

        # Filter available actions for root node
        action_mask = torch.zeros_like(action_logits)

        # 1. Official game valid actions mask
        if available_actions is not None and len(available_actions) > 0:
            action_mask = torch.full_like(action_logits, float('-inf'))
            for action in available_actions:
                action_id = action.value if hasattr(action, 'value') else int(action)
                if 1 <= action_id <= 6:
                    action_mask[action_id - 1] = 0.0

        # 2. Graph Explorer mask: penalize/block actions already tested in this exact state hash
        if tested_actions is not None and len(tested_actions) > 0:
            # We use a huge negative penalty rather than -inf so the agent *can* retry
            # if absolutely all actions are exhausted, but highly prefers untested frontiers.
            for a_idx in tested_actions:
                if a_idx < 5:
                    action_mask[a_idx] = float('-1e9')
                else:
                    # It's a coordinate action
                    if action_mask[5] != float('-inf'):
                        coord_idx = a_idx - 5
                        coord_logits[coord_idx] = float('-1e9')

        action_logits = action_logits + action_mask

        # Apply temperature scaling to root policy to keep decision-making smooth
        temperature = 1.5
        action_probs = F.softmax(action_logits / temperature, dim=0).cpu().numpy()
        coord_probs = F.softmax(coord_logits / temperature, dim=0).cpu().numpy()

        # Build full 4101-size policy prior for root (ACTION1-ACTION5 at 0-4, ACTION6 coordinates at 5-4100)
        root_policy = np.zeros(5 + 4096)
        root_policy[:5] = action_probs[:5]
        root_policy[5:] = action_probs[5] * coord_probs

        # Determine candidate actions to expand at root (before adding noise so we don't pick random junk)
        candidate_actions = self._sample_candidate_actions(action_probs, coord_probs, num_candidates=32)

        # Add Dirichlet noise to the root policy to ensure exploration (MuZero/AlphaZero standard)
        dirichlet_alpha = 0.3
        dirichlet_frac = 0.25
        # Only add noise to the candidate actions to keep it bounded
        noise = np.random.dirichlet([dirichlet_alpha] * len(candidate_actions))
        for i, a in enumerate(candidate_actions):
            root_policy[a] = root_policy[a] * (1 - dirichlet_frac) + noise[i] * dirichlet_frac

        # Re-normalize the candidate priors just in case
        prior_sum = sum(root_policy[a] for a in candidate_actions)
        if prior_sum > 0:
            for a in candidate_actions:
                root_policy[a] /= prior_sum

        # Initialize root node
        root = MCTSNode(prior=1.0)
        root.hidden_state = hidden_state
        
        # Expand root node: run dynamics to get hidden states and rewards for children
        with torch.no_grad():
            num_cand = len(candidate_actions)
            if num_cand > 0:
                cand_tensor = torch.tensor(candidate_actions, dtype=torch.long, device=device)
                action_planes = encode_action_to_spatial_planes(cand_tensor, grid_size=grid_size, device=device)
                
                # Expand root hidden state
                expanded_hidden = root.hidden_state.repeat(num_cand, 1, 1, 1)
                next_hidden, pred_rewards = self.action_model.dynamics_network(expanded_hidden, action_planes)
                
                for idx, a in enumerate(candidate_actions):
                    child = MCTSNode(prior=root_policy[a])
                    child.hidden_state = next_hidden[idx:idx+1]
                    child.reward = float(pred_rewards[idx].item())
                    root.children[a] = child

        # 3. MCTS Simulations Loop
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            actions_path = []

            # Selection
            while len(node.children) > 0:
                action, child = self._select_child(node)
                node = child
                search_path.append(node)
                actions_path.append(action)

            # Expansion & Evaluation of leaf node
            leaf_value = 0.0
            if node.hidden_state is not None:
                with torch.no_grad():
                    combined_logits, state_values = self.action_model.prediction(
                        node.hidden_state, batch_size=1, seq_len=1
                    )
                    combined_logits = combined_logits.squeeze(0).squeeze(0)
                    leaf_value = float(state_values.squeeze(0).squeeze(0).item())

                action_logits = combined_logits[:6]
                coord_logits = combined_logits[6:]

                action_probs = F.softmax(action_logits / temperature, dim=0).cpu().numpy()
                coord_probs = F.softmax(coord_logits / temperature, dim=0).cpu().numpy()

                leaf_policy = np.zeros(5 + 4096)
                leaf_policy[:5] = action_probs[:5]
                leaf_policy[5:] = action_probs[5] * coord_probs

                # Sample candidate actions for this leaf node
                leaf_candidates = self._sample_candidate_actions(action_probs, coord_probs, num_candidates=32)

                # Using Dynamics network to predict next hidden state and reward for each candidate action
                with torch.no_grad():
                    num_cand = len(leaf_candidates)
                    if num_cand > 0:
                        cand_tensor = torch.tensor(leaf_candidates, dtype=torch.long, device=device)
                        action_planes = encode_action_to_spatial_planes(cand_tensor, grid_size=grid_size, device=device)
                        
                        # Expand leaf hidden state to match batch size
                        expanded_hidden = node.hidden_state.repeat(num_cand, 1, 1, 1)
                        next_hidden, pred_rewards = self.action_model.dynamics_network(expanded_hidden, action_planes)
                        
                        for idx, a in enumerate(leaf_candidates):
                            child = MCTSNode(prior=leaf_policy[a])
                            child.hidden_state = next_hidden[idx:idx+1]
                            child.reward = float(pred_rewards[idx].item())
                            node.children[a] = child

            # Backpropagation
            self._backpropagate(search_path, actions_path, leaf_value)

        # 4. Choose action based on root visit counts
        visit_counts = np.zeros(5 + 4096)
        for a, child in root.children.items():
            visit_counts[a] = child.visit_count

        sum_visits = visit_counts.sum()
        if sum_visits == 0:
            selected_action = np.random.choice(candidate_actions)
            visit_probs = np.zeros(5 + 4096)
            visit_probs[candidate_actions] = 1.0 / len(candidate_actions)
        else:
            visit_probs = visit_counts / sum_visits
            selected_action = int(np.argmax(visit_probs))

        return selected_action, visit_probs

    def _sample_candidate_actions(self, action_probs: np.ndarray, coord_probs: np.ndarray, num_candidates: int = 8) -> list[int]:
        """Sample a small subset of candidate actions to keep branching factor bounded.
        Ensures coordinate actions (ACTION6) are always considered if they have non-zero probability.
        """
        candidates = []
        # Add top global actions
        for a in range(5):
            if action_probs[a] > 0.01:
                candidates.append(a)
        
        # Always try to sample coordinate actions if ACTION6 has any meaningful probability
        # Lowered the threshold slightly to ensure coordinates aren't ignored early in training
        if action_probs[5] > 0.005:
            # Ensure we sample at least 1 coordinate action, up to a max of 5
            num_coord_cand = max(1, min(num_candidates - len(candidates), 5))

            # Make sure we don't try to sample more than the available non-zero coords
            nonzero_coords = np.count_nonzero(coord_probs > 0)
            num_coord_cand = min(num_coord_cand, nonzero_coords)

            if num_coord_cand > 0:
                sampled_coords = np.random.choice(len(coord_probs), size=num_coord_cand, replace=False, p=coord_probs)
                for c in sampled_coords:
                    candidates.append(5 + c)
                    
        # Fallback if extremely uncertain
        if not candidates:
            candidates = list(range(5))

        return candidates

    def _select_child(self, node: MCTSNode) -> tuple[int, MCTSNode]:
        """Select child node maximizing UCB formula."""
        best_score = float('-inf')
        best_action = None
        best_child = None

        total_visit_count = sum(child.visit_count for child in node.children.values())

        for a, child in node.children.items():
            q_value = child.reward + self.discount * child.value
            u_value = self.c_puct * child.prior * (np.sqrt(total_visit_count) / (1 + child.visit_count))
            
            score = q_value + u_value
            if score > best_score:
                best_score = score
                best_action = a
                best_child = child

        return best_action, best_child

    def _backpropagate(self, search_path: list[MCTSNode], actions_path: list[int], leaf_value: float):
        """Propagate value and reward returns back up the tree."""
        g_value = leaf_value
        for i in reversed(range(len(search_path) - 1)):
            node = search_path[i]
            child = search_path[i + 1]
            g_value = child.reward + self.discount * g_value
            child.visit_count += 1
            child.value_sum += g_value


# --- Action Agent (AmadeusZero) ---

class AmadeusZero(Agent):
    """CNN-based action learning agent that predicts which actions cause frame changes."""

    MAX_ACTIONS = float('inf')
    _MAX_FRAMES = 10  # PERF: Keep only the last N frames (sliding window)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._save_lock = threading.Lock()
        seed = int(time.time() * 1000000) + hash(self.game_id) % 1000000
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        torch.manual_seed(seed % (2**32 - 1))
        self.start_time = time.time()

        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Action agent using device: {self.device}")

        # Setup experiment directory and logging
        self.base_dir, log_file = setup_experiment_directory()
        setup_logging_for_experiment(log_file)

        env_dir = get_environment_directory(self.base_dir, self.game_id)
        self.current_score = -1

        self.logger = logging.getLogger(f"ActionAgent_{self.game_id}")

        # Visualization disabled for submission
        self.save_action_visualizations = False

        # Initialize action model
        self.grid_size = 64
        self.num_coordinates = self.grid_size * self.grid_size
        self.num_colours = 16
        self.action_model = ActionModel(input_channels=self.num_colours, grid_size=self.grid_size).to(self.device)
        self.optimizer = optim.Adam(self.action_model.parameters(), lr=0.0001, weight_decay=1e-4) # Aligned with user snippet

        # LSTM hidden state is removed in Phase 1 ResNet transition
        self.hidden_state = None

        # Initialize MCTS
        self.mcts = MCTS(self.action_model, num_simulations=50, c_puct=1.5, discount=0.99)

        # Experience buffer (online RL experiences)
        self.experience_buffer = deque(maxlen=200000)
        self.visitation_counts = {}
        # Graph Explorer: map state_hash to a set of tested action indices
        self.tested_actions = {}
        # Expert Map: map state_hash to perfect human action_idx
        self.expert_map = {}
        self.batch_size = 64
        self.train_frequency = 5
        # Pinned human demonstration buffer — never evicted, always sampled
        self.human_demo_buffer: list[dict] = []

        # Track previous state/action
        self.prev_frame = None
        self.prev_action_idx = None

        # Action mapping
        self.action_list = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
                           GameAction.ACTION4, GameAction.ACTION5]

        self.log_dir = env_dir
        self.logger.info(f"Action agent initialized for game_id: {self.game_id}")

        # Checkpoint configuration
        self.checkpoint_dir = "checkpoints"
        self.checkpoint_path = os.path.join(self.checkpoint_dir, f"{self.game_id}_model_v2.pth")
        loaded = self._load_checkpoint()
        if loaded:
            self._pretrain_on_human_demonstration(bc_epochs=0)
        else:
            self._pretrain_on_human_demonstration(bc_epochs=35)
        atexit.register(self._save_checkpoint)

    def _save_checkpoint(self) -> None:
        """Save model and optimizer state dicts to checkpoint file."""
        if not self._save_lock.acquire(blocking=False):
            # Another thread is already saving, skip this redundant save
            return
        try:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            checkpoint = {
                'model_state_dict': self.action_model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'current_score': self.current_score,
                'action_counter': self.action_counter,
            }
            temp_path = self.checkpoint_path + ".tmp"
            # Remove any stale .tmp left by a previous crashed/killed process
            try:
                os.remove(temp_path)
            except FileNotFoundError:
                pass
            try:
                torch.save(checkpoint, temp_path)
                os.replace(temp_path, self.checkpoint_path)
                self.logger.info(f"Saved checkpoint to {self.checkpoint_path}")
                print(f"Saved checkpoint to {self.checkpoint_path}")
            except Exception as e:
                self.logger.error(f"Failed to save checkpoint: {e}")
                print(f"Failed to save checkpoint: {e}")
                try:
                    os.remove(temp_path)
                except FileNotFoundError:
                    pass
        finally:
            self._save_lock.release()

    def _load_checkpoint(self) -> bool:
        """Load model and optimizer state dicts from checkpoint file if it exists.

        Returns:
            bool: True if checkpoint was loaded successfully, False otherwise.
        """
        if not os.path.exists(self.checkpoint_path):
            self.logger.info(f"No checkpoint found at {self.checkpoint_path}, starting fresh.")
            print(f"No checkpoint found at {self.checkpoint_path}, starting fresh.")
            self.action_counter = 0
            return False
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            # Handle architecture mismatch gracefully
            result = self.action_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            if result.missing_keys or result.unexpected_keys:
                incompatible_path = self.checkpoint_path + '.incompatible'
                os.rename(self.checkpoint_path, incompatible_path)
                self.logger.warning(
                    f"Checkpoint architecture mismatch — renamed to {incompatible_path}. Starting fresh."
                    f" Missing: {result.missing_keys}. Unexpected: {result.unexpected_keys}"
                )
                print(f"Checkpoint architecture mismatch — starting fresh. Old checkpoint kept as .incompatible")
                self.action_counter = 0
                return False
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.current_score = checkpoint.get('current_score', -1)
            self.action_counter = checkpoint.get('action_counter', 0)
            self.logger.info(f"Loaded checkpoint from {self.checkpoint_path} (action_counter={self.action_counter})")
            print(f"Loaded checkpoint from {self.checkpoint_path} (action_counter={self.action_counter})")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            print(f"Failed to load checkpoint: {e}")
            self.action_counter = 0
            return False

    def _action_input_to_index(self, action_id: Any, data: dict[str, Any]) -> Optional[int]:
        if action_id is None:
            return None

        if isinstance(action_id, str):
            if action_id == "RESET":
                return None
            elif action_id.startswith("ACTION"):
                try:
                    action_val = int(action_id[6:])
                except ValueError:
                    return None
            else:
                return None
        else:
            action_val = int(action_id)
            if action_val == 0: # RESET
                return None

        if action_val < 6:
            return action_val - 1 # ACTION1-ACTION5 maps to index 0-4
        else:
            # ACTION6
            y = data.get("y", 0)
            x = data.get("x", 0)
            coord_idx = y * self.grid_size + x
            return 5 + coord_idx

    def _pretrain_on_human_demonstration(self, bc_epochs: int = 35) -> None:
        """Find and pre-train the model on matching human/player demonstration recordings.

        Uses two complementary techniques:
        1. Behavioral Cloning (BC): True cross-entropy loss on ground-truth action labels
           gives the model a strong warm start mimicking the human solution exactly.
        2. Experience Injection: Human transitions are stored in a separate pinned buffer
           (self.human_demo_buffer) that persists indefinitely and is blended into every
           training batch, providing a permanent anchor during online RL.
        """
        import json

        # 1. Locate all matching human/player recording files
        human_files = []
        recordings_dirs = ["recordings", os.path.join("recordings", "player")]
        game_prefix = self.game_id.split("-")[0]

        for r_dir in recordings_dirs:
            if os.path.exists(r_dir):
                for f in sorted(os.listdir(r_dir)):
                    file_path = os.path.join(r_dir, f)
                    if os.path.isdir(file_path):
                        continue

                    is_json_or_jsonl = f.endswith(".json") or f.endswith(".jsonl")
                    if not is_json_or_jsonl:
                        continue

                    # Filter: Player playthroughs in records, standard .human. recordings,
                    # or files without agent names (amadeuszero, myagent, myagent2, random)
                    is_agent_recording = any(agent_name in f for agent_name in [".amadeuszero.", ".myagent.", ".myagent2.", ".random."])

                    if f.startswith(self.game_id) or f.startswith(game_prefix):
                        # Always include files in recordings/player or standard .human. recordings
                        if "player" in r_dir or ".human." in f or not is_agent_recording:
                            human_files.append(file_path)

        human_files = sorted(list(set(human_files)))

        if not human_files:
            self.logger.info("No human demonstration files found; skipping pre-training.")
            print("No human demonstration files found; skipping pre-training.")
            return

        self.logger.info(f"Found {len(human_files)} demo files. Parsing...")
        print(f"Found {len(human_files)} human/player demos: {human_files}")

        # ------------------------------------------------------------------
        # 2. Parse all recordings into independent trajectories
        # ------------------------------------------------------------------
        trajectories: list[list[dict]] = []

        for human_file in human_files:
            transitions: list[dict] = []
            try:
                with open(human_file, "r", encoding="utf-8") as fh:
                    content = fh.read().strip()
                    if content.startswith("["):
                        lines = json.loads(content)
                    else:
                        lines = [json.loads(l) for l in content.splitlines() if l.strip()]

                prev_state_np = None
                prev_action_idx = None

                for line in lines:
                    data = line.get("data", {})
                    if not data:
                        continue
                    frame_layers = data.get("frame")
                    if not frame_layers:
                        continue

                    frame_arr = np.array(frame_layers, dtype=np.int64)[-1]  # last channel
                    tensor = torch.zeros(self.num_colours, self.grid_size, self.grid_size, dtype=torch.float32)
                    tensor.scatter_(0, torch.from_numpy(frame_arr).unsqueeze(0), 1)
                    state_np = tensor.numpy().astype(bool)

                    if prev_state_np is not None and prev_action_idx is not None:
                        transitions.append({
                            'state': prev_state_np,
                            'action_idx': prev_action_idx,
                            'reward': 1.0,  # placeholder; will be set below
                            'score': data.get('score', data.get('levels_completed', 0))
                        })
                        # Build the deterministic Expert Map
                        # The frame is stored as the last channel, exactly how we compute current_grid
                        if hasattr(self, 'last_seen_grid_np') and self.last_seen_grid_np is not None:
                            state_hash = hashlib.md5(self.last_seen_grid_np.tobytes()).hexdigest()
                            # Ensure we don't insert None into the map from RESETS or trajectory boundaries
                            if prev_action_idx is not None:
                                self.expert_map[state_hash] = prev_action_idx

                    # Save this grid for the NEXT iteration's map building
                    if data.get("frame"):
                         self.last_seen_grid_np = np.array(data.get("frame")[-1], dtype=np.int64)

                    action_input = data.get("action_input")
                    if action_input:
                        prev_action_idx = self._action_input_to_index(
                            action_input.get("id"), action_input.get("data", {})
                        )
                    else:
                        prev_action_idx = None

                    prev_state_np = state_np

                if not transitions:
                    continue

                # Apply discounted rewards per trajectory
                # Using 10.0 terminal reward and 0.99 gamma to match RL win logic
                gamma = 0.99
                running_reward = 10.0
                last_score = transitions[-1]['score'] if transitions else 0

                for i in reversed(range(len(transitions))):
                    current_score = transitions[i]['score']
                    if current_score != last_score:
                        # New level boundary moving backwards, reset running reward
                        running_reward = 10.0
                        last_score = current_score

                    # Store both step_reward (0.1 for moving) and the discounted terminal reward
                    # This ensures the human buffer looks identical to a winning RL buffer
                    transitions[i]['step_reward'] = 0.1
                    transitions[i]['reward'] = running_reward + 0.1
                    running_reward *= gamma

                trajectories.append(transitions)
                print(f"Parsed {len(transitions)} transitions from {human_file}.")

            except Exception as e:
                self.logger.error(f"Error parsing {human_file}: {e}")
                print(f"Error parsing {human_file}: {e}")
                traceback.print_exc()

        total_transitions = sum(len(t) for t in trajectories)
        if total_transitions == 0:
            print("No valid transitions found in any human demonstration.")
            return

        print(f"Total transitions parsed: {total_transitions} across {len(trajectories)} trajectories.")
        print(f"Expert Map populated with {len(self.expert_map)} unique state hashes.")
        self.logger.info(f"Expert Map populated with {len(self.expert_map)} unique state hashes.")

        # Cleanup temp variable
        if hasattr(self, 'last_seen_grid_np'):
            del self.last_seen_grid_np

        # ------------------------------------------------------------------
        # 3. Behavioral Cloning — truncated BPTT with sequential chunk ordering.
        # ------------------------------------------------------------------
        if bc_epochs > 0:
            chunk_size = 32  # Reduced to 32 to prevent overfitting and VRAM OOM on 6GB GPU
            print(f"Behavioral Cloning: {bc_epochs} epochs, chunk={chunk_size} (truncated BPTT)...")
            try:
                # Fresh optimizer — no stale momentum/variance from the RL checkpoint
                bc_optimizer = optim.Adam(self.action_model.parameters(), lr=0.0003, weight_decay=1e-4) # Aligned with user's snippet

                best_loss = float('inf')
                patience = 5
                epochs_without_improvement = 0

                # Pre-build CPU tensors once before training to avoid 10x overhead of converting/stacking on every epoch
                prebuilt_trajectories = []
                for transitions in trajectories:
                    if len(transitions) < 1:
                        continue
                    states_tensor = torch.stack([
                        torch.from_numpy(t['state']).float() for t in transitions
                    ])  # [N, C, H, W]
                    labels_tensor = torch.tensor(
                        [t['action_idx'] for t in transitions], dtype=torch.long
                    )  # [N]
                    prebuilt_trajectories.append((states_tensor, labels_tensor))

                for epoch in range(bc_epochs):
                    epoch_loss = 0.0
                    total_chunks = 0

                    # Shuffle trajectories to avoid ordering bias across epochs
                    random.shuffle(prebuilt_trajectories)

                    for all_states_cpu, all_labels_cpu in prebuilt_trajectories:
                        n = all_states_cpu.size(0)
                        if n < 1:
                            continue

                        starts = list(range(0, n, chunk_size))  # sequential order, NOT shuffled

                        for start in starts:
                            end = min(start + chunk_size, n)
                            chunk_states = all_states_cpu[start:end].to(self.device).unsqueeze(0)  # [1, L, C, H, W]
                            chunk_labels = all_labels_cpu[start:end].to(self.device)               # [L]

                            bc_optimizer.zero_grad()
                            logits, state_values, _ = self.action_model(chunk_states)  # [1, L, output_dim]
                            logits_sq = logits.squeeze(0)  # [L, 4102]

                            # Labels: if < 5, it's action 0-4. If >= 5, it's action 5 (coord action) and the rest is coord_idx
                            action_type_labels = torch.where(chunk_labels < 5, chunk_labels, torch.tensor(5, device=chunk_labels.device))

                            # 1. Action Type Loss
                            loss_action = F.cross_entropy(logits_sq[:, :6], action_type_labels)

                            # 2. Coordinate Loss (only applied if action 5 was the true label)
                            coord_mask = (chunk_labels >= 5)
                            loss_coord = torch.tensor(0.0, device=logits.device)
                            if coord_mask.any():
                                coord_labels = chunk_labels[coord_mask] - 5
                                coord_logits = logits_sq[coord_mask, 6:]
                                loss_coord = F.cross_entropy(coord_logits, coord_labels)

                            loss = loss_action + loss_coord
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.action_model.parameters(), max_norm=1.0)
                            bc_optimizer.step()
                            epoch_loss += loss.item()
                            total_chunks += 1

                    if (epoch + 1) % 1 == 0:
                        avg = epoch_loss / max(total_chunks, 1)
                        print(f"  BC epoch {epoch + 1}/{bc_epochs}  avg_loss={avg:.4f}", flush=True)

                        # Early stopping logic
                        if avg < best_loss:
                            best_loss = avg
                            epochs_without_improvement = 0
                        else:
                            epochs_without_improvement += 1

                        if epochs_without_improvement >= patience:
                            print(f"Early stopping triggered at epoch {epoch + 1}: Loss hasn't improved for {patience} epochs.")
                            break

                # Restore fresh RL optimizer — BC changed the weight landscape significantly
                self.optimizer = optim.Adam(self.action_model.parameters(), lr=0.0001, weight_decay=1e-4) # Aligned with user snippet
                print("Behavioral Cloning pre-training complete.")
                self.logger.info("Behavioral Cloning pre-training complete.")
            except Exception as e:
                self.logger.error(f"Error during BC pre-training: {e}")
                print(f"Error during BC pre-training: {e}")
                traceback.print_exc()

        # ------------------------------------------------------------------
        # 4. Experience Injection — pin human demos into a permanent buffer
        #    that is blended into every _train_action_model() call.
        # ------------------------------------------------------------------
        self.human_demo_buffer = []
        for transitions in trajectories:
            self.human_demo_buffer.extend(transitions)
        print(f"Experience injection: {len(self.human_demo_buffer)} human transitions pinned to human_demo_buffer.")
        self.logger.info(f"Pinned {len(self.human_demo_buffer)} human transitions into human_demo_buffer.")

    def cleanup(self, scorecard: Any = None) -> None:
        """Override to save checkpoint on exit/interrupt."""
        self.logger.info("Cleanup triggered. Saving checkpoint before exit...")
        print("Cleanup triggered. Saving checkpoint before exit...")
        try:
            self._save_checkpoint()
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint during cleanup: {e}")
            print(f"Failed to save checkpoint during cleanup: {e}")
        try:
            atexit.unregister(self._save_checkpoint)
        except Exception:
            pass
        super().cleanup(scorecard)

    def append_frame(self, frame: FrameData) -> None:
        """Override to cap frames list at _MAX_FRAMES (sliding window)."""
        self.frames.append(frame)
        if len(self.frames) > self._MAX_FRAMES:
            self.frames = self.frames[-self._MAX_FRAMES:]
        if frame.guid:
            self.guid = frame.guid
        if hasattr(self, "recorder") and not self.is_playback:
            import json
            self.recorder.record(json.loads(frame.model_dump_json()))

    def _get_score(self, frame):
        """Get score from frame, compatible with both patched and standard FrameData."""
        score = getattr(frame, 'score', None)
        return score if score is not None else frame.levels_completed

    def _sample_from_combined_output(self, combined_logits, available_actions=None):
        """Sample from combined 6 + 64x64 action space with masking for invalid actions."""
        action_logits = combined_logits[:6]
        coord_logits = combined_logits[6:]

        try:
            has_actions = available_actions is not None and len(available_actions) > 0
        except TypeError:
            has_actions = False

        if has_actions:
            action_mask = torch.full_like(action_logits, float('-inf'))
            for action in available_actions:
                # Gateway sends raw ints [1,2,...,6], not GameAction enums
                action_id = action.value if hasattr(action, 'value') else int(action)
                if 1 <= action_id <= 6:
                    action_mask[action_id - 1] = 0.0
            action_logits = action_logits + action_mask

        # Temperature scaling forces exploration even if the model is highly confident.
        # This prevents the "Spamming UP" 0.999 probability policy collapse.
        temperature = 1.5

        # Hierarchical sampling: First select the action type (0-5)
        action_probs = F.softmax(action_logits / temperature, dim=0)
        action_probs_np = action_probs.cpu().numpy()

        if np.isnan(action_probs_np).any():
            # Fallback
            action_probs_np = np.ones_like(action_probs_np) / 6.0

        action_idx = np.random.choice(6, p=action_probs_np)

        # Coordinate sampling
        coord_probs = F.softmax(coord_logits / temperature, dim=0)
        coord_probs_np = coord_probs.cpu().numpy()

        if np.isnan(coord_probs_np).any():
            coord_probs_np = np.ones_like(coord_probs_np) / len(coord_probs_np)

        if action_idx < 5:
            return action_idx, None, None, action_probs_np
        else:
            coord_idx = np.random.choice(len(coord_probs_np), p=coord_probs_np)
            y_idx = coord_idx // self.grid_size
            x_idx = coord_idx % self.grid_size
            return 5, (y_idx, x_idx), coord_idx, coord_probs_np

    def _frame_to_tensor(self, frame_data):
        """Convert frame data to tensor format for the model."""
        frame = np.array(frame_data.frame, dtype=np.int64)
        frame = frame[-1]
        assert frame.shape == (self.grid_size, self.grid_size), \
            f"Expected frame shape ({self.grid_size}, {self.grid_size}), got {frame.shape}"
        tensor = torch.zeros(self.num_colours, self.grid_size, self.grid_size, dtype=torch.float32)
        tensor.scatter_(0, torch.from_numpy(frame).unsqueeze(0), 1)
        return tensor.to(self.device)

    def _train_action_model(self, full_episode=False):
        """Train the action model on collected experiences using MuZero-style recurrent unrolling.

        Blends online RL experiences with pinned human demonstration transitions.
        Unrolls the dynamics network step-by-step to compute policy, value, reward, and consistency losses.
        """
        if full_episode:
            # Dynamically cap at MAX_ACTIONS (with a fallback of 200) to ensure full episode training
            max_seq = getattr(self, "MAX_ACTIONS", 200)
            seq_len = min(max_seq, len(self.experience_buffer))
            if seq_len < 2:
                return
            sequence = list(self.experience_buffer)[-seq_len:]
        else:
            seq_len = min(10, len(self.experience_buffer))
            if seq_len < 2:
                return
            # Take the most recent `seq_len` online experiences
            sequence = list(self.experience_buffer)[-seq_len:]

        # --- Blend in human demo experiences (up to 25% of the sequence) ---
        if self.human_demo_buffer:
            n_human = max(1, seq_len // 4)
            human_sample = random.sample(
                self.human_demo_buffer,
                min(n_human, len(self.human_demo_buffer))
            )
            sequence = human_sample + sequence  # prepend so gradients flow through them

        # Build tensors from the combined sequence
        states = torch.stack([torch.from_numpy(exp['state']).float().to(self.device) for exp in sequence])
        states = states.unsqueeze(0)  # [batch=1, seq_len, C, H, W]

        action_indices = torch.tensor([exp['action_idx'] for exp in sequence], dtype=torch.long, device=self.device)
        action_indices = action_indices.unsqueeze(0)  # [batch=1, seq_len]

        rewards = torch.tensor([exp['reward'] for exp in sequence], dtype=torch.float32, device=self.device)
        rewards = rewards.unsqueeze(0)  # [batch=1, seq_len]

        step_rewards = torch.tensor([exp.get('step_reward', 0.1) for exp in sequence], dtype=torch.float32, device=self.device)
        step_rewards = step_rewards.unsqueeze(0)  # [batch=1, seq_len]

        self.optimizer.zero_grad()

        # Recurrent unrolling trajectory for dynamics and prediction training
        total_loss = 0.0
        hidden_state, batch_size, _ = self.action_model.representation(states[:, 0:1])

        L = states.size(1)
        # MuZero standard unroll length to prevent compounding prediction errors and VRAM OOM
        unroll_steps = min(L, 5)

        # Truncated BPTT: Instead of unrolling the whole sequence at once, we train by taking random
        # chunks of length 'unroll_steps' from the sequence. This keeps VRAM flat while ensuring
        # the model still sees terminal rewards from anywhere in the episode.
        num_chunks = max(1, L // unroll_steps)

        for _ in range(num_chunks):
            # Pick a random starting point for the unroll chunk
            start_t = random.randint(0, max(0, L - unroll_steps))
            end_t = min(start_t + unroll_steps, L)
            chunk_length = end_t - start_t

            # Re-compute initial hidden state for this specific chunk
            hidden_state, batch_size, _ = self.action_model.representation(states[:, start_t:start_t+1])
            chunk_loss = 0.0

            for t in range(start_t, end_t):
                # 1. Prediction step: policy logits and state value
                combined_logits, state_values = self.action_model.prediction(hidden_state, batch_size=1, seq_len=1)
                combined_logits = combined_logits.squeeze(0)  # [1, 4102]
                state_values = state_values.squeeze(0)        # [1]

                # Policy Loss for step t
                action_type_target = torch.where(action_indices[:, t] < 5, action_indices[:, t], torch.tensor(5, device=self.device))
                loss_action = F.cross_entropy(combined_logits[:, :6], action_type_target)

                coord_mask = (action_indices[:, t] >= 5)
                loss_coord = torch.tensor(0.0, device=self.device)
                if coord_mask.any():
                    coord_target = action_indices[:, t][coord_mask] - 5
                    loss_coord = F.cross_entropy(combined_logits[:, 6:][coord_mask], coord_target)

                # Compute advantage and Policy Gradient loss
                advantage = rewards[:, t] - state_values.detach()
                policy_loss = (loss_action + loss_coord) * advantage.mean()

                # Value Loss
                value_loss = F.mse_loss(state_values, rewards[:, t])

                # Entropy regularization
                action_probs = F.softmax(combined_logits[:, :6], dim=-1)
                action_entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(-1).mean()
                coord_probs = F.softmax(combined_logits[:, 6:], dim=-1)
                coord_entropy = -(coord_probs * torch.log(coord_probs + 1e-8)).sum(-1).mean()

                chunk_loss += policy_loss + 0.5 * value_loss - 0.05 * action_entropy - 0.005 * coord_entropy

                # Dynamics and Consistency Step (only if not at final step)
                if t < end_t - 1:
                    # Target hidden state from representation of the actual next frame
                    with torch.no_grad():
                        target_hidden, _, _ = self.action_model.representation(states[:, t+1:t+2])

                    # Convert action taken to spatial planes
                    action_planes = encode_action_to_spatial_planes(
                        action_indices[:, t], grid_size=self.action_model.grid_size, device=self.device
                    )

                    # Dynamics step: predict next hidden state and immediate reward
                    next_hidden, pred_reward = self.action_model.dynamics_network(hidden_state, action_planes)

                    # Consistency Loss (match predicted hidden state with target representation)
                    consistency_loss = F.mse_loss(next_hidden, target_hidden.detach())

                    # Reward Loss
                    reward_loss = F.mse_loss(pred_reward, step_rewards[:, t])

                    chunk_loss += 0.5 * consistency_loss + 1.0 * reward_loss

                    # Recurrent unroll
                    hidden_state = next_hidden

            # Normalize chunk loss by chunk length
            if chunk_length > 0:
                chunk_loss = chunk_loss / chunk_length
                chunk_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.action_model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad() # Prepare for next chunk

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _has_time_elapsed(self):
        """Check if 8 hours have elapsed since start."""
        return (time.time() - self.start_time) >= 8 * 3600 - 5 * 60

    def is_done(self, frames, latest_frame):
        """Decide if the agent is done playing."""
        try:
            return any([
                latest_frame.state is GameState.WIN,
                self._has_time_elapsed(),
            ])
        except Exception as e:
            print(f"[DEBUG] is_done crashed: {e}")
            traceback.print_exc()
            return True  # bail out on error

    def choose_action(self, frames, latest_frame):
        """Choose action using action model predictions."""
        try:
            # DEBUG: Log frame info on first call
            if self.action_counter == 0:
                print(f"[DEBUG] latest_frame type: {type(latest_frame)}")
                print(f"[DEBUG] latest_frame.state: {latest_frame.state}")
                print(f"[DEBUG] latest_frame.levels_completed: {latest_frame.levels_completed}")
                print(f"[DEBUG] has score: {hasattr(latest_frame, 'score')}")
                print(f"[DEBUG] available_actions: {getattr(latest_frame, 'available_actions', 'N/A')}")
                if hasattr(latest_frame, 'frame') and latest_frame.frame:
                    frame_arr = np.array(latest_frame.frame)
                    print(f"[DEBUG] frame shape: {frame_arr.shape}")

            # Check if score/level has changed (triggers model reset for new level)
            current_level = self._get_score(latest_frame)
            if current_level > self.current_score:
                self.logger.info(f"Score changed from {self.current_score} to {current_level} at action {self.action_counter}")
                print(f"Score changed from {self.current_score} to {current_level} at action {self.action_counter}")

                # Episodic Credit Assignment: Reward the sequence that led to this win
                if len(self.experience_buffer) > 0:
                    gamma = 0.99
                    current_reward = 10.0 # High reward for winning level
                    # Iterate backwards and apply discounted rewards
                    for i in reversed(range(len(self.experience_buffer))):
                        self.experience_buffer[i]['reward'] += current_reward
                        current_reward *= gamma

                    # Train on the full winning episode
                    print(f"Training on winning episode of length {len(self.experience_buffer)}")
                    self._train_action_model(full_episode=True)

                self.experience_buffer.clear()

                self.visitation_counts.clear()
                self.tested_actions.clear() # Clear graph for new level
                print("Cleared experience buffer & graph - new level reached")

                # Note: We purposely do NOT reset network and optimizer here anymore,
                # allowing the agent to retain its learned weights across levels of the same game.
                print("Keeping action model and optimizer weights for new level")

                self.prev_frame = None
                self.prev_action_idx = None

                # Save checkpoint when level completed (excluding initial transition from -1)
                if self.current_score != -1:
                    self._save_checkpoint()
                self.current_score = current_level
            elif current_level < self.current_score:
                 # In case of reset to level 1 for some reason, don't reward
                 self.current_score = current_level
                 self.experience_buffer.clear()

                 self.visitation_counts.clear()
                 self.tested_actions.clear()

            if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
                self.prev_frame = None
                self.prev_action_idx = None
                self.experience_buffer.clear()
                self.visitation_counts.clear()
                self.tested_actions.clear()
                action = GameAction.RESET
                action.reasoning = "Game needs reset."
                return action

            # Convert current frame to tensor
            current_frame = self._frame_to_tensor(latest_frame)
            current_grid = np.array(latest_frame.frame, dtype=np.int64)[-1]
            current_state_hash = hashlib.md5(current_grid.tobytes()).hexdigest()

            # Create experience from previous action
            if self.prev_frame is not None:
                current_frame_np = current_frame.cpu().numpy().astype(bool)
                frame_changed = not np.array_equal(self.prev_frame, current_frame_np)

                # Update Graph Explorer: record the action we took from the previous state
                # Note: We use self.prev_state_hash which we save at the end of the step
                if hasattr(self, 'prev_state_hash') and self.prev_state_hash is not None:
                    if self.prev_state_hash not in self.tested_actions:
                        self.tested_actions[self.prev_state_hash] = set()
                    self.tested_actions[self.prev_state_hash].add(self.prev_action_idx)

                # Calculate curiosity/novelty search reward bonus
                self.visitation_counts[current_state_hash] = self.visitation_counts.get(current_state_hash, 0) + 1

                # Reward: constant time penalty to encourage solving quickly,
                # negative for stagnation (so BCE suppresses wasted actions).
                # We remove the flat +0.1 for frame_changed to prevent the agent from
                # exploiting interaction animations (like bumping a locked door).
                step_reward = -0.1
                reward = step_reward + (0.01 / np.sqrt(self.visitation_counts[current_state_hash]))

                experience = {
                    'state': self.prev_frame,
                    'action_idx': self.prev_action_idx,
                    'reward': reward,
                    'step_reward': step_reward
                }
                self.experience_buffer.append(experience)

            # Check the deterministic Expert Map first!
            # If the human has solved this exact state, bypass the neural network guessing entirely.
            if current_state_hash in self.expert_map and self.expert_map[current_state_hash] is not None:
                action_idx = self.expert_map[current_state_hash]

                if action_idx < 5:
                    selected_action = self.action_list[action_idx]
                    selected_action.reasoning = f"{selected_action.name} (Deterministic Expert Map Match)"
                else:
                    selected_action = GameAction.ACTION6
                    coord_idx = action_idx - 5
                    y = coord_idx // self.grid_size
                    x = coord_idx % self.grid_size
                    selected_action.set_data({"x": int(x), "y": int(y)})
                    selected_action.reasoning = f"ACTION6 at ({x}, {y}) (Deterministic Expert Map Match)"

                # Print debug info to see Expert Map triggering
                self.logger.info(f"[DEBUG] Chosen action: {selected_action.name}, reasoning: {selected_action.reasoning}")
                print(f"[DEBUG] Chosen action: {selected_action.name}, reasoning: {selected_action.reasoning}", flush=True)

            else:
                # Get action predictions using MCTS
                available_actions = getattr(latest_frame, 'available_actions', None)
                tested_in_current_state = self.tested_actions.get(current_state_hash, set())

                # Run MCTS from current frame
                # current_frame shape: [C, H, W] -> Add batch and seq_len: [1, 1, C, H, W]
                current_frame_seq = current_frame.unsqueeze(0).unsqueeze(0)

                action_idx, visit_probs = self.mcts.run(current_frame_seq, available_actions, tested_in_current_state)
                self.action_model.train()

                if action_idx < 5:
                    selected_action = self.action_list[action_idx]
                    selected_action.reasoning = f"{selected_action.name} (MCTS visits: {visit_probs[action_idx]:.3f})"
                else:
                    selected_action = GameAction.ACTION6
                    coord_idx = action_idx - 5
                    y = coord_idx // self.grid_size
                    x = coord_idx % self.grid_size
                    selected_action.set_data({"x": int(x), "y": int(y)})
                    selected_action.reasoning = f"ACTION6 at ({x}, {y}) (MCTS visits: {visit_probs[action_idx]:.3f})"

                # Print debug info to see MCTS visit distributions
                self.logger.info(f"[DEBUG] Chosen action: {selected_action.name}, reasoning: {selected_action.reasoning}, visit_probs (first 10): {visit_probs[:10].tolist()}")
                print(f"[DEBUG] Chosen action: {selected_action.name}, reasoning: {selected_action.reasoning}, visit_probs (first 10): {visit_probs[:10].tolist()}", flush=True)

            # Store current frame and action for next experience creation
            self.prev_frame = current_frame.cpu().numpy().astype(bool)
            self.prev_state_hash = current_state_hash
            if action_idx < 5:
                self.prev_action_idx = action_idx
            else:
                self.prev_action_idx = 5 + coord_idx

            # Train model periodically
            if self.action_counter % self.train_frequency == 0:
                self._train_action_model()

            # Save checkpoint periodically (every 1000 actions)
            if self.action_counter > 0 and self.action_counter % 1000 == 0:
                self._save_checkpoint()

            return selected_action

        except Exception as e:
            print(f"[DEBUG] choose_action CRASHED at action {self.action_counter}: {type(e).__name__}: {e}")
            traceback.print_exc()
            # Fallback: return a random action so the agent doesn't die
            action = random.choice(self.action_list[:5])
            action.reasoning = f"Fallback after error: {e}"
            return action
