# =====================================================================
# StochasticGoose – CNN-based action learning agent
# Source: https://github.com/DriesSmit/ARC3-solution
# Authors: Dries Smit (Lead), Jack Cole (Adviser) — Tufa Labs
#
# Compatibility notes:
#   - Original uses `latest_frame.score` — replaced with `levels_completed`
#   - Original uses `latest_frame.available_actions` — gateway sends raw ints [1,2,...,6]
# =====================================================================
import hashlib
import logging
import os
import random
import time
import traceback
from collections import deque
from datetime import datetime
from typing import Any

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


# --- ActionModel CNN ---

class ActionModel(nn.Module):
    """CNN that predicts which actions will result in new frames with shared conv backbone."""

    def __init__(self, input_channels=16, grid_size=64):
        super().__init__()
        self.grid_size = grid_size
        self.num_action_types = 5  # ACTION1-ACTION5

        # Shared convolutional backbone (Added 2 channels for X/Y meshgrid)
        self.conv1 = nn.Conv2d(input_channels + 2, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        # LSTM Working Memory
        self.action_pool = nn.MaxPool2d(4, 4)
        self.lstm = nn.LSTM(input_size=256 * 16 * 16, hidden_size=512, batch_first=True)

        # Action head
        self.action_head = nn.Linear(512, self.num_action_types)

        # Coordinate head (64x64 action space)
        # Using a sequence of 3x3 convs to create a receptive field for the heatmap
        self.coord_conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.coord_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.coord_conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.coord_conv4 = nn.Conv2d(32, 1, kernel_size=3, padding=1)

        self.dropout = nn.Dropout(0.2)

        # Create the spatial meshgrid priors
        y_coords = torch.linspace(-1, 1, grid_size).view(-1, 1).repeat(1, grid_size)
        x_coords = torch.linspace(-1, 1, grid_size).view(1, -1).repeat(grid_size, 1)
        # Combine into [2, H, W]
        self.register_buffer('meshgrid', torch.stack([y_coords, x_coords], dim=0))

    def forward(self, x, hidden_state=None):
        # x is expected to be shape [batch, seq_len, channels, height, width]
        batch_size, seq_len, c, h, w = x.size()
        x = x.view(batch_size * seq_len, c, h, w)

        # Append spatial meshgrid to the input
        mesh = self.meshgrid.unsqueeze(0).repeat(batch_size * seq_len, 1, 1, 1)
        x = torch.cat([x, mesh], dim=1)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        conv_features = F.relu(self.conv4(x))

        # Flatten for LSTM. We need to pool first to match input_size
        pooled_features = self.action_pool(conv_features)
        flattened = pooled_features.view(batch_size, seq_len, -1)

        lstm_out, hidden_state = self.lstm(flattened, hidden_state)

        # Action head (using LSTM output)
        action_features = self.dropout(lstm_out.contiguous().view(batch_size * seq_len, -1))
        action_logits = self.action_head(action_features)

        # Coordinate head
        coord_features = F.relu(self.coord_conv1(conv_features))
        coord_features = F.relu(self.coord_conv2(coord_features))
        coord_features = F.relu(self.coord_conv3(coord_features))
        coord_logits = self.coord_conv4(coord_features)
        coord_logits = coord_logits.view(batch_size * seq_len, -1)

        combined_logits = torch.cat([action_logits, coord_logits], dim=1)
        # Reshape to [batch, seq_len, 5+4096]
        combined_logits = combined_logits.view(batch_size, seq_len, -1)

        return combined_logits, hidden_state


# --- Action Agent (StochasticGoose) ---

class MyAgent(Agent):
    """CNN-based action learning agent that predicts which actions cause frame changes."""

    MAX_ACTIONS = float('inf')
    _MAX_FRAMES = 10  # PERF: Keep only the last N frames (sliding window)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
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
        self.optimizer = optim.Adam(self.action_model.parameters(), lr=0.0001)
        self.hidden_state = None

        # Experience buffer
        self.experience_buffer = deque(maxlen=200000)
        self.experience_hashes = set()
        self.batch_size = 64
        self.train_frequency = 5

        # Track previous state/action
        self.prev_frame = None
        self.prev_action_idx = None

        # Action mapping
        self.action_list = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
                           GameAction.ACTION4, GameAction.ACTION5]

        self.log_dir = env_dir
        self.logger.info(f"Action agent initialized for game_id: {self.game_id}")

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
        return getattr(frame, 'score', None) or frame.levels_completed

    def _sample_from_combined_output(self, combined_logits, available_actions=None):
        """Sample from combined 5 + 64x64 action space with masking for invalid actions."""
        action_logits = combined_logits[:5]
        coord_logits = combined_logits[5:]

        try:
            has_actions = available_actions is not None and len(available_actions) > 0
        except TypeError:
            has_actions = False

        if has_actions:
            action_mask = torch.full_like(action_logits, float('-inf'))
            action6_available = False
            for action in available_actions:
                # Gateway sends raw ints [1,2,...,6], not GameAction enums
                action_id = action.value if hasattr(action, 'value') else int(action)
                if 1 <= action_id <= 5:
                    action_mask[action_id - 1] = 0.0
                elif action_id == 6:
                    action6_available = True
            action_logits = action_logits + action_mask
            if not action6_available:
                coord_logits = coord_logits + torch.full_like(coord_logits, float('-inf'))

        action_probs = torch.sigmoid(action_logits)
        coord_probs_raw = torch.sigmoid(coord_logits)
        coord_probs_scaled = coord_probs_raw / self.num_coordinates

        all_probs_sampling = torch.cat([action_probs, coord_probs_scaled])
        all_probs_sampling = all_probs_sampling / all_probs_sampling.sum()
        all_probs_sampling_np = all_probs_sampling.cpu().numpy()

        selected_idx = np.random.choice(len(all_probs_sampling_np), p=all_probs_sampling_np)

        coord_probs_viz = torch.sigmoid(coord_logits)
        all_probs_viz = torch.cat([action_probs, coord_probs_viz])
        all_probs_viz_np = all_probs_viz.cpu().numpy()

        if selected_idx < 5:
            return selected_idx, None, None, all_probs_viz_np
        else:
            coord_idx = selected_idx - 5
            y_idx = coord_idx // self.grid_size
            x_idx = coord_idx % self.grid_size
            return 5, (y_idx, x_idx), coord_idx, all_probs_viz_np

    def _frame_to_tensor(self, frame_data):
        """Convert frame data to tensor format for the model."""
        frame = np.array(frame_data.frame, dtype=np.int64)
        frame = frame[-1]
        assert frame.shape == (self.grid_size, self.grid_size), \
            f"Expected frame shape ({self.grid_size}, {self.grid_size}), got {frame.shape}"
        tensor = torch.zeros(self.num_colours, self.grid_size, self.grid_size, dtype=torch.float32)
        tensor.scatter_(0, torch.from_numpy(frame).unsqueeze(0), 1)
        return tensor.to(self.device)

    def _compute_experience_hash(self, frame, action_idx):
        """Compute hash for frame+action combination to ensure uniqueness."""
        frame_bytes = frame.tobytes()
        hash_input = frame_bytes + str(action_idx).encode('utf-8')
        return hashlib.md5(hash_input).hexdigest()

    def _train_action_model(self, full_episode=False):
        """Train the action model on collected experiences."""
        if full_episode:
            seq_len = len(self.experience_buffer)
            if seq_len < 2:
                return
            sequence = list(self.experience_buffer)
        else:
            seq_len = min(10, len(self.experience_buffer))
            if seq_len < 2:
                return
            # We take the most recent `seq_len` experiences
            sequence = list(self.experience_buffer)[-seq_len:]

        # Add seq_len dimension
        states = torch.stack([torch.from_numpy(exp['state']).float().to(self.device) for exp in sequence])
        states = states.unsqueeze(0) # [batch=1, seq_len, C, H, W]

        action_indices = torch.tensor([exp['action_idx'] for exp in sequence], dtype=torch.long, device=self.device)
        action_indices = action_indices.unsqueeze(0) # [batch=1, seq_len]

        rewards = torch.tensor([exp['reward'] for exp in sequence], dtype=torch.float32, device=self.device)
        rewards = rewards.unsqueeze(0) # [batch=1, seq_len]

        self.optimizer.zero_grad()

        # combined_logits shape: [batch=1, seq_len, output_dim]
        combined_logits, _ = self.action_model(states)

        # gather across the output_dim (dim=2)
        selected_logits = combined_logits.gather(2, action_indices.unsqueeze(2)).squeeze(2)

        main_loss = F.binary_cross_entropy_with_logits(selected_logits, rewards)

        all_probs = torch.sigmoid(combined_logits)
        action_entropy = all_probs[:, :, :5].mean()
        coord_entropy = all_probs[:, :, 5:].mean()

        total_loss = main_loss - 0.0001 * action_entropy - 0.00001 * coord_entropy
        total_loss.backward()
        self.optimizer.step()

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
                self.experience_hashes.clear()
                print("Cleared experience buffer - new level reached")

                # Note: We purposely do NOT reset network and optimizer here anymore,
                # allowing the agent to retain its learned weights across levels of the same game.
                print("Keeping action model and optimizer weights for new level")
                self.hidden_state = None

                self.prev_frame = None
                self.prev_action_idx = None
                self.current_score = current_level
            elif current_level < self.current_score:
                 # In case of reset to level 1 for some reason, don't reward
                 self.current_score = current_level
                 self.experience_buffer.clear()
                 self.experience_hashes.clear()

            if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
                self.prev_frame = None
                self.prev_action_idx = None
                action = GameAction.RESET
                action.reasoning = "Game needs reset."
                return action

            # Convert current frame to tensor
            current_frame = self._frame_to_tensor(latest_frame)

            if current_frame is None:
                self.prev_frame = None
                self.prev_action_idx = None
                action = random.choice(self.action_list[:5])
                action.reasoning = f"Skipped weird frame, random {action.value}"
                return action

            # Create experience from previous action
            if self.prev_frame is not None:
                experience_hash = self._compute_experience_hash(self.prev_frame, self.prev_action_idx)
                if experience_hash not in self.experience_hashes:
                    current_frame_np = current_frame.cpu().numpy().astype(bool)
                    frame_changed = not np.array_equal(self.prev_frame, current_frame_np)

                    # We penalize each step slightly to encourage speed, and penalize stagnant frames
                    step_reward = -0.01 if frame_changed else -0.1

                    experience = {
                        'state': self.prev_frame,
                        'action_idx': self.prev_action_idx,
                        'reward': step_reward
                    }
                    self.experience_buffer.append(experience)
                    self.experience_hashes.add(experience_hash)

            # Get action predictions
            available_actions = getattr(latest_frame, 'available_actions', None)
            with torch.no_grad():
                # Add batch and seq_len dimensions: [1, 1, C, H, W]
                current_frame_seq = current_frame.unsqueeze(0).unsqueeze(0)
                combined_logits, self.hidden_state = self.action_model(current_frame_seq, self.hidden_state)
                # Remove batch and seq_len dims for sampling: [1, 1, output_dim] -> [output_dim]
                combined_logits = combined_logits.squeeze(0).squeeze(0)
                action_idx, coords, coord_idx, all_probs = self._sample_from_combined_output(
                    combined_logits, available_actions
                )

                if action_idx < 5:
                    selected_action = self.action_list[action_idx]
                    selected_action.reasoning = f"{selected_action.name} (prob: {all_probs[action_idx]:.3f})"
                else:
                    selected_action = GameAction.ACTION6
                    y, x = coords
                    selected_action.set_data({"x": int(x), "y": int(y)})
                    selected_action.reasoning = f"ACTION6 at ({x}, {y}) (prob: {all_probs[coord_idx]:.3f})"

            # Store current frame and action for next experience creation
            self.prev_frame = current_frame.cpu().numpy().astype(bool)
            if action_idx < 5:
                self.prev_action_idx = action_idx
            else:
                self.prev_action_idx = 5 + coord_idx

            # Train model periodically
            if self.action_counter % self.train_frequency == 0:
                self._train_action_model()

            return selected_action

        except Exception as e:
            print(f"[DEBUG] choose_action CRASHED at action {self.action_counter}: {type(e).__name__}: {e}")
            traceback.print_exc()
            # Fallback: return a random action so the agent doesn't die
            action = random.choice(self.action_list[:5])
            action.reasoning = f"Fallback after error: {e}"
            return action
