# =====================================================================
# AmadeusZero – CNN-based action learning agent
# Source:
# Authors: Chakra (Lead), Gerry Weber (Adviser) — CUDA MUSUME
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


# --- ActionModel CNN ---

class ActionModel(nn.Module):
    """CNN that predicts which actions will result in new frames with shared conv backbone."""

    def __init__(self, input_channels=16, grid_size=64):
        super().__init__()
        self.grid_size = grid_size
        self.num_action_types = 6  # ACTION1-ACTION5, plus ACTION6

        # Shared convolutional backbone (Added 2 channels for X/Y meshgrid)
        self.conv1 = nn.Conv2d(input_channels + 2, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        # LSTM Working Memory
        self.proj = nn.Conv2d(256, 32, kernel_size=1)
        self.action_pool = nn.MaxPool2d(4, 4)
        self.lstm = nn.LSTM(input_size=32 * 16 * 16, hidden_size=512, batch_first=True)

        # Action head
        self.action_head = nn.Linear(512, self.num_action_types)
        self.lstm_to_spatial = nn.Linear(512, 32)

        # Coordinate head (64x64 action space)
        # Using a sequence of 3x3 convs to create a receptive field for the heatmap
        self.coord_conv1 = nn.Conv2d(256 + 32, 128, kernel_size=3, padding=1)
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
        # Project LSTM features back into spatial dimensions [batch * seq_len, 32, h, w]
        mem_features = F.relu(self.lstm_to_spatial(lstm_out.contiguous().view(batch_size * seq_len, -1)))
        mem_spatial = mem_features.view(batch_size * seq_len, 32, 1, 1).expand(-1, -1, h, w)
        coord_input = torch.cat([conv_features, mem_spatial], dim=1)

        coord_features = F.relu(self.coord_conv1(coord_input))
        coord_features = F.relu(self.coord_conv2(coord_features))
        coord_features = F.relu(self.coord_conv3(coord_features))
        coord_logits = self.coord_conv4(coord_features)
        coord_logits = coord_logits.view(batch_size * seq_len, -1)

        combined_logits = torch.cat([action_logits, coord_logits], dim=1)
        # Reshape to [batch, seq_len, 5+4096]
        combined_logits = combined_logits.view(batch_size, seq_len, -1)

        return combined_logits, hidden_state


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
        self.optimizer = optim.Adam(self.action_model.parameters(), lr=0.0001)
        self.hidden_state = None

        # Experience buffer (online RL experiences)
        self.experience_buffer = deque(maxlen=200000)
        self.visitation_counts = {}
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
        self._load_checkpoint()
        self._pretrain_on_human_demonstration()
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

    def _load_checkpoint(self) -> None:
        """Load model and optimizer state dicts from checkpoint file if it exists."""
        if not os.path.exists(self.checkpoint_path):
            self.logger.info(f"No checkpoint found at {self.checkpoint_path}, starting fresh.")
            print(f"No checkpoint found at {self.checkpoint_path}, starting fresh.")
            self.action_counter = 0
            return
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
                return
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.current_score = checkpoint.get('current_score', -1)
            self.action_counter = checkpoint.get('action_counter', 0)
            self.logger.info(f"Loaded checkpoint from {self.checkpoint_path} (action_counter={self.action_counter})")
            print(f"Loaded checkpoint from {self.checkpoint_path} (action_counter={self.action_counter})")
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            print(f"Failed to load checkpoint: {e}")
            self.action_counter = 0

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

    def _pretrain_on_human_demonstration(self) -> None:
        """Find and pre-train the model on any matching human demonstration recording.

        Uses two complementary techniques:
        1. Behavioral Cloning (BC): True cross-entropy loss on ground-truth action labels
           gives the model a strong warm start mimicking the human solution exactly.
        2. Experience Injection: Human transitions are stored in a separate pinned buffer
           (self.human_demo_buffer) that persists indefinitely and is blended into every
           training batch, providing a permanent anchor during online RL.
        """
        import json

        # Locate the human recording file
        human_file = None
        recordings_dir = "recordings"
        if os.path.exists(recordings_dir):
            for f in sorted(os.listdir(recordings_dir)):
                # Primary: standard .human. naming convention
                if f.startswith(self.game_id) and ".human." in f and f.endswith(".recording.jsonl"):
                    human_file = os.path.join(recordings_dir, f)
                    break
        # Fallback: bare game-prefix json (e.g. ls20-777616c6-....json)
        if not human_file and os.path.exists(recordings_dir):
            game_prefix = self.game_id.split("-")[0]
            for f in sorted(os.listdir(recordings_dir)):
                if f.startswith(game_prefix) and f.endswith(".json") and "." not in f.split("-", 1)[1].split(".")[0]:
                    human_file = os.path.join(recordings_dir, f)
                    break

        if not human_file:
            self.logger.info("No human demonstration file found; skipping pre-training.")
            print("No human demonstration file found; skipping pre-training.")
            return

        self.logger.info(f"Human demo found: {human_file}. Starting BC pre-training + experience injection...")
        print(f"Human demo found: {human_file}")

        # ------------------------------------------------------------------
        # 1. Parse the recording into (state_tensor, action_idx) transitions
        # ------------------------------------------------------------------
        transitions: list[dict] = []  # {'state': np.array, 'action_idx': int, 'reward': float}
        try:
            with open(human_file, "r", encoding="utf-8") as fh:
                lines = [json.loads(l) for l in fh if l.strip()]

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

                action_input = data.get("action_input")
                if action_input:
                    prev_action_idx = self._action_input_to_index(
                        action_input.get("id"), action_input.get("data", {})
                    )
                else:
                    prev_action_idx = None

                prev_state_np = state_np

            if not transitions:
                print("No valid transitions found in human demonstration.")
                return

            # Apply discounted rewards — human demo actions are positive BCE targets.
            # Use 1.0 as base (clean BCE target) with mild discounting for earlier steps.
            # Reset discount per level to maintain strong signals for early levels.
            gamma = 0.997
            running = 1.0
            last_score = transitions[-1]['score'] if transitions else 0

            for i in reversed(range(len(transitions))):
                current_score = transitions[i]['score']
                if current_score != last_score:
                    # New level boundary moving backwards, reset running reward
                    running = 1.0
                    last_score = current_score

                transitions[i]['reward'] = running
                running *= gamma

            print(f"Parsed {len(transitions)} transitions from human demo.")

        except Exception as e:
            self.logger.error(f"Error parsing human demo: {e}")
            print(f"Error parsing human demo: {e}")
            traceback.print_exc()
            return

        # ------------------------------------------------------------------
        # 2. Behavioral Cloning — truncated BPTT with sequential chunk ordering.
        #    Chunks are processed IN ORDER with hidden state carried (detached) across
        #    boundaries so the LSTM sees full trajectory context, not isolated windows.
        #    Fresh optimizer avoids stale Adam moments from the loaded RL checkpoint.
        # ------------------------------------------------------------------
        bc_epochs = 150
        chunk_size = 16  # BPTT window — keeps VRAM flat regardless of demo length
        print(f"Behavioral Cloning: {bc_epochs} epochs, chunk={chunk_size} (truncated BPTT), "
              f"{len(transitions)} transitions...")
        try:
            # Build tensors on CPU; move chunk-by-chunk to keep VRAM usage bounded
            all_states_cpu = torch.stack([
                torch.from_numpy(t['state']).float() for t in transitions
            ])  # [N, C, H, W]
            all_labels_cpu = torch.tensor(
                [t['action_idx'] for t in transitions], dtype=torch.long
            )  # [N]

            # Fresh optimizer — no stale momentum/variance from the RL checkpoint
            bc_optimizer = optim.Adam(self.action_model.parameters(), lr=0.001)

            n = len(transitions)
            starts = list(range(0, n, chunk_size))  # sequential order, NOT shuffled
            for epoch in range(bc_epochs):
                epoch_loss = 0.0
                hidden = None  # carry hidden state across chunks (detached each step)
                for start in starts:
                    end = min(start + chunk_size, n)
                    chunk_states = all_states_cpu[start:end].to(self.device).unsqueeze(0)  # [1, L, C, H, W]
                    chunk_labels = all_labels_cpu[start:end].to(self.device)               # [L]
                    bc_optimizer.zero_grad()
                    logits, hidden = self.action_model(chunk_states, hidden)  # [1, L, output_dim]
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
                    # Detach hidden state — prevents gradient graph from growing unbounded
                    hidden = tuple(h.detach() for h in hidden)
                if (epoch + 1) % 30 == 0:
                    avg = epoch_loss / max(len(starts), 1)
                    print(f"  BC epoch {epoch + 1}/{bc_epochs}  avg_loss={avg:.4f}")
            # Restore fresh RL optimizer — BC changed the weight landscape significantly
            self.optimizer = optim.Adam(self.action_model.parameters(), lr=0.0001)
            print("Behavioral Cloning pre-training complete.")
            self.logger.info("Behavioral Cloning pre-training complete.")
        except Exception as e:
            self.logger.error(f"Error during BC pre-training: {e}")
            print(f"Error during BC pre-training: {e}")
            traceback.print_exc()

        # ------------------------------------------------------------------
        # 3. Experience Injection — pin human demos into a permanent buffer
        #    that is blended into every _train_action_model() call.
        # ------------------------------------------------------------------
        self.human_demo_buffer = list(transitions)
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

        # Hierarchical sampling: First select the action type (0-5)
        action_probs = F.softmax(action_logits, dim=0)
        action_probs_np = action_probs.cpu().numpy()

        if np.isnan(action_probs_np).any():
            # Fallback
            action_probs_np = np.ones_like(action_probs_np) / 6.0

        action_idx = np.random.choice(6, p=action_probs_np)

        # Coordinate sampling
        coord_probs = F.softmax(coord_logits, dim=0)
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
        """Train the action model on collected experiences.

        Blends online RL experiences with pinned human demonstration transitions
        (human_demo_buffer) so that the human solution remains an anchor
        throughout all of training, no matter how many RL steps accumulate.
        """
        if full_episode:
            seq_len = len(self.experience_buffer)
            if seq_len < 2:
                return
            sequence = list(self.experience_buffer)
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

        self.optimizer.zero_grad()

        # combined_logits shape: [batch=1, seq_len, output_dim]
        combined_logits, _ = self.action_model(states)

        # --- RL loss: Policy Gradient with categorical cross entropy ---

        # Determine the target labels based on action_indices
        action_type_targets = torch.where(action_indices < 5, action_indices, torch.tensor(5, device=self.device))

        # Cross entropy loss for the action type
        loss_action = F.cross_entropy(
            combined_logits[:, :, :6].reshape(-1, 6),
            action_type_targets.reshape(-1),
            reduction='none'
        ).reshape(1, -1)

        # Mask and calculate coordinate loss where ACTION6 was chosen
        coord_mask = (action_indices >= 5)
        loss_coord = torch.zeros_like(loss_action)
        if coord_mask.any():
            coord_targets = action_indices[coord_mask] - 5
            loss_coord[coord_mask] = F.cross_entropy(
                combined_logits[:, :, 6:][coord_mask],
                coord_targets,
                reduction='none'
            )

        # Apply rewards to the losses
        step_losses = (loss_action + loss_coord) * rewards
        main_loss = step_losses.mean()

        # --- Entropy regularisation (encourages exploration) ---
        action_probs = F.softmax(combined_logits[:, :, :6], dim=-1)
        action_entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(-1).mean()

        coord_probs = F.softmax(combined_logits[:, :, 6:], dim=-1)
        coord_entropy = -(coord_probs * torch.log(coord_probs + 1e-8)).sum(-1).mean()

        total_loss = main_loss - 0.001 * action_entropy - 0.0001 * coord_entropy
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.action_model.parameters(), max_norm=1.0)
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

                self.visitation_counts.clear()
                print("Cleared experience buffer - new level reached")

                # Note: We purposely do NOT reset network and optimizer here anymore,
                # allowing the agent to retain its learned weights across levels of the same game.
                print("Keeping action model and optimizer weights for new level")
                self.hidden_state = None

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

            if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
                self.prev_frame = None
                self.prev_action_idx = None
                action = GameAction.RESET
                action.reasoning = "Game needs reset."
                return action

            # Convert current frame to tensor
            current_frame = self._frame_to_tensor(latest_frame)

            # Create experience from previous action
            if self.prev_frame is not None:
                current_frame_np = current_frame.cpu().numpy().astype(bool)
                frame_changed = not np.array_equal(self.prev_frame, current_frame_np)

                # Calculate curiosity/novelty search reward bonus
                current_grid = np.array(latest_frame.frame, dtype=np.int64)[-1]
                grid_hash = hashlib.md5(current_grid.tobytes()).hexdigest()
                self.visitation_counts[grid_hash] = self.visitation_counts.get(grid_hash, 0) + 1

                # Reward: positive for frame changes (so BCE reinforces good actions),
                # negative for stagnation (so BCE suppresses wasted actions)
                step_reward = 0.1 if frame_changed else -0.1
                reward = step_reward + (0.01 / np.sqrt(self.visitation_counts[grid_hash]))

                experience = {
                    'state': self.prev_frame,
                    'action_idx': self.prev_action_idx,
                    'reward': reward
                }
                self.experience_buffer.append(experience)

            # Get action predictions
            available_actions = getattr(latest_frame, 'available_actions', None)
            self.action_model.eval()
            with torch.no_grad():
                # Add batch and seq_len dimensions: [1, 1, C, H, W]
                current_frame_seq = current_frame.unsqueeze(0).unsqueeze(0)
                combined_logits, self.hidden_state = self.action_model(current_frame_seq, self.hidden_state)
                # Remove batch and seq_len dims for sampling: [1, 1, output_dim] -> [output_dim]
                combined_logits = combined_logits.squeeze(0).squeeze(0)
                action_idx, coords, coord_idx, all_probs = self._sample_from_combined_output(
                    combined_logits, available_actions
                )
            self.action_model.train()

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
