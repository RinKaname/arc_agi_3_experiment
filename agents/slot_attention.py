import torch
import torch.nn as nn
import torch.nn.functional as F

class SlotAttention(nn.Module):
    """
    Slot Attention module as described in "Object-Centric Learning with Slot Attention" (Locatello et al., 2020).
    Dynamically groups visual features into K discrete "slots" (objects) using iterative cross-attention.
    """
    def __init__(self, num_slots, dim, iters=3, eps=1e-8, hidden_dim=128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        # Learnable Gaussian distributions for initializing the slots
        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.xavier_uniform_(self.slots_mu)

        # Attention projections
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        self.gru = nn.GRUCell(dim, dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim)
        )

        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs, num_slots=None):
        """
        inputs: [Batch, N_pixels, Feature_dim]
        returns: [Batch, num_slots, Feature_dim]
        """
        b, n, d = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots

        # 1: Initialize slots from Gaussian
        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_logsigma.exp().expand(b, n_s, -1)
        slots = mu + sigma * torch.randn_like(mu)

        # Project inputs to Keys and Values
        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)

        # Iterative Attention (T=3 typically)
        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            # Dot product attention: [b, n_s, d] x [b, n, d] -> [b, n_s, n]
            dots = torch.einsum('bsd,bnd->bsn', q, k) * self.scale

            # The Competition: Softmax over the slots (dim=1)
            # This forces each pixel to vote for which slot it belongs to
            attn = dots.softmax(dim=1) + self.eps

            # Weighted mean: Normalize the attention weights over the pixels (dim=2)
            attn = attn / attn.sum(dim=2, keepdim=True)

            # Aggregate values
            updates = torch.einsum('bsn,bnd->bsd', attn, v)

            # GRU update per slot
            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )
            slots = slots.reshape(b, n_s, d)

            # Residual MLP per slot
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots


class SpatialBroadcastDecoder(nn.Module):
    """
    Broadcasts 1D slot vectors into a 2D spatial grid so the agent can still predict spatial coordinate actions.
    """
    def __init__(self, slot_dim, grid_size=64, out_channels=128):
        super().__init__()
        self.grid_size = grid_size
        self.slot_dim = slot_dim

        # We append X and Y coordinates to the broadcasted slot
        self.decoder_initial_dim = slot_dim + 2

        self.conv1 = nn.Conv2d(self.decoder_initial_dim, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, out_channels, 3, padding=1)

        # Spatial meshgrid for positional awareness
        y_coords = torch.linspace(-1, 1, grid_size).view(-1, 1).repeat(1, grid_size)
        x_coords = torch.linspace(-1, 1, grid_size).view(1, -1).repeat(grid_size, 1)
        self.register_buffer('meshgrid', torch.stack([y_coords, x_coords], dim=0))

    def forward(self, slots):
        """
        slots: [B, num_slots, slot_dim]
        returns: [B, out_channels, H, W] unified spatial representation
        """
        B, num_slots, D = slots.shape

        # Collapse batch and slots: [B*num_slots, D, 1, 1]
        x = slots.view(B * num_slots, D, 1, 1)

        # Spatial broadcast to [B*num_slots, D, H, W]
        x = x.expand(-1, -1, self.grid_size, self.grid_size)

        # Append meshgrid
        mesh = self.meshgrid.unsqueeze(0).expand(B * num_slots, -1, -1, -1)
        x = torch.cat([x, mesh], dim=1)

        # Decode spatial features
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x)) # [B*num_slots, out_channels, H, W]

        # Sum across the slots to get the final unified spatial representation
        x = x.view(B, num_slots, -1, self.grid_size, self.grid_size)
        out = x.sum(dim=1) # [B, out_channels, H, W]

        return out
