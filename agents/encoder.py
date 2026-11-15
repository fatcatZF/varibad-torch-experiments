"""Recurrent task encoder ``q_phi(m | tau_{:t})`` of VariBAD (Zintgraf et al., ICLR 2020).

The encoder is a GRU that consumes one transition per step and, after every step, emits the
parameters of a diagonal Gaussian over the task embedding ``m``.  The belief before the first
transition is the fixed prior ``N(0, I)``.

Shape conventions (``L`` = sequence length, ``N`` = batch of trajectories, ``Z`` = latent dim)::

    obs, next_obs   (L, N, obs_dim)       already normalised
    act             (L, N) int64          discrete
                    (L, N, action_dim)    continuous
    rew             (L, N, 1)             already normalised
    done            (L, N, 1)             1 if the episode ended at this transition
    h               (1, N, hidden)        GRU state
    mu, logvar      (L, N, Z)
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

LOGVAR_MIN, LOGVAR_MAX = -10.0, 5.0


class RNNEncoder(nn.Module):
    """GRU encoder over transitions.

    Parameters
    ----------
    obs_dim : int
    action_dim : int
        Number of discrete actions, or the action-vector size.
    is_discrete : bool
    latent_dim : int
        Size ``Z`` of the task embedding (paper: 5).
    state_embed, action_embed, reward_embed : int
        Sizes of the input embeddings (paper, MuJoCo: 32 / 16 / 16).
    hidden_size : int
        GRU hidden size (paper, MuJoCo: 128).
    use_prev_state : bool
        If True the transition input contains both ``s_t`` and ``s_{t+1}`` (so the GRU sees the
        full transition ``(s, a, r, s')`` as in PEARL).  If False it only sees ``(a, r, s')``
        like the paper's description of ``tau``; then the GRU has to remember ``s_t`` itself.
    """

    def __init__(self, obs_dim: int, action_dim: int, is_discrete: bool, latent_dim: int = 5,
                 state_embed: int = 32, action_embed: int = 16, reward_embed: int = 16,
                 hidden_size: int = 128, use_prev_state: bool = True):
        super().__init__()
        self.is_discrete, self.action_dim = is_discrete, action_dim
        self.latent_dim, self.hidden_size = latent_dim, hidden_size
        self.use_prev_state = use_prev_state

        self.next_state_fc = nn.Linear(obs_dim, state_embed)
        self.prev_state_fc = nn.Linear(obs_dim, state_embed) if use_prev_state else None
        self.action_fc = nn.Embedding(action_dim, action_embed) if is_discrete else nn.Linear(action_dim, action_embed)
        self.reward_fc = nn.Linear(1, reward_embed)
        in_dim = state_embed * (2 if use_prev_state else 1) + action_embed + reward_embed + 1  # +1: done flag
        self.gru = nn.GRU(in_dim, hidden_size)
        self.mu_head = nn.Linear(hidden_size, latent_dim)
        self.logvar_head = nn.Linear(hidden_size, latent_dim)

    def initial_state(self, batch: int, device) -> torch.Tensor:
        """Zero GRU state, shape (1, batch, hidden)."""
        return torch.zeros(1, batch, self.hidden_size, device=device)

    def forward(self, obs, act, rew, next_obs, done, h0: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode a sequence of transitions.

        Returns
        -------
        mu, logvar : (L, N, Z)
            Parameters of ``q(m | tau_{:t+1})`` after consuming transition ``t`` (t = 0..L-1).
        h_n : (1, N, hidden)
        """
        a = self.action_fc(act.long()) if self.is_discrete else self.action_fc(act)
        feats = [F.relu(self.next_state_fc(next_obs)), F.relu(a), F.relu(self.reward_fc(rew)), done]
        if self.use_prev_state:
            feats.insert(0, F.relu(self.prev_state_fc(obs)))
        out, h_n = self.gru(torch.cat(feats, dim=-1), h0)
        return self.mu_head(out), self.logvar_head(out).clamp(LOGVAR_MIN, LOGVAR_MAX), h_n


def kl_diag_gaussians(mu1, logvar1, mu0, logvar0) -> torch.Tensor:
    """``KL( N(mu1, e^logvar1) || N(mu0, e^logvar0) )`` summed over the last dim.

    All arguments broadcast; returns a tensor with the last dim reduced.
    """
    return 0.5 * (logvar0 - logvar1 + (logvar1.exp() + (mu1 - mu0).pow(2)) / logvar0.exp() - 1.0).sum(-1)
