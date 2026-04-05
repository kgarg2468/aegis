from __future__ import annotations

from typing import Any


try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - optional dependency for training environments
    torch = None
    nn = object  # type: ignore[assignment]


try:
    from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
except ImportError:  # pragma: no cover - optional dependency for training environments
    TorchModelV2 = object  # type: ignore[assignment]


class GraphAttentionLayer(nn.Module):
    """Simple dense GAT layer over fixed-size adjacency."""

    def __init__(self, in_dim: int, out_dim: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.head_dim = out_dim // heads
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Linear(2 * self.head_dim, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, node_emb: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        batch, num_nodes, _ = node_emb.shape
        h = self.W(node_emb).view(batch, num_nodes, self.heads, self.head_dim)

        h_i = h.unsqueeze(2).expand(-1, -1, num_nodes, -1, -1)
        h_j = h.unsqueeze(1).expand(-1, num_nodes, -1, -1, -1)
        attn_input = torch.cat([h_i, h_j], dim=-1)

        e = self.leaky_relu(self.a(attn_input).squeeze(-1))
        adj_mask = adj.unsqueeze(-1).expand_as(e)
        e = e.masked_fill(adj_mask <= 0, float("-inf"))

        alpha = torch.softmax(e, dim=2)
        alpha = torch.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)

        out = torch.einsum("bnjh,bnjhd->bnhd", alpha, h_j)
        out = out.reshape(batch, num_nodes, -1)

        if out.shape[-1] != node_emb.shape[-1]:
            return self.norm(out)
        return self.norm(out + node_emb)


class AegisBlueNet(TorchModelV2, nn.Module):
    """GNN + LSTM network with dual heads (action type + target node)."""

    def __init__(self, obs_space: Any, action_space: Any, num_outputs: int, model_config: dict, name: str):
        if torch is None:
            raise ImportError("torch is required to instantiate AegisBlueNet")
        if TorchModelV2 is object:
            raise ImportError("ray[rllib] is required to instantiate AegisBlueNet")

        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        cfg = model_config.get("custom_model_config", {})
        self.node_embed_dim = int(cfg.get("node_embed_dim", 64))
        self.global_embed_dim = int(cfg.get("global_embed_dim", 32))
        gnn_layers = int(cfg.get("gnn_layers", 2))
        lstm_hidden = int(cfg.get("lstm_hidden", 128))
        fc_hiddens = cfg.get("fc_hiddens", [256, 128])

        self.node_encoder = nn.Sequential(
            nn.Linear(22, self.node_embed_dim),
            nn.ReLU(),
            nn.Linear(self.node_embed_dim, self.node_embed_dim),
            nn.ReLU(),
        )

        self.gat_layers = nn.ModuleList(
            [GraphAttentionLayer(self.node_embed_dim, self.node_embed_dim) for _ in range(gnn_layers)]
        )

        self.global_encoder = nn.Sequential(
            nn.Linear(6, self.global_embed_dim),
            nn.ReLU(),
        )

        self.alert_encoder = nn.LSTM(
            input_size=28,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )

        combined_dim = self.node_embed_dim + self.global_embed_dim + lstm_hidden

        self.policy_head = nn.Sequential(
            nn.Linear(combined_dim, int(fc_hiddens[0])),
            nn.ReLU(),
            nn.Linear(int(fc_hiddens[0]), int(fc_hiddens[1])),
            nn.ReLU(),
        )
        self.action_type_head = nn.Linear(int(fc_hiddens[1]), 6)
        self.target_head = nn.Linear(int(fc_hiddens[1]) + self.node_embed_dim, 1)

        self.value_head = nn.Sequential(
            nn.Linear(combined_dim, int(fc_hiddens[0])),
            nn.ReLU(),
            nn.Linear(int(fc_hiddens[0]), 1),
        )

        self._value: torch.Tensor | None = None

    def forward(self, input_dict: dict, state: list[torch.Tensor], seq_lens: torch.Tensor):
        obs = input_dict["obs"]
        node_feats = obs["node_features"].float()
        global_feats = obs["global_features"].float()
        adj = obs["adjacency"].float()
        alert_hist = obs["alert_history"].float()

        node_emb = self.node_encoder(node_feats)
        for gat in self.gat_layers:
            node_emb = gat(node_emb, adj)

        node_pooled = node_emb.mean(dim=1)
        global_emb = self.global_encoder(global_feats)

        alert_out, _ = self.alert_encoder(alert_hist)
        alert_emb = alert_out[:, -1, :]

        combined = torch.cat([node_pooled, global_emb, alert_emb], dim=-1)

        policy_hidden = self.policy_head(combined)
        action_logits = self.action_type_head(policy_hidden)

        policy_expanded = policy_hidden.unsqueeze(1).expand(-1, node_emb.shape[1], -1)
        target_input = torch.cat([policy_expanded, node_emb], dim=-1)
        target_logits = self.target_head(target_input).squeeze(-1)

        action_mask = obs.get("action_mask") if isinstance(obs, dict) else None
        if action_mask is not None:
            action_mask = action_mask.float()
            action_valid = action_mask.amax(dim=2)
            target_valid = action_mask.amax(dim=1)
            neg_inf = torch.finfo(action_logits.dtype).min
            action_logits = torch.where(action_valid > 0, action_logits, torch.full_like(action_logits, neg_inf))
            target_logits = torch.where(target_valid > 0, target_logits, torch.full_like(target_logits, neg_inf))

        logits = torch.cat([action_logits, target_logits], dim=-1)
        self._value = self.value_head(combined).squeeze(-1)
        return logits, state

    def value_function(self):
        if self._value is None:
            raise RuntimeError("value_function() called before forward()")
        return self._value
