import torch
import torch.nn as nn


def compute_attention_rollout(attn_stack):
    # attn_stack: [batch, num_layers, num_heads, num_nodes, num_nodes]
    attn = attn_stack.mean(dim=2)
    num_nodes = attn.size(-1)
    eye = torch.eye(num_nodes, device=attn.device, dtype=attn.dtype).unsqueeze(0).unsqueeze(0)
    attn = attn + eye
    attn = attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-6)

    rollout = attn[:, 0]
    for layer_idx in range(1, attn.size(1)):
        rollout = torch.bmm(attn[:, layer_idx], rollout)
    return rollout


class AttnTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_attn_weights = None

    def _sa_block(self, x, attn_mask, key_padding_mask, is_causal=False):
        x, attn_weights = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
            is_causal=is_causal,
        )
        self.last_attn_weights = attn_weights.detach()
        return self.dropout1(x)


class SimpleTransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_classes, num_heads=4, num_layers=2, dropout=0.25):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, model_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, input_dim, model_dim))
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)

        transformer_layer = AttnTransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(model_dim)
        self.output_fc = nn.Linear(model_dim, num_classes)

    def _collect_attn_stack(self):
        all_layers = []
        for layer in self.transformer.layers:
            if layer.last_attn_weights is not None:
                all_layers.append(layer.last_attn_weights)
        if not all_layers:
            return None
        return torch.stack(all_layers, dim=1)

    def forward(self, x, return_attention=False, use_rollout=False):
        # x: [batch, num_nodes, num_nodes]
        x = self.input_fc(x)
        x = x + self.pos_embedding[:, :x.size(1), :]

        for layer in self.transformer.layers:
            layer.last_attn_weights = None
        x = self.transformer(x)

        x = self.output_norm(x.mean(dim=1))
        logits = self.output_fc(x)
        if not return_attention:
            return logits

        attn_stack = self._collect_attn_stack()
        if attn_stack is None:
            if use_rollout:
                return logits, None, None
            return logits, None

        last_layer_head_mean = attn_stack[:, -1].mean(dim=1)
        if use_rollout:
            rollout = compute_attention_rollout(attn_stack)
            return logits, last_layer_head_mean, rollout
        return logits, last_layer_head_mean
