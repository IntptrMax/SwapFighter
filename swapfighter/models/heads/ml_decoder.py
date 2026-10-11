import math
import torch
import torch.nn as nn

class MLDecoderHead(nn.Module):

    def __init__(self, in_channels, num_classes,
                 num_heads=8, num_layers=3,
                 group_size=1,
                 dropout=0.0,
                 norm_first=True,
                 feature_dim=None,
                 kv_dim=None):
        super().__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.group_size = group_size
        self.num_queries = (num_classes + group_size - 1) // group_size
        self.num_layers = num_layers
        self.num_heads = num_heads

        kv_dim = kv_dim or in_channels
        self.kv_dim = kv_dim

        self.queries = nn.Parameter(
            torch.zeros(1, self.num_queries, in_channels))
        nn.init.trunc_normal_(self.queries, std=0.02)

        if feature_dim is not None and feature_dim != kv_dim:
            self.kv_proj = nn.Linear(feature_dim, kv_dim)
        else:
            self.kv_proj = nn.Identity()

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=in_channels,
            nhead=num_heads,
            dim_feedforward=in_channels * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=norm_first,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(in_channels)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(in_channels, group_size)

        print(f"[MLDecoderHead] num_classes={num_classes}, "
              f"num_queries={self.num_queries}, "
              f"layers={num_layers}, heads={num_heads}, "
              f"group_size={group_size}, "
              f"in_channels={in_channels}, kv_dim={kv_dim}")

    def forward(self, x, extra_features=None):
        B, C, H, W = x.shape

        seq = x.flatten(2).transpose(1, 2)

        seq = self.kv_proj(seq)

        if extra_features is not None:
            extra_seqs = []
            for feat in extra_features:
                s = feat.flatten(2).transpose(1, 2)
                s = self.kv_proj(s)
                extra_seqs.append(s)
            seq = torch.cat([seq] + extra_seqs, dim=1)

        if seq.shape[-1] != self.in_channels:

            if not hasattr(self, '_kv_align'):
                self._kv_align = nn.Linear(seq.shape[-1],
                                           self.in_channels, bias=False
                                           ).to(seq.device)
            seq = self._kv_align(seq)

        queries = self.queries.expand(B, -1, -1)

        out = self.decoder(queries, seq)

        out = self.norm(out)
        out = self.dropout(out)

        logits = self.classifier(out)
        logits = logits.reshape(B, -1)

        if logits.shape[1] > self.num_classes:
            logits = logits[:, :self.num_classes]

        return logits