import torch
import torch.nn as nn
from torch_geometric.nn import LayerNorm, TransformerConv


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation.
    Allows the electro-thermal condition (frequency + temperature) to scale
    (gamma) and shift (beta) the geometry features. Simulates how physics rules
    change with frequency and temperature (e.g. skin effect, thermal resistivity).
    """

    def __init__(self, in_channels, cond_channels):
        super().__init__()
        self.gamma = nn.Linear(cond_channels, in_channels)
        self.beta = nn.Linear(cond_channels, in_channels)

    def forward(self, x, cond):
        # cond is the electro-thermal (frequency + temperature) embedding
        gamma = self.gamma(cond)
        beta = self.beta(cond)

        # If batch sizes match (which they should in standard Linear), expand isn't needed,
        # but for graph nodes vs graph-level freq, we need to broadcast.
        # x: [N_nodes, Channels], cond: [N_nodes, Cond_Channels] (Assuming freq repeated per node)
        return x * (1 + gamma) + beta


class TSVPhysicsGNN(nn.Module):
    """Physics-informed Graph Neural Network for TSV S-parameter prediction.

    Architecture:
        1. Separate geometry (5D) and electro-thermal (3D->32D) encoders
        2. FiLM conditioning: [freq, temp, freq*temp] modulates geometry features
        3. 4-layer Graph Transformer with edge attention and residual connections
        4. Dual output heads: node-level (S21, S11) and edge-level (NEXT, FEXT)
           with built-in reciprocity enforcement via forward/backward averaging
    """

    def __init__(self, node_in_dim=7, edge_in_dim=3, hidden_dim=128, layers=4, heads=4,
                 cond_mode="film", reciprocity=True):
        super().__init__()
        self.cond_mode = cond_mode
        self.reciprocity = reciprocity

        # --- 1. Encoders ---
        # Node features: [Type, Radius, Pitch, Height, Liner, Temperature, Freq].
        # Geometry (first node_in_dim-2 cols) is encoded directly; Temperature
        # and Frequency are pulled out as the FiLM conditioner (see forward).
        self.geom_encoder = nn.Linear(node_in_dim - 2, hidden_dim)
        # Conditioner embeds [freq, temp, freq*temp]. The explicit product term
        # hands FiLM the joint electro-thermal coordinate that skin effect
        # couples multiplicatively (R_ac ~ sqrt(f * rho(T))).
        self.cond_encoder = nn.Sequential(
            nn.Linear(3, 32),
            nn.SiLU(),
            nn.Linear(32, 32)
        )
        self.edge_encoder = nn.Linear(edge_in_dim, hidden_dim)

        # --- 2. Conditioning ---
        if cond_mode == "film":
            self.film = FiLM(hidden_dim, 32)
        elif cond_mode == "concat":
            # Parameter-matched alternative: concat geometry emb (128) + cond emb (32)
            # -> bottleneck MLP -> 128, fusing instead of multiplicatively modulating.
            # Linear(160,29)+SiLU+Linear(29,128) = 8509 params (FiLM = 8448; +61, +0.015%).
            self.cond_fuse = nn.Sequential(
                nn.Linear(hidden_dim + 32, 29),
                nn.SiLU(),
                nn.Linear(29, hidden_dim),
            )
        else:
            raise ValueError(f"unknown cond_mode={cond_mode!r}")

        # --- 3. Message Passing (Graph Transformer) ---
        # TransformerConv explicitly calculates edge weights (Attention),
        # allowing it to "learn" shielding (assigning low attention to shielded paths).
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(layers):
            self.layers.append(
                TransformerConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // heads,
                    heads=heads,
                    edge_dim=hidden_dim,  # We will pass encoded edge features here
                    dropout=0.1,
                    beta=True  # Allows bias in attention
                )
            )
            self.norms.append(LayerNorm(hidden_dim))

        # --- 4. Decoders ---
        # Head A: Node Tasks (S21, S11)
        self.node_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),  # SiLU (Swish) is often better for physics than ReLU
            nn.Linear(hidden_dim, 4)
        )

        # Head B: Edge Tasks (NEXT, FEXT)
        self.edge_head = nn.Sequential(
            # Source + Target + EdgeAttr
            nn.Linear(2*hidden_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 4)
        )

    def forward(self, x, edge_index, edge_attr):
        # Node features: [Type, Radius, Pitch, Height, Liner, Temperature, Freq].
        # Geometry = all but the last two columns; Temperature (col -2) and
        # Frequency (col -1) form the electro-thermal FiLM conditioner.
        geom_x = x[:, :-2]
        temp_x = x[:, -2:-1]
        freq_x = x[:, -1:]
        # [freq, temp, freq*temp]: explicit multiplicative term for the joint
        # electro-thermal regime (skin-effect coupling).
        cond_x = torch.cat([freq_x, temp_x, freq_x * temp_x], dim=1)

        # 1. Initial Embeddings
        h = self.geom_encoder(geom_x)
        cond_emb = self.cond_encoder(cond_x)
        e_emb = self.edge_encoder(edge_attr)

        # 2. Apply Physics Conditioning
        # FiLM: multiplicative gate of geometry by the electro-thermal regime.
        # concat: parameter-matched fusion (ablation/film).
        if self.cond_mode == "film":
            h = self.film(h, cond_emb)
        else:  # "concat"
            h = self.cond_fuse(torch.cat([h, cond_emb], dim=1))

        # 3. Message Passing
        for layer, norm in zip(self.layers, self.norms):
            h_in = h
            # TransformerConv
            h = layer(h, edge_index, edge_attr=e_emb)
            h = norm(h)
            h = h + h_in # Residual Connection (Vital for deep shielding info)

        # 4. Node Predictions (Nodes don't have directionality issues)
        node_pred = self.node_head(h)

        # 5. SYMMETRIC Edge Predictions
        row, col = edge_index

        # Path A: Source -> Target
        # Input: [Source_Emb, Target_Emb, Edge_Feat]
        features_forward = torch.cat([h[row], h[col], e_emb], dim=1)
        pred_forward = self.edge_head(features_forward)

        if self.reciprocity:
            # Path B: Target -> Source (Swap node embeddings)
            # Note: Edge features (distance) are symmetric, so e_emb stays same.
            features_backward = torch.cat([h[col], h[row], e_emb], dim=1)
            pred_backward = self.edge_head(features_backward)
            # Average to enforce Reciprocity: S_ij = (P(i,j) + P(j,i)) / 2
            edge_pred = (pred_forward + pred_backward) / 2.0
        else:
            # ablation/reciprocity: forward-only, no symmetrization.
            edge_pred = pred_forward

        return node_pred, edge_pred
