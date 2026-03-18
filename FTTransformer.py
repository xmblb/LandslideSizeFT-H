import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import numpy as np
from scipy.stats import pearsonr
from typing import List, Dict, Tuple, Optional
import pandas as pd

def encode_categoricals(
    df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame, cat_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Dict]]:
    """
    Integer-encode categoricals based on TRAIN categories.
    Unknown categories mapped to 0; known categories start at 1.
    Returns mappings so you can reuse later.
    """
    mappings = {}
    for c in cat_cols:
        cats = pd.Series(df_train[c].astype("category")).cat.categories.tolist()
        cat_to_idx = {cat: i + 1 for i, cat in enumerate(cats)}  # 1..K
        mappings[c] = {"cat_to_idx": cat_to_idx, "num_classes": len(cats) + 1}  # +1 for unknown=0

        def map_series(s: pd.Series) -> np.ndarray:
            return s.map(cat_to_idx).fillna(0).astype(int).values

        df_train[c] = map_series(df_train[c])
        df_val[c] = map_series(df_val[c])
        df_test[c] = map_series(df_test[c])
    return df_train, df_val, df_test, mappings


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device,
) -> Dict[str, float]:
    model.eval()
    zs, mus, sigmas = [], [], []
    nlls = []

    for x_num, x_cat, y_z in loader:
        x_num = x_num.to(device)
        x_cat = x_cat.to(device)
        y_z = y_z.to(device)

        mu, sigma = model(x_num, x_cat)
        nll = (y_z - mu) ** 2 / (2 * sigma ** 2) + torch.log(sigma)

        zs.append(y_z.cpu().numpy())
        mus.append(mu.cpu().numpy())
        sigmas.append(sigma.cpu().numpy())
        nlls.append(nll.cpu().numpy())

    z = np.vstack(zs).reshape(-1)
    mu = np.vstack(mus).reshape(-1)
    sigma = np.vstack(sigmas).reshape(-1)
    nll = np.vstack(nlls).reshape(-1)

    mae_z = float(np.mean(np.abs(z - mu)))
    rmse_z = float(np.sqrt(np.mean((z - mu) ** 2)))
    r_z = pearsonr(np.array(z).ravel(), mu)[0]
    mean_nll = float(np.mean(nll))

    # 90% prediction interval coverage on z
    lower = mu - 1.645 * sigma
    upper = mu + 1.645 * sigma
    coverage_90 = float(np.mean((z >= lower) & (z <= upper)))

    # also compute MAE on original y_ratio (optional)
    y_true = np.exp(z)
    y_pred_med = np.exp(mu)  # median of lognormal
    mae_y = float(np.mean(np.abs(y_true - y_pred_med)))

    r_y = pearsonr(np.array(y_true).ravel(), y_pred_med)[0]

    # tail MAE on y (top 10% by true y)
    thr = np.quantile(y_true, 0.9)
    mask = y_true >= thr
    tail_mae_y = float(np.mean(np.abs(y_true[mask] - y_pred_med[mask]))) if np.any(mask) else float("nan")

    return {
        "MAE_z": mae_z,
        "RMSE_z": rmse_z,
        "NLL": mean_nll,
        "Coverage90_z": coverage_90,
        "MAE_y": mae_y,
        "TailMAE_y": tail_mae_y,
        "R_z": r_z,
        "R_y": r_y
    }

class TabDataset(torch.utils.data.Dataset):
    def __init__(self, X_num: np.ndarray, X_cat: np.ndarray, y_z: np.ndarray):
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.X_cat = torch.tensor(X_cat, dtype=torch.long) if X_cat.size else torch.zeros((len(X_num), 0), dtype=torch.long)
        self.y = torch.tensor(y_z, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return self.X_num.shape[0]

    def __getitem__(self, idx):
        return self.X_num[idx], self.X_cat[idx], self.y[idx]



class TransformerEncoderLayerWithAttn(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation="gelu", norm_first=True):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if activation == "gelu":
            self.activation = F.gelu
        elif activation == "relu":
            self.activation = F.relu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def _sa_block(self, x):
        # attn_w: (B, n_heads, T, T) because average_attn_weights=False
        attn_out, attn_w = self.self_attn(
            x, x, x, need_weights=True, average_attn_weights=False
        )
        return self.dropout1(attn_out), attn_w

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    def forward(self, x):
        attn_w = None
        if self.norm_first:
            sa_out, attn_w = self._sa_block(self.norm1(x))
            x = x + sa_out
            x = x + self._ff_block(self.norm2(x))
        else:
            sa_out, attn_w = self._sa_block(x)
            x = self.norm1(x + sa_out)
            x = self.norm2(x + self._ff_block(x))
        return x, attn_w

# ----------------------------
# FT-Transformer model
# ----------------------------
class FTTransformer(nn.Module):
    """
    Minimal FT-Transformer-like architecture:
      - Each numeric feature -> token embedding via (x * w + b) to R^d
      - Each categorical feature -> embedding to R^d
      - Tokens -> TransformerEncoder
      - Pool (mean) -> heads output mu and sigma for z = log(y)
    """

    def __init__(
        self,
        n_num: int,
        cat_cardinalities: List[int],
        d_token: int = 32,
        n_heads: int = 4,
        n_layers: int = 4,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.n_num = n_num
        self.n_cat = len(cat_cardinalities)
        self.d_token = d_token

        # Numeric feature -> token: t_i = x_i * w_i + b_i (each produces a d_token vector)
        # Implemented as learnable (w, b) per numeric feature with shape (n_num, d_token)
        self.num_w = nn.Parameter(torch.randn(n_num, d_token) * 0.02)
        self.num_b = nn.Parameter(torch.zeros(n_num, d_token))


        # Categorical embeddings
        self.cat_embeds = nn.ModuleList([nn.Embedding(card, d_token) for card in cat_cardinalities])

        # Optional [CLS] token (often helps)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_token))
        nn.init.normal_(self.cls, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_token * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.layers = nn.ModuleList([
            TransformerEncoderLayerWithAttn(
                d_model=d_token,
                nhead=n_heads,
                dim_feedforward=d_token * 4,
                dropout=dropout,
                activation="gelu",
                norm_first=True,
            )
            for _ in range(n_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_token)

        # Heads for heteroscedastic Normal on z: mu(x), sigma(x)
        self.mu_head = nn.Sequential(
            nn.Linear(d_token, d_token),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_token, 1),
        )
        self.sigma_head = nn.Sequential(
            nn.Linear(d_token, d_token),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_token, 1),
        )

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor, return_attn: bool = False):
        B = x_num.size(0)

        num_tokens = x_num.unsqueeze(-1) * self.num_w.unsqueeze(0) + self.num_b.unsqueeze(0)

        cat_tokens = []
        for i, emb in enumerate(self.cat_embeds):
            cat_tokens.append(emb(x_cat[:, i]))
        cat_tokens = torch.stack(cat_tokens, dim=1) if cat_tokens else torch.zeros((B, 0, self.d_token),
                                                                                   device=x_num.device)

        cls_token = self.cls.expand(B, -1, -1)
        tokens = torch.cat([cls_token, num_tokens, cat_tokens], dim=1)  # (B, T, d)
        tokens = self.dropout(tokens)

        attn_list = []  # list of (B, heads, T, T)
        h = tokens
        for layer in self.layers:
            h, attn_w = layer(h)
            if return_attn:
                attn_list.append(attn_w)

        h_cls = self.norm(h[:, 0, :])
        mu = self.mu_head(h_cls)
        raw_sigma = self.sigma_head(h_cls)
        sigma = F.softplus(raw_sigma) + 1e-6
        sigma = torch.clamp(sigma, min=5e-3, max=3.0)

        if return_attn:
            return mu, sigma, attn_list
        return mu, sigma

