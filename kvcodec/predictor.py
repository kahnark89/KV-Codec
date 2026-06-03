"""
kvcodec/predictor.py
====================
Lightweight MLP predictor — the motion vector equivalent.

Given an anchor KV vector and a positional offset (Δlayer, Δpos, or both),
predicts the target KV vector. The residual after prediction is smaller
in magnitude than the raw delta, compressing better under INT8 quantization.

Three prediction modes:
  predict()       — sequence axis only (Δpos)
  predict_layer() — layer axis only    (Δlayer)
  predict_joint() — both axes          (Δlayer, Δpos)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class KVPredictor(nn.Module):
    """
    Tiny MLP: (anchor_vec, delta_embedding) → predicted_vec

    Architecture:
      - Anchor projected to hidden_dim
      - Delta(s) embedded via sinusoidal encoding then projected
      - Two residual blocks
      - Output projected back to head_dim

    Shared weights between K and V prediction (separate output heads).
    ~15-50K parameters depending on head_dim and hidden_dim.
    Frozen after calibration training — not updated during inference.
    """

    def __init__(
        self,
        head_dim: int,
        hidden_dim: int = 64,
        n_delta_axes: int = 1,       # 1 = seq or layer only, 2 = joint
        max_delta: int = 128,
        n_sinusoid_freqs: int = 16,
    ):
        super().__init__()
        self.head_dim        = head_dim
        self.hidden_dim      = hidden_dim
        self.n_delta_axes    = n_delta_axes
        self.max_delta       = max_delta
        self.n_sinusoid_freqs = n_sinusoid_freqs

        delta_embed_dim = n_sinusoid_freqs * 2 * n_delta_axes

        # Anchor encoder
        self.anchor_proj = nn.Linear(head_dim, hidden_dim)

        # Delta encoder
        self.delta_proj  = nn.Linear(delta_embed_dim, hidden_dim)

        # Fusion + residual blocks
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        self.block1 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.block2 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Separate output heads for K and V
        self.out_k = nn.Linear(hidden_dim, head_dim)
        self.out_v = nn.Linear(hidden_dim, head_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _sinusoidal_embed(self, delta: torch.Tensor) -> torch.Tensor:
        """
        Encode scalar delta values as sinusoidal features.
        delta: [N] float
        Returns: [N, n_freqs*2]
        """
        freqs = torch.arange(self.n_sinusoid_freqs, dtype=torch.float,
                             device=delta.device)
        freqs = self.max_delta ** (freqs / self.n_sinusoid_freqs)
        angles = delta.unsqueeze(-1) / freqs.unsqueeze(0)   # [N, n_freqs]
        return torch.cat([angles.sin(), angles.cos()], dim=-1)  # [N, n_freqs*2]

    def _encode_deltas(self, *deltas) -> torch.Tensor:
        """Encode one or two delta vectors into a combined embedding."""
        embeds = [self._sinusoidal_embed(d.flatten()) for d in deltas]
        return torch.cat(embeds, dim=-1)  # [N, n_delta_axes * n_freqs*2]

    def forward(
        self,
        anchor_k: torch.Tensor,    # [..., D]
        anchor_v: torch.Tensor,    # [..., D]
        *deltas,                    # one or two delta tensors matching leading dims
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        General forward pass.
        anchor_k/v: arbitrary leading dims + head_dim
        deltas: matching leading dims, scalar per position

        Returns predicted k, v with same shape as inputs.
        """
        orig_shape = anchor_k.shape
        D = orig_shape[-1]
        N = anchor_k.reshape(-1, D).shape[0]

        ak = anchor_k.reshape(N, D)
        av = anchor_v.reshape(N, D)

        # Encode anchor
        h_anchor = F.gelu(self.anchor_proj(ak))  # [N, hidden]

        # Encode deltas — broadcast to N if needed
        delta_flat = []
        for d in deltas:
            df = d.reshape(-1)
            if df.shape[0] == 1:
                df = df.expand(N)
            elif df.shape[0] != N:
                # Tile to match N (handles layer × pos broadcasting)
                df = df.repeat_interleave(N // df.shape[0])[:N]
            delta_flat.append(df)

        delta_embed = self._encode_deltas(*delta_flat)    # [N, embed_dim]
        h_delta = F.gelu(self.delta_proj(delta_embed))    # [N, hidden]

        # Fuse
        h = F.gelu(self.fusion(torch.cat([h_anchor, h_delta], dim=-1)))

        # Residual blocks (shared)
        h = h + self.block1(h)
        h = h + self.block2(h)

        # Output heads — predict residual from anchor, not absolute value
        pred_k = ak + self.out_k(h)   # residual connection from anchor
        pred_v = av + self.out_v(h)

        return pred_k.reshape(orig_shape), pred_v.reshape(orig_shape)

    # ── Convenience wrappers ──────────────────────────────────

    def predict(
        self,
        anchor_k: torch.Tensor,   # [T, H, D]
        anchor_v: torch.Tensor,
        delta_pos: torch.Tensor,  # [T]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sequence-axis prediction."""
        T, H, D = anchor_k.shape
        ak = anchor_k.reshape(T * H, D)
        av = anchor_v.reshape(T * H, D)
        dp = delta_pos.repeat_interleave(H)
        pk, pv = self.forward(ak, av, dp)
        return pk.reshape(T, H, D), pv.reshape(T, H, D)

    def predict_layer(
        self,
        anchor_k: torch.Tensor,    # [L, H, T, D]
        anchor_v: torch.Tensor,
        delta_layer: torch.Tensor, # [L]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Layer-axis prediction."""
        L, H, T, D = anchor_k.shape
        ak = anchor_k.reshape(L * H * T, D)
        av = anchor_v.reshape(L * H * T, D)
        dl = delta_layer.repeat_interleave(H * T)
        pk, pv = self.forward(ak, av, dl)
        return pk.reshape(L, H, T, D), pv.reshape(L, H, T, D)

    def predict_joint(
        self,
        anchor_k: torch.Tensor,    # [L, T, H, D]
        anchor_v: torch.Tensor,
        delta_layer: torch.Tensor, # [L]
        delta_pos: torch.Tensor,   # [T]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Joint 2D prediction."""
        L, T, H, D = anchor_k.shape
        ak = anchor_k.reshape(L * T * H, D)
        av = anchor_v.reshape(L * T * H, D)
        # Broadcast deltas over (L, T, H)
        dl = delta_layer.unsqueeze(1).unsqueeze(2).expand(L, T, H).reshape(-1)
        dp = delta_pos.unsqueeze(0).unsqueeze(2).expand(L, T, H).reshape(-1)
        pk, pv = self.forward(ak, av, dl, dp)
        return pk.reshape(L, T, H, D), pv.reshape(L, T, H, D)


class PredictorTrainer:
    """
    Trains the KVPredictor on a small calibration corpus.
    ~100-500 sequences, 5-20 minutes on CPU, seconds on GPU.
    Frozen after training — no updates during inference.
    """

    def __init__(
        self,
        predictor: KVPredictor,
        mode: str = 'seq',      # 'seq' | 'layer' | 'joint'
        lr: float = 1e-3,
        n_epochs: int = 10,
        device: str = 'cpu',
        verbose: bool = True,
    ):
        self.predictor = predictor.to(device)
        self.mode      = mode
        self.lr        = lr
        self.n_epochs  = n_epochs
        self.device    = device
        self.verbose   = verbose
        self.optimizer = torch.optim.Adam(predictor.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=n_epochs, eta_min=lr * 0.1
        )

    def _log(self, msg): 
        if self.verbose: print(f"[trainer] {msg}")

    def train_on_kv(
        self,
        all_keys_list: list,    # list of samples; each sample = list of [H,T,D] per layer
        all_values_list: list,
        anchor_stride_seq: int = 16,
        anchor_stride_layer: int = 4,
    ):
        """Train predictor on extracted KV tensors."""
        self._log(f"Training {self.mode} predictor on {len(all_keys_list)} samples...")
        self.predictor.train()

        for epoch in range(self.n_epochs):
            total_loss = 0.0
            n_batches  = 0

            for sample_keys, sample_vals in zip(all_keys_list, all_values_list):
                loss = self._sample_loss(
                    sample_keys, sample_vals,
                    anchor_stride_seq, anchor_stride_layer
                )
                if loss is None:
                    continue
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), 1.0)
                self.optimizer.step()
                total_loss += loss.item()
                n_batches  += 1

            self.scheduler.step()
            avg = total_loss / max(n_batches, 1)
            self._log(f"  epoch {epoch+1}/{self.n_epochs}  loss={avg:.6f}  "
                      f"lr={self.scheduler.get_last_lr()[0]:.2e}")

        self.predictor.eval()
        self._log("Training complete.")

    def _sample_loss(self, keys, values, stride_seq, stride_layer):
        """Compute prediction loss for one sample."""
        if self.mode == 'seq':
            return self._seq_loss(keys[0].to(self.device),
                                  values[0].to(self.device), stride_seq)
        elif self.mode == 'layer':
            return self._layer_loss(keys, values, stride_layer)
        elif self.mode == 'joint':
            return self._joint_loss(keys, values, stride_seq, stride_layer)
        return None

    def _seq_loss(self, k, v, stride):
        """k, v: [H, T, D]"""
        H, T, D = k.shape
        anchor_positions = list(range(0, T, stride))
        if not anchor_positions:
            return None
        loss = torch.tensor(0.0, device=self.device)
        count = 0
        for ap in anchor_positions:
            ak = k[:, ap, :].unsqueeze(1).expand(H, min(stride, T-ap), D)
            av = v[:, ap, :].unsqueeze(1).expand(H, min(stride, T-ap), D)
            end = min(ap + stride, T)
            deltas = torch.arange(end - ap, dtype=torch.float, device=self.device)
            target_k = k[:, ap:end, :].permute(1, 0, 2)   # [t, H, D]
            target_v = v[:, ap:end, :].permute(1, 0, 2)
            ak_t = ak.permute(1, 0, 2)
            av_t = av.permute(1, 0, 2)
            pred_k, pred_v = self.predictor.predict(ak_t, av_t, deltas)
            loss += F.mse_loss(pred_k, target_k) + F.mse_loss(pred_v, target_v)
            count += 1
        return loss / max(count, 1)

    def _layer_loss(self, keys, values, stride):
        n = len(keys)
        anchor_layers = list(range(0, n, stride))
        if not anchor_layers:
            return None
        k_stack = torch.stack([k.float().to(self.device) for k in keys])
        v_stack = torch.stack([v.float().to(self.device) for v in values])
        loss = torch.tensor(0.0, device=self.device)
        for al in anchor_layers:
            end = min(al + stride, n)
            deltas = torch.arange(end - al, dtype=torch.float, device=self.device)
            ak = k_stack[al].unsqueeze(0).expand(end-al, -1, -1, -1)
            av = v_stack[al].unsqueeze(0).expand(end-al, -1, -1, -1)
            pred_k, pred_v = self.predictor.predict_layer(ak, av, deltas)
            loss += (F.mse_loss(pred_k, k_stack[al:end]) +
                     F.mse_loss(pred_v, v_stack[al:end]))
        return loss / len(anchor_layers)

    def _joint_loss(self, keys, values, stride_seq, stride_layer):
        n = len(keys)
        H, T, D = keys[0].shape
        anchor_layers = list(range(0, n, stride_layer))
        anchor_pos    = list(range(0, T, stride_seq))
        if not anchor_layers or not anchor_pos:
            return None
        k_stack = torch.stack([k.float().to(self.device) for k in keys]).permute(0,2,1,3)
        v_stack = torch.stack([v.float().to(self.device) for v in values]).permute(0,2,1,3)
        loss = torch.tensor(0.0, device=self.device)
        for al in anchor_layers:
            for ap in anchor_pos:
                le = min(al + stride_layer, n)
                pe = min(ap + stride_seq, T)
                dl = torch.arange(le-al, dtype=torch.float, device=self.device)
                dp = torch.arange(pe-ap, dtype=torch.float, device=self.device)
                ak = k_stack[al, ap].unsqueeze(0).unsqueeze(0).expand(le-al, pe-ap, H, D)
                av = v_stack[al, ap].unsqueeze(0).unsqueeze(0).expand(le-al, pe-ap, H, D)
                pred_k, pred_v = self.predictor.predict_joint(ak, av, dl, dp)
                loss += (F.mse_loss(pred_k, k_stack[al:le, ap:pe]) +
                         F.mse_loss(pred_v, v_stack[al:le, ap:pe]))
        return loss

    def save(self, path: str):
        torch.save({
            'state_dict': self.predictor.state_dict(),
            'head_dim':   self.predictor.head_dim,
            'hidden_dim': self.predictor.hidden_dim,
            'n_delta_axes': self.predictor.n_delta_axes,
            'mode':       self.mode,
        }, path)
        if self.verbose:
            print(f"[trainer] saved → {path}")

    @staticmethod
    def load(path: str, device: str = 'cpu') -> KVPredictor:
        ckpt = torch.load(path, map_location=device)
        p = KVPredictor(
            head_dim=ckpt['head_dim'],
            hidden_dim=ckpt['hidden_dim'],
            n_delta_axes=ckpt['n_delta_axes'],
        ).to(device)
        p.load_state_dict(ckpt['state_dict'])
        p.eval()
        return p
