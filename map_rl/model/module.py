import torch
import torch.nn as nn
import torch.nn.functional as F
# from pytorch3d.ops import ball_query, sample_farthest_points, knn_points
import xformers.ops as xops

from torch_geometric.nn import MLP, PointTransformerConv
from torch_geometric.nn.pool import radius
from torch_geometric.utils import to_dense_batch

from typing import Optional
from utils.operator import rotary_pe_3d
import numpy as np


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Orthogonal layer initialization matching CleanRL PPO defaults."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class PointNet(nn.Module):
    def __init__(self, input_dim, output_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(input_dim, 256)),
            nn.LayerNorm(256),
            nn.GELU(),
            layer_init(nn.Linear(256, 256)),
            nn.LayerNorm(256),
            nn.GELU(),
            layer_init(nn.Linear(256, output_dim)),
        )

    def forward(self, x, pad_mask=None):   # pad_mask: (B, L) True=pad
        h = self.net(x)                     # (B, L, C)
        if pad_mask is not None:
            h = h.masked_fill(pad_mask[..., None], float('-inf'))
        out = h.max(dim=1).values           # (B, C)
        out[out.eq(float('-inf'))] = 0
        return out

class TransformerLayer(nn.Module):
    def __init__(
        self, 
        d_model=256, 
        n_heads=8, 
        dim_feedforward=1024, 
        dropout=0.1,
        use_xformers: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.use_xformers = use_xformers

        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_ff = nn.Dropout(dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.activation = F.gelu

    def forward(
        self, 
        src: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        coords_src: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, S, _ = src.shape
        
        q = self.W_q(src).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(src).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(src).view(B, S, self.n_heads, self.head_dim)

        if coords_src is not None:
            q = rotary_pe_3d(q, coords_src)
            k = rotary_pe_3d(k, coords_src)
        
        if self.use_xformers:
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            
            attn_bias = None
            if key_padding_mask is not None:
                # xformers >=0.0.23 removed `KeyPaddingMask`. Build a dense bias tensor instead.
                seq_len = key_padding_mask.size(1)
                mask = key_padding_mask[:, None, None, :].to(q.dtype)
                attn_bias = mask.expand(-1, self.n_heads, seq_len, -1) * (-1e9)
            else:
                attn_bias = None
                
            attn = xops.memory_efficient_attention(
                q, k, v,
                attn_bias=attn_bias,
                p=self.dropout_attn.p if self.training else 0.0,
            )  # (B, S, H, D)    

        else:
            # PyTorch's scaled_dot_product_attention expects (B, n_heads, S, head_dim)
            v = v.transpose(1, 2).contiguous()
            # Build an attention mask from the key\_padding\_mask (True → ignore)
            attn_mask = None
            if key_padding_mask is not None:
                # expected shape: (B, 1, 1, K) broadcastable to (B, H, Q, K)
                attn_mask = key_padding_mask[:, None, None, :].to(torch.bool)
            
            attn = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout_attn.p if self.training else 0.0,
            )
            attn = attn.transpose(1, 2).contiguous() # (B, S, H, D)
        
        # Collapse heads ---------------------------------------------------
        attn = attn.reshape(B, S, self.d_model).contiguous()

        # Residual & FF -----------------------------------------------------
        src2 = self.norm1(src + self.dropout_attn(self.out_proj(attn)))
        ff = self.linear2(self.activation(self.linear1(src2)))
        out = self.norm2(src2 + self.dropout_ff(ff))
        return out

class ZeroPos(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.out_dim = out_dim
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.new_zeros(x.size(0), self.out_dim)

class LocalFeatureFusion(nn.Module):
    def __init__(
        self,
        dim: int,
        num_layers: int = 1,
        ff_mult: int = 4,
        radius: float = 0.12,
        k: int = 1,
        dropout: float = 0.1,
        use_rel_pos: bool = False
    ):
        super().__init__()
        self.radius, self.k = radius, k

        self.convs = nn.ModuleList()
        self.ffns = nn.ModuleList()
        self.norm1s = nn.ModuleList()
        self.norm2s = nn.ModuleList()

        for _ in range(num_layers):
            # PointTransformerConv for local feature aggregation.
            # It will update q_feat based on nearby kv_feat.
            pos_nn = (MLP([3, dim, dim], plain_last=False, batch_norm=False) if use_rel_pos else ZeroPos(dim))
            attn_nn = MLP([dim, dim], plain_last=False, batch_norm=False) # Maps q - k + pos_emb

            self.convs.append(
                PointTransformerConv(
                    in_channels=dim,
                    out_channels=dim,
                    pos_nn=pos_nn,
                    attn_nn=attn_nn,
                    add_self_loops=False  # This is a bipartite graph
                )
            )
            self.norm1s.append(nn.LayerNorm(dim))

            # Feed-forward network
            self.ffns.append(
                nn.Sequential(
                    nn.Linear(dim, dim * ff_mult),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim * ff_mult, dim),
                    nn.Dropout(dropout)
                )
            )
            self.norm2s.append(nn.LayerNorm(dim))


    def forward(
        self,
        q_xyz:   torch.Tensor,                # (B, N, 3)
        q_feat:  torch.Tensor,                # (B, N, C)
        kv_xyz:  torch.Tensor,                # (B, L, 3)
        kv_feat: torch.Tensor,                # (B, L, C)
        kv_pad:  Optional[torch.Tensor] = None  # (B, L) bool
    ) -> torch.Tensor:
        B, N, C = q_feat.shape
        L = kv_xyz.shape[1]

        # 1. Convert dense tensors to PyG format (flat vectors + batch indices)
        q_xyz_flat = q_xyz.reshape(-1, 3)
        out = q_feat.reshape(-1, C)
        q_batch = torch.arange(B, device=q_xyz.device).repeat_interleave(N)

        if kv_pad is not None:
            kv_mask = ~kv_pad
            kv_xyz_flat = kv_xyz[kv_mask]
            kv_feat_flat = kv_feat[kv_mask]
            kv_batch_full = torch.arange(B, device=kv_xyz.device).unsqueeze(1).expand(B, L)
            kv_batch = kv_batch_full[kv_mask]
        else:
            kv_xyz_flat = kv_xyz.reshape(-1, 3)
            kv_feat_flat = kv_feat.reshape(-1, C)
            kv_batch = torch.arange(B, device=kv_xyz.device).repeat_interleave(L)

        # 2. Find neighbors from kv for each q point
        target_idx, source_idx = radius(x=kv_xyz_flat, y=q_xyz_flat, r=self.radius,
                          batch_x=kv_batch, batch_y=q_batch, max_num_neighbors=self.k)

        edge_index = torch.stack([source_idx, target_idx], dim=0)

        # 3. Apply layers of PointTransformerConv for bipartite cross-attention
        for i in range(len(self.convs)):
            updated_q_feat = self.convs[i](
                x=(kv_feat_flat, out),
                pos=(kv_xyz_flat, q_xyz_flat),
                edge_index=edge_index
            )

            # Residual connection, FFN, and normalization
            out = self.norm1s[i](out + updated_q_feat)
            out2 = self.ffns[i](out)
            out = self.norm2s[i](out + out2)

        # 5. Convert back to dense tensor (B, N, C)
        final_feat, _ = to_dense_batch(out, q_batch, batch_size=B, max_num_nodes=N)

        return final_feat