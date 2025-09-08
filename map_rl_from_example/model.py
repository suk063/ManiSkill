import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List
from torch.cuda.amp import autocast
from torch.distributions.normal import Normal
from torchvision import transforms
import torch.nn.functional as F
from torch_geometric.nn import MLP, PointTransformerConv
from torch_geometric.nn.pool import radius
from torch_geometric.utils import to_dense_batch
from map_rl_from_example.operator import get_3d_coordinates

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

transform = transforms.Compose(
    [
        transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ]
)

class DINO2DFeatureEncoder(nn.Module):
    """
    Thin wrapper around DINOv2 ViT-S/14 to produce dense 2D feature maps.
    Inputs are expected in shape (B, C, H, W) with values in [0, 1].
    Normalization is applied internally.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        model_name: str = "dinov2_vits14",
    ) -> None:
        super().__init__()

        # Load backbone lazily via torch.hub to avoid extra dependencies
        self.backbone = torch.hub.load("facebookresearch/dinov2", model_name)
        self.dino_output_dim = 384
        self.dino_head = nn.Sequential(
            nn.Conv2d(self.dino_output_dim, 256, kernel_size=1, bias=False),
            nn.GroupNorm(1, 256),  # Equivalent to LayerNorm for channels
            nn.GELU(),
            nn.Conv2d(256, embed_dim, kernel_size=1),
        )
        self.embed_dim = embed_dim
        self.transform = transform

        for p in self.backbone.parameters():
            p.requires_grad = False

    def _forward_dino_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns per-patch token embeddings without the [CLS] token.
        Shape: (B, N, C), where N = (H/14)*(W/14), C = embed_dim.
        """
        x = self.backbone.prepare_tokens_with_masks(x)
        for blk in self.backbone.blocks:
            x = blk(x)
        x = self.backbone.norm(x)  # (B, 1 + N, C)
        x = x[:, 1:, :]            # drop CLS → (B, N, C)
        return x

    def forward(self, images_bchw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images_bchw: Float tensor in [0, 1], shape (B, 3, H, W)

        Returns:
            fmap: (B, C, Hf, Wf) where C = embed_dim and Hf = H//14, Wf = W//14
        """
        if images_bchw.dtype != torch.float32:
            images_bchw = images_bchw.float()

        images_bchw = self.transform(images_bchw)
        B, _, H, W = images_bchw.shape
        
        with torch.no_grad():
            tokens = self._forward_dino_tokens(images_bchw)

        C = self.dino_output_dim
        Hf, Wf = H // 14, W // 14
        fmap = tokens.permute(0, 2, 1).reshape(B, C, Hf, Wf).contiguous()
        fmap = self.dino_head(fmap)
        
        return fmap

class PointNet(nn.Module):
    def __init__(self, input_dim, output_dim=256, L=10):
        super().__init__()
        self.L = L
        pe_dim = 3 * 2 * self.L
        self.net = nn.Sequential(
            layer_init(nn.Linear(input_dim + pe_dim, 256)),
            nn.LayerNorm(256),
            nn.GELU(),
            layer_init(nn.Linear(256, 256)),
            nn.LayerNorm(256),
            nn.GELU(),
            layer_init(nn.Linear(256, output_dim)),
        )

    def _sinusoidal_pe(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: (..., 3)
        # returns: (..., 3 * 2 * L)
        freqs = 2**torch.arange(self.L, device=coords.device).float()
        
        # unsqueeze to broadcast with coords
        freqs = freqs.view(*([1] * (coords.dim() - 1)), -1)
        
        coords_scaled = coords.unsqueeze(-1) * freqs
        
        sines = torch.sin(coords_scaled)
        cosines = torch.cos(coords_scaled)
        
        pe = torch.cat([sines, cosines], dim=-1)
        
        # reshape to (..., 3 * 2 * L)
        pe = pe.view(*coords.shape[:-1], -1)
        return pe

    def forward(self, x, coords, pad_mask=None):   # pad_mask: (B, L) True=pad
        pe = self._sinusoidal_pe(coords)
        if pad_mask is not None:
            pe = pe.masked_fill(pad_mask[..., None], 0)
        x_with_pe = torch.cat([x, pe], dim=-1)
        h = self.net(x_with_pe)                     # (B, L, C)
        if pad_mask is not None:
            h = h.masked_fill(pad_mask[..., None], float('-inf'))
        out = h.max(dim=1).values           # (B, C)
        out[out.eq(float('-inf'))] = 0
        return out


class NatureCNN(nn.Module):
    def __init__(self, sample_obs, vision_encoder: str, num_tasks: int = 0, decoder: Optional[nn.Module] = None, use_map: bool = False, device=None, start_condition_map: bool = False, use_local_fusion: bool = False, use_rel_pos_in_fusion: bool = True, use_online_mapping: bool = False):
        super().__init__()
        self.vision_encoder_name = vision_encoder
        self.vision_encoder = None

        self.out_features = 0
        feature_size = 256
        in_channels=sample_obs["rgb"].shape[-1]
        image_size=(sample_obs["rgb"].shape[1], sample_obs["rgb"].shape[2])


        # here we use a NatureCNN architecture to process images, but any architecture is permissble here
        if self.vision_encoder_name == "dino":
            self.vision_encoder = DINO2DFeatureEncoder(embed_dim=64)
            self.embed_dim = self.vision_encoder.embed_dim
            self.cnn_part = self.vision_encoder
            self.flatten_part = nn.Flatten()

            # to easily figure out the dimensions after flattening, we pass a test tensor
            with torch.no_grad():
                sample_input = sample_obs["rgb"].float().permute(0,3,1,2) / 255.0

                # Temporarily move to the target device for shape inference to avoid xformers error on CPU
                if device and "cuda" in str(device):
                    cnn_tmp = self.cnn_part.to(device)
                    sample_tmp = sample_input.to(device)
                    with autocast():
                        n_flatten = self.flatten_part(cnn_tmp(sample_tmp)).shape[1]
                    # The cnn object itself is not moved, so it stays on CPU.
                else:
                    # This will raise the xformers error if not on CUDA, which is expected.
                    n_flatten = self.flatten_part(self.cnn_part(sample_input.cpu())).shape[1]
                
                self.fc_part = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
        elif self.vision_encoder_name == "plain_cnn":
            self.embed_dim = 64
            self.cnn_part = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=32,
                    kernel_size=8,
                    stride=4,
                    padding=0,
                ),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0
                ),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=64, out_channels=self.embed_dim, kernel_size=3, stride=1, padding=0
                ),
                nn.ReLU(),
            )
            self.flatten_part = nn.Flatten()
            # to easily figure out the dimensions after flattening, we pass a test tensor
            with torch.no_grad():
                n_flatten = self.flatten_part(self.cnn_part(sample_obs["rgb"].float().permute(0,3,1,2).cpu() / 255.0)).shape[1]
                self.fc_part = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
        else:
            raise ValueError(f"Unknown vision_encoder: {self.vision_encoder_name}")
        self.out_features += feature_size

        self.state_extractor = None
        if "state" in sample_obs:
            # for state data we simply pass it through a single linear layer
            state_size = sample_obs["state"].shape[-1]
            self.state_extractor = nn.Linear(state_size, 256)
            self.out_features += 256

        self.task_embedding = None
        if num_tasks > 0:
            self.task_embedding = nn.Embedding(num_tasks, feature_size)
            self.out_features += feature_size

        # Map-related components
        object.__setattr__(self, "_decoder", decoder)
        self.use_map = use_map
        self.start_condition_map = start_condition_map
        self.use_online_mapping = use_online_mapping
        self.use_local_fusion = self.use_map and use_local_fusion
        self.last_gate_value = None
        if self.use_map:
            map_raw_dim = 768  # output dim of *decoder*
            self.map_encoder = PointNet(map_raw_dim, feature_size)
            self.out_features += feature_size
            self.map_gate_trainable = nn.Parameter(torch.full((feature_size,), -5.0))
            # self.map_gate = nn.Parameter(torch.zeros((feature_size,)))
            # if not self.start_condition_map:
            #     self.map_gate_trainable.requires_grad = False
            
            if self.use_local_fusion:
                self.map_feature_proj = nn.Linear(map_raw_dim, self.embed_dim)
                self.local_fusion = LocalFeatureFusion(dim=self.embed_dim, k=1, radius=0.12, num_layers=1, use_rel_pos=use_rel_pos_in_fusion)


    def _local_fusion(
        self,
        observations: dict,
        image_fmap: torch.Tensor,
        coords_batch: List[torch.Tensor],
        dec_split: List[torch.Tensor],
    ) -> torch.Tensor:
        B = image_fmap.size(0)

        kv_xyz = torch.nn.utils.rnn.pad_sequence(coords_batch, batch_first=True)  # (B,Lmax,3)
        kv_raw = torch.nn.utils.rnn.pad_sequence(dec_split, batch_first=True)     # (B,Lmax,768)

        Lmax = kv_xyz.size(1)
        pad_mask = torch.arange(Lmax, device=kv_xyz.device).expand(B, -1)
        pad_mask = pad_mask >= torch.tensor(
            [c.size(0) for c in coords_batch], device=kv_xyz.device
        ).unsqueeze(1)  # (B,Lmax)

        kv_feat = self.map_feature_proj(kv_raw)  # (B,Lmax,embed_dim)

        depth = observations["depth"].permute(0, 3, 1, 2).float() / 1000.0
        pose = observations["sensor_param"]["hand_camera"]["extrinsic_cv"]

        Hf = Wf = image_fmap.size(2)
        depth_s = F.interpolate(depth, size=(Hf, Wf), mode="nearest-exact")
        
        fx = observations["sensor_param"]["hand_camera"]["intrinsic_cv"][0][0][0]
        fy = observations["sensor_param"]["hand_camera"]["intrinsic_cv"][0][1][1]
        cx = observations["sensor_param"]["hand_camera"]["intrinsic_cv"][0][0][2]
        cy = observations["sensor_param"]["hand_camera"]["intrinsic_cv"][0][1][2]

        q_xyz, _ = get_3d_coordinates(
            depth_s,
            pose,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            original_size=(224, 224),
        )  # B, H, W, 3

        q_xyz = q_xyz.permute(0, 2, 3, 1).reshape(B, -1, 3)            # (B, N, 3)
        q_feat = image_fmap.permute(0, 2, 3, 1).reshape(B, -1, self.embed_dim)     # (B, N, C)

        fused = self.local_fusion(q_xyz, q_feat, kv_xyz, kv_feat, pad_mask)  # (B, N, C)
        image_fmap = fused.permute(0, 2, 1).reshape(
            B, self.embed_dim, Hf, Wf
        )  # (B, C, Hf, Wf)

        return image_fmap

    def forward(self, observations, env_target_obj_idx=None, map_features=None) -> torch.Tensor:
        # 1. Process RGB features
        obs = observations["rgb"]
        obs = obs.float().permute(0,3,1,2) / 255.0
        image_fmap = self.cnn_part(obs)

        # 2. Process Map features and fuse with RGB if enabled
        map_vec = None
        if self.use_map:
            if self.use_local_fusion or self.start_condition_map:
                assert map_features is not None, "map_features must be provided when use_map=True and (start_condition_map=True or use_local_fusion=True)"
                
                coords_batch, raw_batch = [], []
                for g in map_features:
                    level = g.levels[1]
                    if self.use_online_mapping:
                        accessed_indices = level.get_accessed_indices()
                        coords = level.coords[accessed_indices]
                    else:
                        coords = level.coords
                    coords_batch.append(coords)
                    raw_batch.append(g.query_voxel_feature(coords))
                # The decoder is pre-trained and used in inference mode.
                with torch.no_grad():
                    dec_cat = self._decoder(torch.cat(raw_batch, dim=0))
                    dec_split = dec_cat.split([c.size(0) for c in coords_batch], dim=0)

                    if self.use_local_fusion:
                        image_fmap = self._local_fusion(observations, image_fmap, coords_batch, dec_split)
                    
                    if self.start_condition_map:
                        lengths = [c.size(0) for c in coords_batch]
                        pad_3d = torch.nn.utils.rnn.pad_sequence(dec_split, batch_first=True)
                        coords_padded = torch.nn.utils.rnn.pad_sequence(coords_batch, batch_first=True)
                        Lmax = pad_3d.size(1)

                        pad_mask = torch.arange(Lmax, device=pad_3d.device).expand(len(lengths), -1) >= \
                                torch.tensor(lengths, device=pad_3d.device)[:, None]
                        
                        map_vec = self.map_encoder(pad_3d, coords_padded, pad_mask)

            if map_vec is None:
                # Get feature size from the map encoder's last layer
                map_feature_size = self.map_encoder.net[-1].out_features
                # Get batch size from any observation tensor
                some_obs_tensor = next(iter(observations.values()))
                batch_size = some_obs_tensor.shape[0]
                device = some_obs_tensor.device
                map_vec = torch.zeros(batch_size, map_feature_size, device=device)

        # 3. Build the final feature list in the correct order
        flattened_fmap = self.flatten_part(image_fmap)
        rgb_features = self.fc_part(flattened_fmap)
        encoded_tensor_list = [rgb_features]

        if self.state_extractor is not None:
            encoded_tensor_list.append(self.state_extractor(observations["state"]))
        
        if self.task_embedding is not None:
            assert env_target_obj_idx is not None, "env_target_obj_idx must be provided for task embedding"
            task_emb = self.task_embedding(env_target_obj_idx)
            encoded_tensor_list.append(task_emb)

        if map_vec is not None:
            ### DEBUG: just use the map_vec directly
            if self.start_condition_map:
                gate = torch.sigmoid(self.map_gate_trainable)
                self.last_gate_value = gate
                encoded_tensor_list.append(gate * map_vec)
            else:
                encoded_tensor_list.append(map_vec)
            # gate = torch.sigmoid(self.map_gate)
            # encoded_tensor_list.append(gate * map_vec)

        return torch.cat(encoded_tensor_list, dim=1)



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
        k: int = 2,
        dropout: float = 0.1,
        use_rel_pos: bool = True
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

        self.fusion_gate = nn.Parameter(torch.zeros(dim))


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
            out = self.norm1s[i](out + self.fusion_gate * updated_q_feat)
            out2 = self.ffns[i](out)
            out = self.norm2s[i](out + out2)

        # 5. Convert back to dense tensor (B, N, C)
        final_feat, _ = to_dense_batch(out, q_batch, batch_size=B, max_num_nodes=N)

        return final_feat

class Agent(nn.Module):
    def __init__(self, envs, sample_obs, vision_encoder: str, num_tasks: int = 0, decoder: Optional[nn.Module] = None, use_map: bool = False, device=None, start_condition_map: bool = False, use_local_fusion: bool = False, use_rel_pos_in_fusion: bool = True, use_online_mapping: bool = False):
        super().__init__()
        self.feature_net = NatureCNN(sample_obs=sample_obs, vision_encoder=vision_encoder, num_tasks=num_tasks, decoder=decoder, use_map=use_map, device=device, start_condition_map=start_condition_map, use_local_fusion=use_local_fusion, use_rel_pos_in_fusion=use_rel_pos_in_fusion, use_online_mapping=use_online_mapping)
        # latent_size = np.array(envs.unwrapped.single_observation_space.shape).prod()
        latent_size = self.feature_net.out_features
        self.critic = nn.Sequential(
            layer_init(nn.Linear(latent_size, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, 1)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(latent_size, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, np.prod(envs.unwrapped.single_action_space.shape)), std=0.01*np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, np.prod(envs.unwrapped.single_action_space.shape)) * -0.5)

    def get_features(self, x, env_target_obj_idx=None, map_features=None):
        return self.feature_net(x, env_target_obj_idx=env_target_obj_idx, map_features=map_features)

    def get_value(self, x, env_target_obj_idx=None, map_features=None):
        x = self.get_features(x, env_target_obj_idx=env_target_obj_idx, map_features=map_features)
        return self.critic(x)

    def get_action(self, x, env_target_obj_idx=None, map_features=None, deterministic=False):
        x = self.get_features(x, env_target_obj_idx=env_target_obj_idx, map_features=map_features)
        action_mean = self.actor_mean(x)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()

    def get_action_and_value(self, x, env_target_obj_idx=None, map_features=None, action=None):
        x = self.get_features(x, env_target_obj_idx=env_target_obj_idx, map_features=map_features)
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)
