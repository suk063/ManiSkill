import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from torch.cuda.amp import autocast
from torch.distributions.normal import Normal
from torchvision import transforms

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

        B, _, H, W = images_bchw.shape
        images_bchw = self.transform(images_bchw)
        
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
    def __init__(self, sample_obs, vision_encoder: str, num_tasks: int = 0, decoder: Optional[nn.Module] = None, use_map: bool = False, device=None, start_condition_map: bool = False):
        super().__init__()
        self.vision_encoder_name = vision_encoder
        self.vision_encoder = None
        extractors = {}

        self.out_features = 0
        feature_size = 256
        in_channels=sample_obs["rgb"].shape[-1]
        image_size=(sample_obs["rgb"].shape[1], sample_obs["rgb"].shape[2])


        # here we use a NatureCNN architecture to process images, but any architecture is permissble here
        if self.vision_encoder_name == "dino":
            self.vision_encoder = DINO2DFeatureEncoder(embed_dim=64)
            cnn = nn.Sequential(
                self.vision_encoder,
                nn.Flatten(),
            )

            # to easily figure out the dimensions after flattening, we pass a test tensor
            with torch.no_grad():
                sample_input = sample_obs["rgb"].float().permute(0,3,1,2) / 255.0

                # Temporarily move to the target device for shape inference to avoid xformers error on CPU
                if device and "cuda" in str(device):
                    cnn_tmp = cnn.to(device)
                    sample_tmp = sample_input.to(device)
                    with autocast():
                        n_flatten = cnn_tmp(sample_tmp).shape[1]
                    # The cnn object itself is not moved, so it stays on CPU.
                else:
                    # This will raise the xformers error if not on CUDA, which is expected.
                    n_flatten = cnn(sample_input.cpu()).shape[1]
                
                fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
        elif self.vision_encoder_name == "plain_cnn":
            cnn = nn.Sequential(
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
                    in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0
                ),
                nn.ReLU(),
                nn.Flatten(),
            )
            # to easily figure out the dimensions after flattening, we pass a test tensor
            with torch.no_grad():
                n_flatten = cnn(sample_obs["rgb"].float().permute(0,3,1,2).cpu() / 255.0).shape[1]
                fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
        else:
            raise ValueError(f"Unknown vision_encoder: {self.vision_encoder_name}")
        extractors["rgb"] = nn.Sequential(cnn, fc)
        self.out_features += feature_size

        if "state" in sample_obs:
            # for state data we simply pass it through a single linear layer
            state_size = sample_obs["state"].shape[-1]
            extractors["state"] = nn.Linear(state_size, 256)
            self.out_features += 256

        self.task_embedding = None
        if num_tasks > 0:
            self.task_embedding = nn.Embedding(num_tasks, feature_size)
            self.out_features += feature_size

        self.extractors = nn.ModuleDict(extractors)

        # Map-related components
        object.__setattr__(self, "_decoder", decoder)
        self.use_map = use_map and (decoder is not None)
        self.start_condition_map = start_condition_map
        if self.use_map:
            map_raw_dim = 768  # output dim of *decoder*
            self.map_encoder = PointNet(map_raw_dim, feature_size)
            self.out_features += feature_size
            self.map_gate = nn.Parameter(torch.tensor(0.0))
            if not self.start_condition_map:
                self.map_gate.requires_grad = False


    def forward(self, observations, env_target_obj_idx=None, map_features=None) -> torch.Tensor:
        encoded_tensor_list = []
        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            obs = observations[key]
            if key == "rgb":
                obs = obs.float().permute(0,3,1,2)
                obs = obs / 255.0
            encoded_tensor_list.append(extractor(obs))
        
        if self.task_embedding is not None:
            assert env_target_obj_idx is not None, "env_target_obj_idx must be provided for task embedding"
            task_emb = self.task_embedding(env_target_obj_idx)
            encoded_tensor_list.append(task_emb)

        if self.use_map:
            if self.start_condition_map:
                assert map_features is not None, "map_features must be provided when use_map=True and start_condition_map=True"
                
                coords_batch, raw_batch = [], []
                for g in map_features:
                    coords = g.levels[1].coords
                    coords_batch.append(coords)
                    raw_batch.append(g.query_voxel_feature(coords))

                # The decoder is pre-trained and used in inference mode.
                with torch.no_grad():
                    dec_cat = self._decoder(torch.cat(raw_batch, dim=0))
                
                dec_split = dec_cat.split([c.size(0) for c in coords_batch], dim=0)
                
                lengths = [c.size(0) for c in coords_batch]
                pad_3d = torch.nn.utils.rnn.pad_sequence(dec_split, batch_first=True)
                coords_padded = torch.nn.utils.rnn.pad_sequence(coords_batch, batch_first=True)
                Lmax = pad_3d.size(1)

                pad_mask = torch.arange(Lmax, device=pad_3d.device).expand(len(lengths), -1) >= \
                        torch.tensor(lengths, device=pad_3d.device)[:, None]
                
                map_vec = self.map_encoder(pad_3d, coords_padded, pad_mask)
                encoded_tensor_list.append(self.map_gate * map_vec)
            else:
                # Get feature size from the map encoder's last layer
                map_feature_size = self.map_encoder.net[-1].out_features
                # Get batch size from any observation tensor
                some_obs_tensor = next(iter(observations.values()))
                batch_size = some_obs_tensor.shape[0]
                device = some_obs_tensor.device
                zero_map_vec = torch.zeros(batch_size, map_feature_size, device=device)
                encoded_tensor_list.append(zero_map_vec)

        return torch.cat(encoded_tensor_list, dim=1)

class Agent(nn.Module):
    def __init__(self, envs, sample_obs, vision_encoder: str, num_tasks: int = 0, decoder: Optional[nn.Module] = None, use_map: bool = False, device=None, start_condition_map: bool = False):
        super().__init__()
        self.feature_net = NatureCNN(sample_obs=sample_obs, vision_encoder=vision_encoder, num_tasks=num_tasks, decoder=decoder, use_map=use_map, device=device, start_condition_map=start_condition_map)
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
