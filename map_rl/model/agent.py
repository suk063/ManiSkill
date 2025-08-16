import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import gymnasium as gym
from typing import Optional, List, Union
import xformers.ops as xops
from model.module import PointNet, LocalFeatureFusion, layer_init, ActionTransformerDecoder
from utils.operator import get_3d_coordinates
from model.vision_encoder import DINO2DFeatureEncoder, PlainCNNFeatureEncoder
from torchvision import transforms

transform = transforms.Compose(
    [
        transforms.Resize(size=84, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ]
) 

class FeatureExtractor(nn.Module):
    def __init__(
            self,
            sample_obs: dict,
            feature_size: int = 256,
            vision_encoder: str = "plain_cnn",
            decoder: Optional[nn.Module] = None,
            use_map: bool = True,
            use_local_fusion: bool = False,
            text_embeddings: Optional[torch.Tensor] = None,
            camera_uids: Union[str, List[str]] = "base_camera",
        ) -> None:
        super().__init__()
        
        if isinstance(camera_uids, str):
            camera_uids = [camera_uids]
        self.camera_uids = camera_uids
        self.num_cameras = len(self.camera_uids)
        # --------------------------------------------------------------- Vision
        object.__setattr__(self, "_decoder", decoder)  # None → RGB-only mode
        
        if vision_encoder == 'dino':
            self.vision_encoder = DINO2DFeatureEncoder(embed_dim=feature_size)
        elif vision_encoder == 'plain_cnn':
            self.vision_encoder = PlainCNNFeatureEncoder(embed_dim=feature_size)
        else:
            raise ValueError(f"Vision encoder {vision_encoder} not supported")
        
        # self.visual_feature_proj = nn.Linear(self.vision_encoder.embed_dim, feature_size)
        
        # --------------------------------------------------------------- Map 3-D
        self.use_map = use_map and (decoder is not None)
        self.use_local_fusion = self.use_map and use_local_fusion

        if self.use_map:
            map_raw_dim = 768  # output dim of *decoder*
            self.map_encoder = PointNet(map_raw_dim, feature_size)

            if self.use_local_fusion:
                self.map_feature_proj = nn.Linear(map_raw_dim, self.vision_encoder.embed_dim)
                self.local_fusion = LocalFeatureFusion(dim=self.vision_encoder.embed_dim, k=1, radius=0.12, num_layers=1)

        # --------------------------------------------------------------- State
        self.state_proj = None
        if "state" in sample_obs:
            state_size = sample_obs["state"].shape[-1]
            self.state_proj = nn.Linear(state_size, feature_size)

        self.text_proj = None
        if text_embeddings is not None:
            self.text_proj = nn.Linear(text_embeddings.shape[-1], feature_size)

        num_branches = 1  # state
        num_branches += self.num_cameras  # global RGB 
        if self.use_map:
            num_branches += 1  # global map
        if text_embeddings is not None:
            num_branches += 1
        self.output_dim = num_branches * feature_size

    def _local_fusion(
        self,
        observations: dict,
        image_fmap: torch.Tensor,
        coords_batch: List[torch.Tensor],
        dec_split: List[torch.Tensor],
        camera_uid: str,
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
        # pose = observations["sensor_param"]["base_camera"]["extrinsic_cv"]
        pose = observations["sensor_param"][camera_uid]["extrinsic_cv"]

        Hf = Wf = 6
        depth_s = F.interpolate(depth, size=(Hf, Wf), mode="nearest-exact")
        
        fx = fy = observations["sensor_param"][camera_uid]["intrinsic_cv"][0][0][0]
        cx = cy = observations["sensor_param"][camera_uid]["intrinsic_cv"][0][0][2]
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
        q_feat = image_fmap.permute(0, 2, 3, 1).reshape(B, -1, self.vision_encoder.embed_dim)     # (B, N, C)

        fused = self.local_fusion(q_xyz, q_feat, kv_xyz, kv_feat, pad_mask)  # (B, N, C)
        image_fmap = fused.permute(0, 2, 1).reshape(
            B, self.vision_encoder.embed_dim, Hf, Wf
        )  # (B, C, Hf, Wf)

        return image_fmap

    def forward(
        self,
        observations: dict,
        map_features: Optional[List] = None,
        env_target_obj_idx: Optional[torch.Tensor] = None,
        train_map_features: bool =False,
    ) -> Union[torch.Tensor, dict]:
    
        B = observations["rgb"].size(0)
        features: dict[str, torch.Tensor] = {}

        if self.state_proj is not None:
            features["state"] = self.state_proj(observations["state"]).unsqueeze(1)

        if self.text_proj is not None:
            assert env_target_obj_idx is not None, "env_target_obj_idx must be provided when using text embeddings"
            target_text_embeddings = self.text_embeddings[env_target_obj_idx]
            features["text"] = self.text_proj(target_text_embeddings).unsqueeze(1)

        # 1. Process multiple camera inputs by batching them
        rgb_all_cameras = observations["rgb"].float().permute(0, 3, 1, 2) / 255.0
        rgb_per_camera = torch.chunk(rgb_all_cameras, self.num_cameras, dim=1)
        image_batch = torch.cat(rgb_per_camera, dim=0)

        image = transform(image_batch)
        image_fmap = self.vision_encoder(image) # (B * num_cameras, C, Hf, Wf)

        if not self.use_map:
            B_cam, C, H, W = image_fmap.shape
            image_tokens = image_fmap.flatten(2).permute(0, 2, 1) # (B*num_cam, H*W, C)
            # projected_tokens = self.visual_feature_proj(image_tokens) # (B*num_cam, 36, 256)
            
            # Reshape to (B, num_cameras * 36, 256)
            projected_tokens = image_tokens.reshape(B, self.num_cameras * H * W, -1)
            features["visual"] = projected_tokens
            
            return features

        assert map_features is not None, "map_features must be provided when use_map=True"
        
        # 2. Process map features once per environment (batch size B)
        coords_batch, raw_batch = [], []
        for g in map_features:
            coords = g.levels[1].coords
            coords_batch.append(coords)
            raw_batch.append(g.query_voxel_feature(coords))

        with torch.no_grad():
            dec_cat = self._decoder(torch.cat(raw_batch, dim=0))          # (ΣL, 768)

        dec_split = dec_cat.split([c.size(0) for c in coords_batch], dim=0)

        lengths = [c.size(0) for c in coords_batch]
        pad_3d = torch.nn.utils.rnn.pad_sequence(dec_split, batch_first=True)  # (B, Lmax, 768)
        Lmax = pad_3d.size(1)

        pad_mask = torch.arange(Lmax, device=pad_3d.device).expand(len(lengths), -1) >= \
                torch.tensor(lengths, device=pad_3d.device)[:, None]
        if train_map_features:
            map_vec = self.map_encoder(pad_3d, pad_mask)
        else:
            with torch.no_grad():
                map_vec = self.map_encoder(pad_3d, pad_mask)
            
        features["map"] = map_vec.unsqueeze(1)
        
        if self.use_local_fusion and train_map_features:
            # Fuse each camera's image_fmap with the same map features
            fused_image_fmaps = []
            image_fmaps_per_cam = torch.chunk(image_fmap, self.num_cameras, dim=0)
            depth_all_cameras = observations["depth"]
            depth_per_camera = torch.chunk(depth_all_cameras, self.num_cameras, dim=-1)

            for i, cam_uid in enumerate(self.camera_uids):
                # Create a temporary observation dict for _local_fusion
                obs_cam = {
                    "depth": depth_per_camera[i],
                    "sensor_param": {cam_uid: observations["sensor_param"][cam_uid]}
                }
                fused = self._local_fusion(
                    observations=obs_cam,
                    image_fmap=image_fmaps_per_cam[i],
                    coords_batch=coords_batch,
                    dec_split=dec_split,
                    camera_uid=cam_uid,
                )
                fused_image_fmaps.append(fused)
            image_fmap = torch.cat(fused_image_fmaps, dim=0)
        
        B_cam, C, H, W = image_fmap.shape
        image_tokens = image_fmap.flatten(2).permute(0, 2, 1) # (B*num_cam, H*W, C)
        # projected_tokens = self.visual_feature_proj(image_tokens) # (B*num_cam, 36, 256)
        
        # Reshape to (B, num_cameras * 36, 256)
        projected_tokens = image_tokens.reshape(B, self.num_cameras * H * W, -1)
        features["visual"] = projected_tokens

        return features

class Agent(nn.Module):
    """Actor-Critic network (Gaussian continuous actions) built on NatureCNN features."""
    def __init__(
        self, 
        envs, 
        sample_obs, 
        decoder: Optional[nn.Module] = None, 
        use_map: bool = True, 
        use_local_fusion: bool = False, 
        vision_encoder: str = "plain_cnn",
        text_embeddings: Optional[torch.Tensor] = None,
        camera_uids: Union[str, List[str]] = "base_camera",
    ):
        super().__init__()
        if text_embeddings is not None:
            self.register_buffer("text_embeddings", text_embeddings)
        else:
            self.text_embeddings = None

        self.feature_net = FeatureExtractor(
            sample_obs=sample_obs, 
            decoder=decoder, 
            use_map=use_map, 
            use_local_fusion=use_local_fusion, 
            vision_encoder=vision_encoder,
            text_embeddings=self.text_embeddings,
            camera_uids=camera_uids,
        )
        latent_size = self.feature_net.output_dim
        
        self.actor_critic = ActionTransformerDecoder(
            d_model=256,
            transf_input_dim=256,
            nhead=8,
            num_decoder_layers=4,
            dim_feedforward=1024,
            dropout=0.1,
            action_dim=int(np.prod(envs.unwrapped.single_action_space.shape)),
            action_pred_horizon=1
        )
        
        # Log-std is state-independent
        self.actor_logstd = nn.Parameter(
            torch.ones(1, int(np.prod(envs.unwrapped.single_action_space.shape))) * -0.5
        )
        self.feature_net.text_embeddings = self.text_embeddings

    # Convenience helpers -----------------------------------------------------

    def get_features(self, x, map_features=None, env_target_obj_idx=None, train_map_features: bool = True):
        return self.feature_net(x, map_features=map_features, env_target_obj_idx=env_target_obj_idx, train_map_features=train_map_features)

    def get_value(self, x, map_features=None, env_target_obj_idx=None, train_map_features: bool = True):
        features = self.get_features(x, map_features=map_features, env_target_obj_idx=env_target_obj_idx, train_map_features=train_map_features)
        _, value = self.actor_critic(features)
        return value


    def get_action(self, x, map_features=None, env_target_obj_idx=None, deterministic: bool = False, train_map_features: bool = True):
        features = self.get_features(x, map_features=map_features, env_target_obj_idx=env_target_obj_idx, train_map_features=train_map_features)
        action_mean, _ = self.actor_critic(features)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()

    def get_action_and_value(self, x, map_features=None, env_target_obj_idx=None, action=None, train_map_features: bool = True):
        features = self.get_features(x, map_features=map_features, env_target_obj_idx=env_target_obj_idx, train_map_features=train_map_features)
        action_mean, value = self.actor_critic(features)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            value,
        )