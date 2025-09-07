# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from collections import defaultdict
import os
import random
import time
from dataclasses import dataclass, field
from typing import Optional, List

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import tyro
from torch.utils.tensorboard import SummaryWriter

# ManiSkill specific imports
import mani_skill.envs
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

# Mapping-related imports
from mapping.mapping_lib.voxel_hash_table import VoxelHashTable
from mapping.mapping_lib.implicit_decoder import ImplicitDecoder

from map_rl_from_example.model import Agent
from map_rl_from_example.utils import DictArray, GridSampler, Logger


@dataclass
class Args:
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    
    checkpoint: str = ""
    """path to a pretrained checkpoint file to start evaluation/training from"""
    render_mode: str = "all"
    """the environment rendering mode"""

    # Algorithm specific arguments
    env_id: str = "PickCube-v1"
    """the id of the environment"""
    include_state: bool = True
    """whether to include state information in observations"""
    num_eval_envs: int = 20
    """the number of parallel evaluation environments"""
    eval_partial_reset: bool = False
    """whether to let parallel evaluation environments reset upon termination instead of truncation"""
    num_eval_steps: int = 200
    """the number of steps to run in each evaluation environment during evaluation"""
    eval_reconfiguration_freq: Optional[int] = None
    """for benchmarking purposes we want to reconfigure the eval environment each reset to ensure objects are randomized in some tasks"""
    control_mode: Optional[str] = "pd_joint_delta_pos"
    """the control mode to use for the environment"""
    total_envs: int = 120
    """Total number of discrete environments available for sampling with global_idx"""
    sampling_envs: Optional[int] = 100
    """Number of environments to sample from for training. If None, defaults to total_envs."""
    # task specification
    model_ids: List[str] = field(default_factory=lambda: ["013_apple", "014_lemon", "017_orange", "012_strawberry", "011_banana"])
    """the list of model ids to use for the environment"""
    object_num: int = 2
    """the number of target objects to use for the environment"""
    
    # Map-related arguments
    use_map: bool = False
    """if toggled, use the pre-trained environment map features as part of the observation"""
    start_condition_map: bool = False
    """If toggled, the map conditioning gate becomes learnable."""
    use_local_fusion: bool = False
    """If toggled, use local feature fusion."""
    use_rel_pos_in_fusion: bool = False
    """If toggled, use relative positional encoding in local feature fusion."""
    vision_encoder: str = "dino" # "plain_cnn" or "dino"
    """the vision encoder to use for the agent"""
    map_dir: str = "mapping/multi_env_maps_custom"
    """Directory where the trained environment maps are stored."""
    decoder_path: str = "mapping/multi_env_maps_custom/shared_decoder.pt"
    """Path to the trained shared decoder model."""
    load_actor_logstd: bool = False
    """if toggled, actor_logstd weights will be loaded from the checkpoint"""

    # Evaluation-specific arguments
    evaluate: bool = True
    """if toggled, only runs evaluation with the given model checkpoint and saves the evaluation trajectories"""
    eval_distribution: str = "in"
    """the distribution of environments to evaluate on ('in' or 'out')"""
    env_idx: Optional[int] = None
    """if specified, evaluates on a single environment with this index"""
    
    # to be filled in runtime
    num_tasks: int = 0
    """the number of tasks (computed in runtime)"""


if __name__ == "__main__":
    args = tyro.cli(Args)
    assert args.checkpoint, "Checkpoint must be provided for evaluation."
    assert args.evaluate, "This script is for evaluation only, --evaluate must be True."
    
    args.num_tasks = len(args.model_ids)
    if args.sampling_envs is None:
        args.sampling_envs = args.total_envs
    
    if args.env_idx is not None:
        run_name = f"eval_{args.env_id}_env_{args.env_idx}_{int(time.time())}"
    else:
        run_name = f"eval_{args.env_id}_{args.eval_distribution}_dist_{int(time.time())}"
    if args.exp_name is not None:
        run_name = args.exp_name

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # --- Load Maps and Decoder ---
    all_grids, decoder = None, None
    if args.use_map:
        print("--- Loading maps and decoder for evaluation ---")
        try:
            decoder = ImplicitDecoder(
                voxel_feature_dim=64 * 2, # GRID_FEAT_DIM * GRID_LVLS from map_multi_table.py
                hidden_dim=240,
                output_dim=768,
            ).to(device)
            decoder.load_state_dict(torch.load(args.decoder_path, map_location=device))
            decoder.eval()
            for param in decoder.parameters():
                param.requires_grad = False
            print(f"Loaded shared decoder from {args.decoder_path}")
        except FileNotFoundError:
            print(f"[ERROR] Decoder file not found at {args.decoder_path}. Exiting.")
            sys.exit(1)

        all_grids = []
        if not os.path.exists(args.map_dir):
            print(f"[ERROR] Map directory not found: {args.map_dir}. Exiting.")
            sys.exit(1)

        if args.env_idx is not None:
            num_maps_to_load = args.total_envs
        else:
            num_maps_to_load = args.total_envs if args.eval_distribution == "out" else args.sampling_envs
        print(f"Loading {num_maps_to_load} maps from {args.map_dir} ...")
        for i in range(num_maps_to_load):
            grid_path = os.path.join(args.map_dir, f"env_{i:03d}_grid.sparse.pt")
            if not os.path.exists(grid_path):
                print(f"[ERROR] Map file not found: {grid_path}. Exiting.")
                sys.exit(1)
            all_grids.append(VoxelHashTable.load_sparse(grid_path, device=device))
        print(f"--- Loaded {len(all_grids)} maps. ---")

        for grid in all_grids:
            for p in grid.parameters():
                p.requires_grad = False

    # env setup
    if args.use_local_fusion:
        env_kwargs = dict(obs_mode="rgb+depth", render_mode=args.render_mode, sim_backend="physx_cuda")
    else:
        env_kwargs = dict(obs_mode="rgb", render_mode=args.render_mode, sim_backend="physx_cuda")
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode
    env_kwargs["object_num"] = args.object_num
    
    # Set up evaluation indices BEFORE creating the environment
    if args.env_idx is not None:
        print(f"Evaluating on a single environment with index {args.env_idx}")
        args.num_eval_envs = 1
        eval_indices = np.array([args.env_idx])
    else:
        rng_eval = np.random.RandomState(args.seed)
        if args.eval_distribution == "in":
            print(f"Evaluating on in-distribution environments (0-{args.sampling_envs - 1})")
            all_indices = np.arange(args.sampling_envs)
            rng_eval.shuffle(all_indices)
            eval_indices = np.sort(all_indices[:args.num_eval_envs])
        elif args.eval_distribution == "out":
            ood_start_idx = args.sampling_envs
            ood_end_idx = args.total_envs
            num_ood_envs = ood_end_idx - ood_start_idx
            print(f"Evaluating on out-of-distribution environments ({ood_start_idx}-{ood_end_idx - 1})")
            if args.num_eval_envs > num_ood_envs:
                print(f"[WARNING] num_eval_envs ({args.num_eval_envs}) > num_ood_envs ({num_ood_envs}). Using all OOD envs.")
                args.num_eval_envs = num_ood_envs
            
            all_indices = np.arange(ood_start_idx, ood_end_idx)
            rng_eval.shuffle(all_indices)
            eval_indices = np.sort(all_indices[:args.num_eval_envs])
        else:
            raise ValueError(f"Invalid eval_distribution: {args.eval_distribution}")

    eval_env_kwargs = env_kwargs.copy()
    eval_env_kwargs["is_eval"] = True
    eval_envs = gym.make(args.env_id, num_envs=args.num_eval_envs, reconfiguration_freq=args.eval_reconfiguration_freq, **eval_env_kwargs)

    # rgbd obs mode returns a dict of data, we flatten it so there is just a rgbd key and state key
    eval_envs = FlattenRGBDObservationWrapper(eval_envs, rgb=True, depth=True, state=args.include_state, include_camera_params=True)

    if isinstance(eval_envs.action_space, gym.spaces.Dict):
        eval_envs = FlattenActionSpaceWrapper(eval_envs)

    if args.capture_video:
        video_dir = "eval_video"
        eval_output_dir = f"{video_dir}/{run_name}"
        print(f"Saving eval videos to {eval_output_dir}")
        eval_envs = RecordEpisode(eval_envs, output_dir=eval_output_dir, save_trajectory=args.evaluate, trajectory_name="trajectory", max_steps_per_video=args.num_eval_steps, video_fps=30)

    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=not args.eval_partial_reset, record_metrics=True)
    assert isinstance(eval_envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    eval_grids = None
    if args.use_map:
        eval_grids = [all_grids[i] for i in eval_indices]

    # Dummy env for agent creation
    dummy_gym_env = gym.make(args.env_id, num_envs=1, **env_kwargs)
    dummy_gym_env = FlattenRGBDObservationWrapper(dummy_gym_env, rgb=True, depth=True, state=args.include_state, include_camera_params=True)
    dummy_env = ManiSkillVectorEnv(dummy_gym_env, 1)
    dummy_obs, _ = dummy_env.reset()

    agent = Agent(dummy_env, sample_obs=dummy_obs, vision_encoder=args.vision_encoder, num_tasks=args.num_tasks, decoder=decoder, use_map=args.use_map, device=device, start_condition_map=args.start_condition_map, use_local_fusion=args.use_local_fusion, use_rel_pos_in_fusion=args.use_rel_pos_in_fusion).to(device)
    dummy_env.close()

    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint_state_dict = torch.load(args.checkpoint, map_location=device)

    if not args.load_actor_logstd:
        if 'actor_logstd' in checkpoint_state_dict:
            del checkpoint_state_dict['actor_logstd']
            print("--- Excluded 'actor_logstd' from checkpoint loading ---")

    model_state_dict = agent.state_dict()
    
    # Find mismatched keys
    mismatched_keys = []
    checkpoint_keys = checkpoint_state_dict.keys()
    model_keys = model_state_dict.keys()

    for k in checkpoint_keys:
        if k in model_keys:
            if checkpoint_state_dict[k].shape != model_state_dict[k].shape:
                mismatched_keys.append(
                    (k, checkpoint_state_dict[k].shape, model_state_dict[k].shape)
                )
    
    missing_keys = [k for k in model_keys if k not in checkpoint_keys and not k.startswith('feature_net.vision_encoder.backbone.')]
    if not args.load_actor_logstd:
        # If we are not loading actor_logstd, it's okay for it to be missing from the checkpoint.
        if 'actor_logstd' in missing_keys:
            missing_keys.remove('actor_logstd')

    unexpected_keys = [k for k in checkpoint_keys if k not in model_keys]

    if mismatched_keys or missing_keys or unexpected_keys:
        print("[ERROR] Checkpoint loading failed due to mismatches.")
        if mismatched_keys:
            print("\nSize mismatches:")
            for key, ckpt_shape, model_shape in mismatched_keys:
                print(f"  - {key}: Checkpoint has {ckpt_shape}, model needs {model_shape}")
        if missing_keys:
            print("\nMissing keys in checkpoint:")
            for key in missing_keys:
                print(f"  - {key}")
        if unexpected_keys:
            print("\nUnexpected keys in checkpoint:")
            for key in unexpected_keys:
                print(f"  - {key}")
        sys.exit(1)

    agent.load_state_dict(checkpoint_state_dict, strict=False)
    agent.eval()

    print("--- Starting Evaluation ---")
    stime = time.perf_counter()
    eval_obs, eval_infos = eval_envs.reset(seed=args.seed, options={"global_idx": eval_indices.tolist()})
    eval_metrics = defaultdict(list)
    num_episodes = 0

    for _ in range(args.num_eval_steps):
        with torch.no_grad():
            eval_target_obj_idx = torch.where(
                eval_infos['success_obj_1'], 
                eval_infos['env_target_obj_idx_2'], 
                eval_infos['env_target_obj_idx_1']
            )
            action = agent.get_action(eval_obs, env_target_obj_idx=eval_target_obj_idx, map_features=eval_grids, deterministic=True)
            eval_obs, eval_rew, eval_terminations, eval_truncations, eval_infos = eval_envs.step(action)
            if "final_info" in eval_infos:
                mask = eval_infos["_final_info"]
                if mask.any():
                    num_episodes += mask.sum().item()
                    for k, v in eval_infos["final_info"]["episode"].items():
                        eval_metrics[k].extend(v[mask].cpu().numpy())
                    if "robot_cumulative_force" in eval_infos["final_info"]:
                        eval_metrics["robot_cumulative_force"].extend(eval_infos["final_info"]["robot_cumulative_force"][mask].cpu().numpy())

    print(f"Evaluated {args.num_eval_steps * args.num_eval_envs} steps resulting in {num_episodes} episodes")
    
    print("--- Evaluation Results ---")
    for k, v in eval_metrics.items():
        mean_val = np.mean(v)
        std_val = np.std(v)
        print(f"eval/{k}: {mean_val:.4f} +/- {std_val:.4f}")

    eval_time = time.perf_counter() - stime
    print(f"Total evaluation time: {eval_time:.2f}s")
    
    eval_envs.close()

    if args.env_idx is not None and args.capture_video:
        eval_output_dir = f"eval_video/{run_name}"
        video_path = os.path.join(eval_output_dir, "0.mp4")
        if os.path.exists(video_path):
            # As we are evaluating a single environment, success_once should have a single value.
            success_once = eval_metrics.get('success_once', [False])[0]
            new_video_path = os.path.join(eval_output_dir, f"{args.env_idx}.mp4")
            os.rename(video_path, new_video_path)
            print(f"Renamed video to {new_video_path} (success_once={success_once})")

    print("--- Evaluation Finished ---")
