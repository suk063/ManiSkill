#!/usr/bin/env bash
set -euo pipefail

# Run all six configurations once, sequentially, with distinct wandb tags
# Project: PPO-RL-Map

cd "$(dirname "$0")/.."

COMMON_ARGS=(
  --env_id=PickYCBSequential-v1 # PickYCBCustom-v1
  --robot_uids=xarm6_robotiq
  --control_mode=pd_joint_vel
  --num_envs=12
  --num_eval_envs=12
  --eval_freq=20
  --total_timesteps=100_000_000
  --num_steps=500
  --num_eval_steps=500
  --gamma=0.99
  --gae_lambda=0.95
  --update_epochs=8
  --learning_rate=3e-4
  --capture-video
  --track
  --wandb_project_name "PPO-RL-Map"
)

run_cfg() {
  local TAG="$1"; shift
  echo "=== Running: ${TAG} ==="
  python map_rl/train_ppo.py \
    "${COMMON_ARGS[@]}" \
    --exp_name=PickYCB_xarm6_ppo__${TAG} \
    --wandb_tags ${TAG} \
    "$@"
}

run_cfg cnn-map-local-fusion-base-and-hand-cam-priviledged-state  \
  --use_map \
  --use_local_fusion \
  --vision_encoder=plain_cnn \
  --map_start_iteration=10000000 \
  # --camera_uids=base_camera,hand_camera \
  # --checkpoint=pretrained/ckpt_latest.pt \