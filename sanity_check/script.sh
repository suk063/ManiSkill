# python sanity_check/ppo_state.py \
#     --env_id=PickYCBSequential-v1 \
#     --control_mode=pd_joint_vel \
#     --exp_name=YCB_sequential_xarm6_ppo_state \
#     --num_envs=32 \
#     --num_eval_envs=20 \
#     --eval_freq=20 \
#     --total_timesteps=100_000_000 \
#     --num_steps=400 \
#     --num_eval_steps=400 \
#     --gamma=0.9 \
#     --learning_rate=3e-4 \
#     --capture-video \
#     --track \
#     --wandb_project_name "PPO-RL-Map" \
#     --checkpoint "runs/YCB_sequential_xarm6_ppo_state/ckpt_41.pt"


python sanity_check/ppo_rgb.py \
    --env_id=PickYCBSequential-v1 \
    --control_mode=pd_joint_vel \
    --exp_name=YCB_sequential_xarm6_ppo_rgb \
    --num_envs=32 \
    --num_eval_envs=20 \
    --eval_freq=20 \
    --total_timesteps=100_000_000 \
    --num_steps=300 \
    --num_eval_steps=300 \
    --gamma=0.95 \
    --learning_rate=3e-4 \
    --capture-video \
    --track \
    --wandb_project_name "PPO-RL-Map" \
    --checkpoint "pretrained/image_based.pt"