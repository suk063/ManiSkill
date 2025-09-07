for i in {50..119}
do
    python map_rl_from_example/ppo_map_eval.py \
        --env_id=PickYCBSequential-v1 \
        --control_mode=pd_joint_vel \
        --exp_name=YCB_sequential_xarm6_ppo_map_dino_EVAL_IN_DIST_IMAGE \
        --num_eval_envs=100 \
        --num_eval_steps=200 \
        --vision_encoder=dino \
        --use_map \
        --object_num=2 \
        --eval_distribution="in" \
        --checkpoint="save_checkpoint/ckpt_2181.pt" \
        --env_idx=$i
done