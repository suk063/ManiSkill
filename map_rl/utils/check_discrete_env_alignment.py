import argparse
import numpy as np
import torch
import gymnasium as gym
import mani_skill.envs  # Register ManiSkill environments

"""Check whether PickYCBSequential and PickYCBCustomNoRobot
return consistent YCB object positions for the same env indices.

Usage
-----
python -m map_rl.utils.check_discrete_env_alignment --indices 0 65 128 191
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Check YCB env alignment (sequential vs no-robot)")
    parser.add_argument(
        "--indices",
        type=int,
        nargs="*",
        default=[0, 65, 128, 191],
        help="List of env indices (0-based) to test; valid range depends on env (e.g., 0..199)",
    )
    parser.add_argument(
        "--tol", type=float, default=1e-5, help="Tolerance for position mismatch (m)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    indices = sorted(set(int(i) for i in args.indices))
    if len(indices) == 0:
        print("No indices provided.")
        return

    max_idx = int(max(indices))

    # ------------------------------------------------------------------
    # Create both environments with enough parallel envs to cover max_idx
    # The sequential env uses env_idx to lay out YCB objects, so we must
    # allocate up to max_idx + 1 sub-environments and selectively reset.
    # ------------------------------------------------------------------
    env_seq = gym.make(
        "PickYCBSequential-v1",
        num_envs=200,
        robot_uids="xarm6_robotiq",
        obs_mode="state",
        render_mode="none",
    )

    env_nr = gym.make(
        "PickYCBCustomNoRobot-v1",
        num_envs=200,
        obs_mode="state",
        render_mode="none",
    )

    # Initial build/reset for all envs
    env_seq.reset()
    env_nr.reset()

    ok_all = True
    for idx in indices:
        # Reset only the sub-env at env_idx == idx for both envs, placing the
        # index tensor on each env's device to avoid CPU/GPU mismatches.
        idx_tensor_seq = torch.as_tensor([idx], dtype=torch.long, device=env_seq.unwrapped.device)
        idx_tensor_nr = torch.as_tensor([idx], dtype=torch.long, device=env_nr.unwrapped.device)
        env_seq.reset(options={"env_idx": idx_tensor_seq})
        env_nr.reset(options={"env_idx": idx_tensor_nr})

        # Compare positions of each YCB object type (5 types)
        per_object_ok = []
        per_object_diffs = []
        for obj_i in range(len(env_seq.unwrapped.ycb_objects)):
            p_seq = env_seq.unwrapped.ycb_objects[obj_i].pose.p[idx]
            p_nr = env_nr.unwrapped.ycb_objects[obj_i].pose.p[idx]
            diff = torch.linalg.norm(p_seq - p_nr).item()
            per_object_diffs.append(diff)
            per_object_ok.append(diff < args.tol)

        # Also compare the two target pick objects
        p1_seq = env_seq.unwrapped.pick_obj_1.pose.p[idx]
        p1_nr = env_nr.unwrapped.pick_obj_1.pose.p[idx]
        diff_p1 = torch.linalg.norm(p1_seq - p1_nr).item()

        p2_seq = env_seq.unwrapped.pick_obj_2.pose.p[idx]
        p2_nr = env_nr.unwrapped.pick_obj_2.pose.p[idx]
        diff_p2 = torch.linalg.norm(p2_seq - p2_nr).item()

        objs_ok = all(per_object_ok)
        picks_ok = (diff_p1 < args.tol) and (diff_p2 < args.tol)
        same = objs_ok and picks_ok
        ok_all &= same

        status = "OK" if same else "MISMATCH"
        details = ", ".join([f"ycb[{i}]:{d:.6f}" for i, d in enumerate(per_object_diffs)])
        print(
            f"Idx {idx:3d}: {status} | {details}, pick1:{diff_p1:.6f}, pick2:{diff_p2:.6f}"
        )

    env_seq.close()
    env_nr.close()

    if ok_all:
        print("\nAll tested indices consistent ✅")
    else:
        print("\nSome indices mismatched ❌")


if __name__ == "__main__":
    main()
