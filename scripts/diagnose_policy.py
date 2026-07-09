#!/usr/bin/env python3
"""Read-only policy diagnostic for the SO101 EE deployment.

Sets up the SAME context as deploy_ee_real_robot.py (robot, cameras, policy,
FK/IK processors, action-key workaround) but instead of the control loop it
reads the observation, computes the EE state, and runs inference WITHOUT ever
sending an action to the motors. Prints the EE observation and the predicted
action chunk, and flags anything out of the training distribution.

Run exactly like the deploy command but with this script name, e.g.:

    sudo -E PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/diagnose_policy.py \
        --policy-path outputs/train/smolvla_so101_cube_to_drawer/checkpoints/last/pretrained_model \
        --task "Pick up the black cube on top of the blue box and place it inside the drawer" \
        --follower-port /dev/tty.usbmodem5B140303851 --wrist-cam 0 --agent-cam-serial 234222301106 \
        --steps 8
"""

import argparse
import os

import numpy as np
import torch
from safetensors.torch import load_file

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--policy-path", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--device", default=None)
    p.add_argument("--follower-port", default="/dev/tty.usbmodem5B140303851")
    p.add_argument("--follower-id", default="follower_arm")
    p.add_argument("--wrist-cam", default="0")
    p.add_argument("--agent-cam-serial", default=None)
    p.add_argument("--agent-cam", default="/dev/video8")
    p.add_argument("--cam-width", type=int, default=640)
    p.add_argument("--cam-height", type=int, default=480)
    p.add_argument("--urdf", default=os.path.join(ROOT_DIR, "SO101", "so101_new_calib.urdf"))
    p.add_argument("--steps", type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()

    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.cameras.realsense import RealSenseCameraConfig
    from lerobot.configs import PreTrainedConfig
    from lerobot.utils.feature_utils import build_dataset_frame
    from lerobot.model.kinematics import RobotKinematics
    from lerobot.processor import (
        RobotProcessorPipeline,
        observation_to_transition,
        robot_action_observation_to_transition,
        transition_to_observation,
        transition_to_robot_action,
    )
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    from lerobot.robots.so_follower.robot_kinematic_processor import (
        ForwardKinematicsJointsToEE,
        InverseKinematicsEEToJoints,
    )
    from lerobot.rollout import BaseStrategyConfig, RolloutConfig, build_rollout_context
    from lerobot.rollout.inference import SyncInferenceConfig
    from lerobot.rollout.strategies import BaseStrategy
    from lerobot.types import RobotAction, RobotObservation
    from lerobot.utils.constants import ACTION, OBS_STR
    from lerobot.utils.device_utils import auto_select_torch_device, is_torch_device_available
    from lerobot.utils.process import ProcessSignalHandler
    from lerobot.utils.utils import init_logging

    init_logging()

    wrist_index = int(args.wrist_cam) if str(args.wrist_cam).isdigit() else args.wrist_cam
    if args.agent_cam_serial:
        agent_view_config = RealSenseCameraConfig(
            serial_number_or_name=args.agent_cam_serial,
            width=args.cam_width, height=args.cam_height, fps=args.fps,
        )
    else:
        agent_view_config = OpenCVCameraConfig(
            index_or_path=int(args.agent_cam) if str(args.agent_cam).isdigit() else args.agent_cam,
            width=args.cam_width, height=args.cam_height, fps=args.fps, fourcc="MJPG",
        )
    camera_config = {
        "wrist": OpenCVCameraConfig(
            index_or_path=wrist_index, width=args.cam_width, height=args.cam_height, fps=args.fps, fourcc="MJPG",
        ),
        "agent_view": agent_view_config,
    }
    robot_config = SO101FollowerConfig(
        port=args.follower_port, id=args.follower_id, cameras=camera_config, use_degrees=True,
    )

    temp_robot = SO101Follower(robot_config)
    motor_names = list(temp_robot.bus.motors.keys())
    kinematics = RobotKinematics(urdf_path=args.urdf, target_frame_name="gripper_frame_link", joint_names=motor_names)

    robot_observation_processor = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[ForwardKinematicsJointsToEE(kinematics=kinematics, motor_names=motor_names)],
        to_transition=observation_to_transition, to_output=transition_to_observation,
    )
    teleop_action_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[ForwardKinematicsJointsToEE(kinematics=kinematics, motor_names=motor_names)],
        to_transition=robot_action_observation_to_transition, to_output=transition_to_robot_action,
    )
    robot_action_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[InverseKinematicsEEToJoints(kinematics=kinematics, motor_names=motor_names, initial_guess_current_joints=True)],
        to_transition=robot_action_observation_to_transition, to_output=transition_to_robot_action,
    )

    policy_config = PreTrainedConfig.from_pretrained(args.policy_path)
    policy_config.pretrained_path = args.policy_path
    device = args.device
    if device is None or not is_torch_device_available(device):
        device = auto_select_torch_device().type
    policy_config.device = device

    cfg = RolloutConfig(
        robot=robot_config, policy=policy_config, strategy=BaseStrategyConfig(),
        inference=SyncInferenceConfig(), fps=args.fps, duration=args.steps + 5, task=args.task, device=device,
        return_to_initial_position=False,
    )

    # Load training stats for an in-distribution check.
    stats = load_file(os.path.join(args.policy_path, "policy_preprocessor_step_5_normalizer_processor.safetensors"))
    st_min = stats["observation.state.min"].flatten().numpy()
    st_max = stats["observation.state.max"].flatten().numpy()
    ac_min = stats.get("action.min")
    ac_max = stats.get("action.max")
    ac_min = ac_min.flatten().numpy() if ac_min is not None else None
    ac_max = ac_max.flatten().numpy() if ac_max is not None else None
    labels = ["ee.x", "ee.y", "ee.z", "ee.wx", "ee.wy", "ee.wz", "grip"]

    signal_handler = ProcessSignalHandler(use_threads=True)
    ctx = build_rollout_context(
        cfg, signal_handler.shutdown_event,
        teleop_action_processor=teleop_action_processor,
        robot_action_processor=robot_action_processor,
        robot_observation_processor=robot_observation_processor,
    )

    correct_action_keys = list(ctx.data.dataset_features[ACTION]["names"])
    ctx.data.ordered_action_keys[:] = correct_action_keys
    ctx.policy.inference._ordered_action_keys = correct_action_keys
    print("\nordered_action_keys:", correct_action_keys)
    print("state.min :", [round(float(x), 3) for x in st_min])
    print("state.max :", [round(float(x), 3) for x in st_max])

    robot = ctx.hardware.robot_wrapper
    engine = ctx.policy.inference
    features = ctx.data.dataset_features

    strategy = BaseStrategy(cfg.strategy)
    strategy.setup(ctx)
    try:
        import time as _time
        timings = {"get_observation": [], "obs_processor": [], "build_frame": [], "get_action": []}
        for step in range(args.steps):
            _t = _time.perf_counter()
            obs_raw = robot.get_observation()
            timings["get_observation"].append((_time.perf_counter() - _t) * 1000)

            _t = _time.perf_counter()
            obs_processed = ctx.processors.robot_observation_processor(obs_raw)
            timings["obs_processor"].append((_time.perf_counter() - _t) * 1000)

            engine.notify_observation(obs_processed)
            _t = _time.perf_counter()
            obs_frame = build_dataset_frame(features, obs_processed, prefix=OBS_STR)
            timings["build_frame"].append((_time.perf_counter() - _t) * 1000)

            state = np.asarray(obs_frame.get("observation.state")).flatten()
            oob = [labels[i] for i in range(len(state)) if state[i] < st_min[i] - 1e-6 or state[i] > st_max[i] + 1e-6]

            _t = _time.perf_counter()
            action_tensor = engine.get_action(obs_frame)
            timings["get_action"].append((_time.perf_counter() - _t) * 1000)
            print(f"\n--- step {step} ---")
            print("  EE state :", [round(float(x), 3) for x in state])
            print("  OUT-OF-DISTRIBUTION state dims:", oob if oob else "none")
            if action_tensor is not None:
                a = action_tensor.detach().cpu().numpy().flatten()
                print("  action   :", [round(float(x), 3) for x in a])
                if ac_min is not None:
                    a_oob = [labels[i] for i in range(min(len(a), 7)) if a[i] < ac_min[i] - 1e-6 or a[i] > ac_max[i] + 1e-6]
                    print("  OUT-OF-DISTRIBUTION action dims:", a_oob if a_oob else "none")
                print("  action - state (delta):", [round(float(a[i] - state[i]), 3) for i in range(min(len(a), len(state)))])

                # --- IK check: EE action -> joint targets, compared to current joints (no send) ---
                cur_joints = {n: float(obs_raw[f"{n}.pos"]) for n in motor_names if f"{n}.pos" in obs_raw}

                # Round-trip: IK of the CURRENT EE state should reproduce the CURRENT joints.
                # A large residual here means FK and IK disagree (calibration/URDF/convention bug);
                # a small residual means the IK is self-consistent and the jumps are singularity-driven.
                rt_action = {k: float(state[i]) for i, k in enumerate(correct_action_keys)}
                rt_out = ctx.processors.robot_action_processor((rt_action, obs_raw))
                rt_joints = {k.replace(".pos", ""): float(v) for k, v in rt_out.items()}
                rt_resid = {n: round(rt_joints.get(n, float("nan")) - cur_joints[n], 2) for n in cur_joints}
                print("  ROUND-TRIP IK(current EE) - current joints:", rt_resid)
                print("  max |round-trip residual| deg:", round(max(abs(v) for v in rt_resid.values()), 2))
                action_dict = {k: float(a[i]) for i, k in enumerate(correct_action_keys)}
                ik_out = ctx.processors.robot_action_processor((action_dict, obs_raw))
                ik_joints = {k.replace(".pos", ""): round(float(v), 2) for k, v in ik_out.items()}
                cur_round = {n: round(v, 2) for n, v in cur_joints.items()}
                jdelta = {n: round(ik_joints.get(n, float("nan")) - cur_joints[n], 2) for n in cur_joints}
                print("  current joints :", cur_round)
                print("  IK target joints:", ik_joints)
                print("  JOINT DELTA (IK target - current):", jdelta)
                print("  max |joint delta| deg:", round(max(abs(v) for v in jdelta.values()), 2))
            else:
                print("  action   : None (engine returned no action)")
    finally:
        # Teardown WITHOUT returning to any commanded position; we never sent one.
        strategy.teardown(ctx)

    import statistics as _st
    print("\n=== PER-STEP TIMING (ms), median across steps ===")
    total = 0.0
    for k, v in timings.items():
        if v:
            med = _st.median(v)
            total += med
            print(f"  {k:16s} median={med:8.1f} ms   (min={min(v):.0f} max={max(v):.0f})")
    print(f"  {'TOTAL/step':16s}        ={total:8.1f} ms  -> {1000.0/total:.1f} Hz ceiling")
    print("\nDONE (no motor commands were sent).")


if __name__ == "__main__":
    main()
