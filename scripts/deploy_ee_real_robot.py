#!/usr/bin/env python3
"""Deploy a trained EE-space policy (ACT or Pi0.5) on the real SO101 robot.

Both ACT and Pi0.5 are trained on 7-D end-effector actions
(ee.x, ee.y, ee.z, ee.wx, ee.wy, ee.wz, ee.gripper_pos).
This script wraps the rollout engine with the correct FK/IK processors
so joint observations are converted to EE space before the policy sees
them, and EE action predictions are converted back to joint commands
before they reach the motors.

Usage:
    python scripts/deploy_ee_real_robot.py \\
        --policy-path outputs/train/pi05_so101_bowl_placement/checkpoints/last/pretrained_model \\
        --task "place the black bowl on the stove" \\
        --duration 60

    python scripts/deploy_ee_real_robot.py \\
        --policy-path outputs/train/act_so101_bowl_placement/checkpoints/last/pretrained_model \\
        --task "place the black bowl on the stove"
"""

import argparse
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--policy-path",
        required=True,
        help="Local checkpoint directory or HuggingFace model repo id.",
    )
    parser.add_argument("--task", required=True, help="Natural-language task description.")
    parser.add_argument("--duration", type=float, default=60.0, help="Episode duration in seconds.")
    parser.add_argument("--fps", type=int, default=30, help="Control frequency.")
    parser.add_argument(
        "--device",
        default=None,
        help=(
            "Torch device for inference (cuda/mps/cpu). Defaults to auto-select. The checkpoint "
            "records device='cuda'; on a Mac (no CUDA) leave this unset to auto-pick mps/cpu."
        ),
    )
    parser.add_argument("--follower-port", default="/dev/tty.usbmodem5B140303851", help="Follower arm USB port.")
    parser.add_argument("--follower-id", default="follower_arm", help="Follower arm calibration ID.")
    parser.add_argument(
        "--wrist-cam",
        default="/dev/video2",
        help="Wrist camera device path (Linux) or integer OpenCV index (macOS, e.g. 0).",
    )
    parser.add_argument("--agent-cam", default="/dev/video8", help="Agent-view OpenCV camera device path.")
    parser.add_argument(
        "--agent-cam-serial",
        default=None,
        help=(
            "Intel RealSense serial number for the agent-view camera. When set, the agent-view "
            "stream is captured via the RealSense (librealsense) backend instead of OpenCV. "
            "Required on macOS, where the RealSense does not stream through OpenCV/AVFoundation."
        ),
    )
    parser.add_argument(
        "--wrist-fourcc",
        default=None,
        help=(
            "FOURCC video format for the wrist OpenCV camera (e.g. MJPG, YUYV). Default None = "
            "auto-detect. On macOS forcing MJPG can fail to set and leave the camera unable to "
            "deliver frames during connect, so leaving this unset is the robust choice."
        ),
    )
    parser.add_argument("--cam-width", type=int, default=640)
    parser.add_argument("--cam-height", type=int, default=480)
    parser.add_argument(
        "--urdf",
        default=os.path.join(ROOT_DIR, "SO101", "so101_new_calib.urdf"),
        help="Path to SO101 URDF for kinematics.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.cameras.realsense import RealSenseCameraConfig
    from lerobot.configs import PreTrainedConfig
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
    from lerobot.utils.constants import ACTION
    from lerobot.utils.process import ProcessSignalHandler
    from lerobot.utils.utils import init_logging

    init_logging()

    # Wrist camera: OpenCV. Accept an integer index (macOS/AVFoundation) or a device path (Linux).
    wrist_index = int(args.wrist_cam) if str(args.wrist_cam).isdigit() else args.wrist_cam

    # Agent-view camera. On macOS the Intel RealSense will not stream through
    # OpenCV/AVFoundation (opens but delivers 0 frames), so it must be captured via the
    # librealsense backend, selected by passing --agent-cam-serial. The RealSense backend
    # needs to seize the device from the macOS kernel UVC driver, which requires running
    # this script with `sudo`. On Linux the RealSense also works as a plain OpenCV device.
    if args.agent_cam_serial:
        agent_view_config = RealSenseCameraConfig(
            serial_number_or_name=args.agent_cam_serial,
            width=args.cam_width,
            height=args.cam_height,
            fps=args.fps,
            warmup_s=20,
        )
    else:
        agent_view_config = OpenCVCameraConfig(
            index_or_path=int(args.agent_cam) if str(args.agent_cam).isdigit() else args.agent_cam,
            width=args.cam_width,
            height=args.cam_height,
            fps=args.fps,
            fourcc="MJPG",
        )

    wrist_cam_config = OpenCVCameraConfig(
        index_or_path=wrist_index,
        width=args.cam_width,
        height=args.cam_height,
        fps=args.fps,
        fourcc=args.wrist_fourcc,
        warmup_s=3,
    )
    # macOS contention fix: connect the RealSense (librealsense) FIRST so it seizes the
    # color interface from the kernel UVC driver BEFORE OpenCV/AVFoundation opens the wrist
    # and locks the UVC subsystem. Wrist-first makes the RealSense color seize time out,
    # even though the RealSense delivers color instantly when connected alone. dict order
    # is connect order in SO101Follower.connect().
    camera_config = {
        "agent_view": agent_view_config,
        "wrist": wrist_cam_config,
    }

    robot_config = SO101FollowerConfig(
        port=args.follower_port,
        id=args.follower_id,
        cameras=camera_config,
        use_degrees=True,
    )

    # Peek at motor names without connecting (used to configure kinematics).
    temp_robot = SO101Follower(robot_config)
    motor_names = list(temp_robot.bus.motors.keys())

    kinematics = RobotKinematics(
        urdf_path=args.urdf,
        target_frame_name="gripper_frame_link",
        joint_names=motor_names,
    )

    # Joint obs → EE obs (what the policy was trained on).
    robot_observation_processor = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[ForwardKinematicsJointsToEE(kinematics=kinematics, motor_names=motor_names)],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )

    # Declares the policy's action space (EE pose) to the rollout context, mirroring
    # record_data.py's `leader_joints_to_ee`. Without this, build_rollout_context falls
    # back to an identity teleop_action_processor over raw joint names, and a feature-
    # transform bug in ForwardKinematicsJointsToEE (it injects `ee.*` into the ACTION
    # feature dict even when only computing observation features) corrupts
    # dataset_features["action"]["names"] into a 13-entry mix of joint + EE names —
    # causing `make_robot_action` to index past the policy's 7-element action tensor.
    teleop_action_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[ForwardKinematicsJointsToEE(kinematics=kinematics, motor_names=motor_names)],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    # EE action → joint action (what the motors accept).
    robot_action_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            InverseKinematicsEEToJoints(
                kinematics=kinematics,
                motor_names=motor_names,
                initial_guess_current_joints=True,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    policy_config = PreTrainedConfig.from_pretrained(args.policy_path)
    policy_config.pretrained_path = args.policy_path

    # The checkpoint hard-codes device="cuda". On this machine that may be unavailable, and
    # RolloutConfig's device resolution trusts the policy's device string without checking
    # availability — so pick a concrete, available device here and pin it everywhere.
    from lerobot.utils.device_utils import auto_select_torch_device, is_torch_device_available

    device = args.device
    if device is None or not is_torch_device_available(device):
        device = auto_select_torch_device().type
    policy_config.device = device

    cfg = RolloutConfig(
        robot=robot_config,
        policy=policy_config,
        strategy=BaseStrategyConfig(),
        inference=SyncInferenceConfig(),
        fps=args.fps,
        duration=args.duration,
        task=args.task,
        device=device,
    )

    print(f"Policy:   {args.policy_path}")
    print(f"Robot:    so101_follower on {args.follower_port} ({args.follower_id})")
    if args.agent_cam_serial:
        print(f"Cameras:  wrist={args.wrist_cam} (opencv)  agent_view=RealSense#{args.agent_cam_serial}")
    else:
        print(f"Cameras:  wrist={args.wrist_cam} (opencv)  agent_view={args.agent_cam} (opencv)")
    print(f"Task:     {args.task}")
    print(f"Device:   {device}")
    print(f"Duration: {args.duration}s @ {args.fps} FPS")
    # confirm = input("Place the robot in a safe start pose, clear the workspace, then type RUN: ")
    # if confirm.strip() != "RUN":
    #     print("Aborted.")
    #     sys.exit(0)

    signal_handler = ProcessSignalHandler(use_threads=True)
    ctx = build_rollout_context(
        cfg,
        signal_handler.shutdown_event,
        teleop_action_processor=teleop_action_processor,
        robot_action_processor=robot_action_processor,
        robot_observation_processor=robot_observation_processor,
    )

    # Work around a library bug in build_rollout_context: it resolves
    # `ordered_action_keys` against the robot's raw joint names (since this
    # policy has no `action_feature_names` set), but `make_robot_action` keys
    # its output dict by `dataset_features["action"]["names"]` — which, for an
    # EE-space policy, is the EE pose names (ee.x, ee.y, ...), not joint names.
    # That mismatch raises `KeyError: 'shoulder_pan.pos'` inside get_action().
    # Realign both copies of the ordering to the actual EE action-feature names.
    correct_action_keys = list(ctx.data.dataset_features[ACTION]["names"])
    ctx.data.ordered_action_keys[:] = correct_action_keys
    ctx.policy.inference._ordered_action_keys = correct_action_keys

    strategy = BaseStrategy(cfg.strategy)
    try:
        strategy.setup(ctx)
        strategy.run(ctx)
    finally:
        strategy.teardown(ctx)


if __name__ == "__main__":
    main()
