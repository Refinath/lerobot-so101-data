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
    parser.add_argument("--follower-port", default="/dev/ttyACM0", help="Follower arm USB port.")
    parser.add_argument("--follower-id", default="follower_arm", help="Follower arm calibration ID.")
    parser.add_argument("--wrist-cam", default="/dev/video2", help="Wrist camera device path.")
    parser.add_argument("--agent-cam", default="/dev/video8", help="Agent-view camera device path.")
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
    from lerobot.utils.process import ProcessSignalHandler
    from lerobot.utils.utils import init_logging

    init_logging()

    camera_config = {
        "wrist": OpenCVCameraConfig(
            index_or_path=args.wrist_cam,
            width=args.cam_width,
            height=args.cam_height,
            fps=args.fps,
            fourcc="MJPG",
        ),
        "agent_view": OpenCVCameraConfig(
            index_or_path=args.agent_cam,
            width=args.cam_width,
            height=args.cam_height,
            fps=args.fps,
            fourcc="MJPG",
        ),
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

    cfg = RolloutConfig(
        robot=robot_config,
        policy=policy_config,
        strategy=BaseStrategyConfig(),
        inference=SyncInferenceConfig(),
        fps=args.fps,
        duration=args.duration,
        task=args.task,
    )

    print(f"Policy:   {args.policy_path}")
    print(f"Robot:    so101_follower on {args.follower_port} ({args.follower_id})")
    print(f"Cameras:  wrist={args.wrist_cam}  agent={args.agent_cam}")
    print(f"Task:     {args.task}")
    print(f"Duration: {args.duration}s @ {args.fps} FPS")
    confirm = input("Place the robot in a safe start pose, clear the workspace, then type RUN: ")
    if confirm.strip() != "RUN":
        print("Aborted.")
        sys.exit(0)

    signal_handler = ProcessSignalHandler(use_threads=True)
    ctx = build_rollout_context(
        cfg,
        signal_handler.shutdown_event,
        robot_action_processor=robot_action_processor,
        robot_observation_processor=robot_observation_processor,
    )

    strategy = BaseStrategy(cfg.strategy)
    try:
        strategy.setup(ctx)
        strategy.run(ctx)
    finally:
        strategy.teardown(ctx)


if __name__ == "__main__":
    main()
