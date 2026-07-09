#!/usr/bin/env python3
"""No-motor A/B test: does appending the scene-graph `Context:` suffix change
the policy's predicted action at a fixed pose? Samples K fresh inferences for
each prompt on the SAME observation and reports the action distribution.

Run under sudo (RealSense). No motor commands are sent.
"""
import argparse, os, sys, statistics
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy-path", required=True)
    p.add_argument("--instruction", required=True, help="plain instruction (A)")
    p.add_argument("--context", default="\nContext: (black_cube_1, is_on_top_of, blue_box)",
                   help="suffix appended for the augmented instruction (B)")
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--follower-port", default="/dev/tty.usbmodem5B140303851")
    p.add_argument("--wrist-cam", default="0")
    p.add_argument("--agent-cam-serial", default="234222301106")
    p.add_argument("--urdf", default=os.path.join(ROOT_DIR, "SO101", "so101_new_calib.urdf"))
    args = p.parse_args()

    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.cameras.realsense import RealSenseCameraConfig
    from lerobot.configs import PreTrainedConfig
    from lerobot.model.kinematics import RobotKinematics
    from lerobot.processor import (RobotProcessorPipeline, observation_to_transition,
        robot_action_observation_to_transition, transition_to_observation, transition_to_robot_action)
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    from lerobot.robots.so_follower.robot_kinematic_processor import ForwardKinematicsJointsToEE, InverseKinematicsEEToJoints
    from lerobot.rollout import BaseStrategyConfig, RolloutConfig, build_rollout_context
    from lerobot.rollout.inference import SyncInferenceConfig
    from lerobot.rollout.strategies import BaseStrategy
    from lerobot.types import RobotAction, RobotObservation
    from lerobot.utils.constants import ACTION, OBS_STR
    from lerobot.utils.device_utils import auto_select_torch_device
    from lerobot.utils.feature_utils import build_dataset_frame
    from lerobot.utils.process import ProcessSignalHandler
    from lerobot.utils.utils import init_logging
    init_logging()

    device = auto_select_torch_device().type
    cams = {
        "wrist": OpenCVCameraConfig(index_or_path=int(args.wrist_cam), width=640, height=480, fps=args.fps, fourcc=None, warmup_s=3),
        "agent_view": RealSenseCameraConfig(serial_number_or_name=args.agent_cam_serial, width=640, height=480, fps=args.fps, warmup_s=2),
    }
    robot_config = SO101FollowerConfig(port=args.follower_port, id="follower_arm", cameras=cams, use_degrees=True)
    motor_names = list(SO101Follower(robot_config).bus.motors.keys())
    kin = RobotKinematics(urdf_path=args.urdf, target_frame_name="gripper_frame_link", joint_names=motor_names)
    op = RobotProcessorPipeline[RobotObservation, RobotObservation](steps=[ForwardKinematicsJointsToEE(kinematics=kin, motor_names=motor_names)], to_transition=observation_to_transition, to_output=transition_to_observation)
    tp = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](steps=[ForwardKinematicsJointsToEE(kinematics=kin, motor_names=motor_names)], to_transition=robot_action_observation_to_transition, to_output=transition_to_robot_action)
    ap = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](steps=[InverseKinematicsEEToJoints(kinematics=kin, motor_names=motor_names, initial_guess_current_joints=True)], to_transition=robot_action_observation_to_transition, to_output=transition_to_robot_action)

    pc = PreTrainedConfig.from_pretrained(args.policy_path)
    pc.pretrained_path = args.policy_path
    pc.device = device
    cfg = RolloutConfig(robot=robot_config, policy=pc, strategy=BaseStrategyConfig(), inference=SyncInferenceConfig(),
                        fps=args.fps, duration=10, task=args.instruction, device=device, return_to_initial_position=False)
    sh = ProcessSignalHandler(use_threads=True)
    ctx = build_rollout_context(cfg, sh.shutdown_event, teleop_action_processor=tp, robot_action_processor=ap, robot_observation_processor=op)
    keys = list(ctx.data.dataset_features[ACTION]["names"])
    ctx.data.ordered_action_keys[:] = keys
    ctx.policy.inference._ordered_action_keys = keys

    robot = ctx.hardware.robot_wrapper
    engine = ctx.policy.inference
    strategy = BaseStrategy(cfg.strategy)
    strategy.setup(ctx)

    obs = robot.get_observation()
    obs_proc = ctx.processors.robot_observation_processor(obs)
    engine.notify_observation(obs_proc)
    frame = build_dataset_frame(ctx.data.dataset_features, obs_proc, prefix=OBS_STR)
    state = np.asarray(frame["observation.state"]).flatten()
    labels = ["ee.x", "ee.y", "ee.z", "ee.wx", "ee.wy", "ee.wz", "grip"]
    print("\nEE state:", [round(float(x), 3) for x in state])

    prompts = {"A_plain": args.instruction, "B_augmented": args.instruction + args.context}
    results = {}
    for name, prompt in prompts.items():
        engine._task = prompt
        acts = []
        for _ in range(args.samples):
            engine.reset()               # clear action queue -> fresh inference
            engine.notify_observation(obs_proc)
            a = engine.get_action(frame)
            acts.append(a.detach().cpu().numpy().flatten())
        acts = np.array(acts)
        results[name] = acts
        mean = acts.mean(0); std = acts.std(0)
        print(f"\n=== {name} ===  prompt={prompt!r}")
        print("  mean action :", [round(float(x), 3) for x in mean])
        print("  std  action :", [round(float(x), 3) for x in std])
        print("  mean-state  :", [round(float(mean[i] - state[i]), 3) for i in range(min(len(mean), len(state)))])

    a, b = results["A_plain"].mean(0), results["B_augmented"].mean(0)
    print("\n=== A vs B (augmented - plain), mean action shift ===")
    for i, lab in enumerate(labels[:len(a)]):
        print(f"  {lab:6s} plain={a[i]:+.3f}  aug={b[i]:+.3f}  shift={b[i]-a[i]:+.3f}")
    strategy.teardown(ctx)
    print("\nDONE (no motor commands were sent).")


if __name__ == "__main__":
    main()
