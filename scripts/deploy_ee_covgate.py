#!/usr/bin/env python3
"""Deploy an EE-space policy with CovGate variance reduction on the real SO101 robot.

Identical to deploy_ee_real_robot.py except the policy is called K times per re-plan
and the K action chunks are combined by eigenvalue-weighted covariance gating before
execution. Use --k-samples to set K (default 4) and --gamma for gate sharpness
(default 1.0 = linear gate).

Usage:
    python scripts/deploy_ee_covgate.py \\
        --policy-path outputs/train/pi05_<task>/checkpoints/last/pretrained_model \\
        --task "<task description>" \\
        --k-samples 4
"""

import argparse
import math
import os
import sys
import time
from collections import deque
from contextlib import nullcontext
from copy import copy

import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from covgate.gate import gate_action  # noqa: E402

_EE_WY_IDX = 4


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--follower-port", default="/dev/ttyACM0")
    parser.add_argument("--follower-id", default="follower_arm")
    parser.add_argument("--wrist-cam", default="/dev/video2")
    parser.add_argument("--agent-cam", default="/dev/video8")
    parser.add_argument("--cam-width", type=int, default=640)
    parser.add_argument("--cam-height", type=int, default=480)
    parser.add_argument(
        "--urdf",
        default=os.path.join(ROOT_DIR, "SO101", "so101_new_calib.urdf"),
    )
    parser.add_argument(
        "--k-samples", type=int, default=4,
        help="Number of independent policy samples per re-plan (default 4).",
    )
    parser.add_argument(
        "--gamma", type=float, default=1.0,
        help="Gate sharpness exponent; 1.0 = linear (default).",
    )
    return parser.parse_args()


def _make_covgate_engine(base_engine, k_samples: int, gamma: float):
    """Return a CovGate-wrapped inference engine built from an existing one.

    On each re-plan:
      1. Call policy.predict_action_chunk() K times (each call draws fresh noise).
      2. Stack K chunks into shape (K, T, A).
      3. Apply gate_action → one (T, A) gated chunk.
      4. Fill internal queue with the gated steps.
    On subsequent ticks within the chunk: pop from queue without calling policy.
    """
    import torch
    from lerobot.policies.utils import make_robot_action, prepare_observation_for_inference
    from lerobot.rollout.inference.sync import SyncInferenceEngine

    class _CovGateEngine(SyncInferenceEngine):
        def __init__(self, *args, k: int, g: float, **kwargs):
            super().__init__(*args, **kwargs)
            self._k = k
            self._g = g
            self._queue: deque = deque()

        def reset(self):
            super().reset()
            self._queue.clear()

        def get_action(self, obs_frame):
            if obs_frame is None:
                return None

            observation = copy(obs_frame)
            use_amp = getattr(self._policy.config, "use_amp", False)
            autocast_ctx = (
                torch.autocast(device_type=self._device.type)
                if self._device.type == "cuda" and use_amp
                else nullcontext()
            )

            with torch.inference_mode(), autocast_ctx:
                observation = prepare_observation_for_inference(
                    observation, self._device, self._task, self._robot_type
                )
                observation = self._preprocessor(observation)

                if not self._queue:
                    chunks = []
                    for _ in range(self._k):
                        chunk = self._policy.predict_action_chunk(observation)
                        chunks.append(chunk.squeeze(0).float().cpu().numpy())

                    samples = np.stack(chunks)                     # (K, T, A)
                    gated = gate_action(samples, gamma=self._g)    # (T, A)

                    n = self._policy.config.n_action_steps
                    gated_t = torch.from_numpy(gated[:n]).float()
                    self._queue.extend(gated_t)                    # n tensors of (A,)

                action_step = self._queue.popleft().unsqueeze(0)   # (1, A)
                action = self._postprocessor(action_step)

            action_tensor = action.squeeze(0).cpu()
            action_dict = make_robot_action(action_tensor, self._dataset_features)
            return torch.tensor([action_dict[kk] for kk in self._ordered_action_keys])

    return _CovGateEngine(
        policy=base_engine._policy,
        preprocessor=base_engine._preprocessor,
        postprocessor=base_engine._postprocessor,
        dataset_features=base_engine._dataset_features,
        ordered_action_keys=base_engine._ordered_action_keys,
        task=base_engine._task,
        device=str(base_engine._device),
        robot_type=base_engine._robot_type,
        k=k_samples,
        g=gamma,
    )


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
    from lerobot.rollout.strategies.core import send_next_action
    from lerobot.types import RobotAction, RobotObservation
    from lerobot.utils.constants import ACTION
    from lerobot.utils.process import ProcessSignalHandler
    from lerobot.utils.robot_utils import precise_sleep
    from lerobot.utils.utils import init_logging

    init_logging()

    camera_config = {
        "wrist": OpenCVCameraConfig(
            index_or_path=args.wrist_cam,
            width=args.cam_width, height=args.cam_height,
            fps=args.fps, fourcc="MJPG",
        ),
        "agent_view": OpenCVCameraConfig(
            index_or_path=args.agent_cam,
            width=args.cam_width, height=args.cam_height,
            fps=args.fps, fourcc="MJPG",
        ),
    }

    robot_config = SO101FollowerConfig(
        port=args.follower_port, id=args.follower_id,
        cameras=camera_config, use_degrees=True,
    )

    temp_robot = SO101Follower(robot_config)
    motor_names = list(temp_robot.bus.motors.keys())

    kinematics = RobotKinematics(
        urdf_path=args.urdf,
        target_frame_name="gripper_frame_link",
        joint_names=motor_names,
    )

    robot_observation_processor = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[ForwardKinematicsJointsToEE(kinematics=kinematics, motor_names=motor_names)],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )
    teleop_action_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[ForwardKinematicsJointsToEE(kinematics=kinematics, motor_names=motor_names)],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    robot_action_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[InverseKinematicsEEToJoints(
            kinematics=kinematics, motor_names=motor_names,
            initial_guess_current_joints=True,
        )],
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

    print(f"Policy:    {args.policy_path}")
    print(f"Robot:     so101_follower on {args.follower_port} ({args.follower_id})")
    print(f"Task:      {args.task}")
    print(f"CovGate:   K={args.k_samples}  gamma={args.gamma}")
    print(f"Duration:  {args.duration}s @ {args.fps} FPS")
    confirm = input("Place the robot in a safe start pose, clear the workspace, then type RUN: ")
    if confirm.strip() != "RUN":
        print("Aborted.")
        sys.exit(0)

    signal_handler = ProcessSignalHandler(use_threads=True)
    ctx = build_rollout_context(
        cfg,
        signal_handler.shutdown_event,
        teleop_action_processor=teleop_action_processor,
        robot_action_processor=robot_action_processor,
        robot_observation_processor=robot_observation_processor,
    )

    correct_action_keys = list(ctx.data.dataset_features[ACTION]["names"])
    ctx.data.ordered_action_keys[:] = correct_action_keys

    ctx.policy.inference = _make_covgate_engine(
        ctx.policy.inference, k_samples=args.k_samples, gamma=args.gamma,
    )
    ctx.policy.inference._ordered_action_keys = correct_action_keys

    class EeWyUnwrapStrategy(BaseStrategy):
        def run(self, ctx) -> None:
            engine = self._engine
            cfg = ctx.runtime.cfg
            robot = ctx.hardware.robot_wrapper
            interpolator = self._interpolator
            control_interval = interpolator.get_control_interval(cfg.fps)

            prev_wy = None
            start_time = time.perf_counter()
            engine.resume()

            while not ctx.runtime.shutdown_event.is_set():
                loop_start = time.perf_counter()

                if cfg.duration > 0 and (time.perf_counter() - start_time) >= cfg.duration:
                    print(f"Duration limit reached ({cfg.duration:.0f}s)")
                    break

                obs = robot.get_observation()
                obs_processed = self._process_observation_and_notify(ctx.processors, obs)

                if self._handle_warmup(cfg.use_torch_compile, loop_start, control_interval):
                    continue

                state = obs_processed["observation.state"]
                wy = float(state[_EE_WY_IDX])
                if prev_wy is not None:
                    diff = wy - prev_wy
                    if diff > math.pi:
                        wy -= 2 * math.pi
                    elif diff < -math.pi:
                        wy += 2 * math.pi
                    try:
                        state[_EE_WY_IDX] = wy
                    except (TypeError, ValueError):
                        state = state.clone()
                        state[_EE_WY_IDX] = wy
                        obs_processed["observation.state"] = state
                prev_wy = wy

                action_dict = send_next_action(obs_processed, obs, ctx, interpolator)
                self._log_telemetry(obs_processed, action_dict, ctx.runtime)

                dt = time.perf_counter() - loop_start
                if (sleep_t := control_interval - dt) > 0:
                    precise_sleep(sleep_t)
                else:
                    print(f"[warn] loop slower ({1/dt:.1f} Hz) than target ({cfg.fps} Hz)")

    strategy = EeWyUnwrapStrategy(cfg.strategy)
    try:
        strategy.setup(ctx)
        strategy.run(ctx)
    finally:
        strategy.teardown(ctx)


if __name__ == "__main__":
    main()
