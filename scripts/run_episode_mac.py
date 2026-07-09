#!/usr/bin/env python3
"""Run one real-robot experiment episode on macOS and record it.

Modes (via --condition / --context-mode):
  * standard  : plain instruction (variant a = C0, and the language ablations C1-C8)
  * scene_graph: instruction augmented ONCE at episode start with a Gemini scene graph (variant b)

Video: we cannot open the RealSense twice on macOS, so the episode video is the
exact `agent_view` frames the policy sees, tapped from the control loop.

Output: experiments/results/media/<task>/<condition>/trial_<timestamp>/
    snapshot_start.png, episode.mp4, meta.json, [scene_graph_log.json]

This is NON-interactive (no RUN prompt) so it can run under `sudo -S`. Coordinate
scene readiness before launching. After the episode it prints the exp_log.py
command (with --media-dir filled in) to record the outcome.

Dry-run: `--dry-run` connects only the cameras, saves a snapshot + a few seconds
of agent_view video, writes meta.json, and exits — no robot, no policy, no motion.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_MEDIA = os.path.join(ROOT_DIR, "experiments", "results", "media")
EMBODIMENT_SEMANTIC = "/Users/refinath/work/EmbodimentSemantic"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy-path", help="SmolVLA checkpoint dir (not needed for --dry-run).")
    p.add_argument("--task", required=True, help="Task key (e.g. cube_to_drawer) — used for the output folder.")
    p.add_argument("--instruction", default="", help="Exact instruction string sent to the policy (may be empty).")
    p.add_argument("--condition", required=True, help="C0..C8 or scene_graph (for the output folder + logging).")
    p.add_argument("--context-mode", default="standard", choices=["standard", "scene_graph"])
    p.add_argument("--env", type=int, default=1, help="Env-setup index (1..10), for bookkeeping.")
    p.add_argument("--duration", type=float, default=25.0)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--device", default=None)
    p.add_argument("--follower-port", default="/dev/tty.usbmodem5B140303851")
    p.add_argument("--follower-id", default="follower_arm")
    p.add_argument("--wrist-cam", default="0")
    p.add_argument("--agent-cam-serial", default="339222071083",
                   help="RealSense serial (librealsense path). Only used if --agent-cam-opencv is not set.")
    p.add_argument("--agent-cam-opencv", default=None,
                   help="OpenCV index for the RealSense RGB via AVFoundation (preferred on macOS, where "
                        "librealsense streaming crashes). Identify it each session with a camera probe.")
    p.add_argument("--cam-width", type=int, default=640)
    p.add_argument("--cam-height", type=int, default=480)
    p.add_argument("--urdf", default=os.path.join(ROOT_DIR, "SO101", "so101_new_calib.urdf"))
    # scene-graph (variant b)
    p.add_argument("--objects", nargs="+", default=None)
    p.add_argument("--vlm-model", default="gemini-2.5-flash")
    p.add_argument("--scene-graph-format", default="triplet", choices=["triplet", "natural_language", "json"])
    p.add_argument("--sg-min-interval", type=float, default=5.0,
                   help="Min seconds between background VLM refreshes (budget control). Actual cadence is "
                        "max(this, VLM latency ~7s for flash).")
    # dry-run
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--dry-seconds", type=float, default=6.0)
    return p.parse_args()


def build_camera_configs(args):
    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.cameras.realsense import RealSenseCameraConfig
    wrist_index = int(args.wrist_cam) if str(args.wrist_cam).isdigit() else args.wrist_cam

    # agent_view: prefer OpenCV/AVFoundation by index (--agent-cam-opencv) because
    # librealsense streaming segfaults on this Mac. The D435I RGB streams cleanly over
    # AVFoundation (no IR dots). Fall back to the librealsense-by-serial path only if
    # no OpenCV index is given (e.g. on Linux).
    if args.agent_cam_opencv is not None:
        agent_view = OpenCVCameraConfig(
            index_or_path=int(args.agent_cam_opencv), width=args.cam_width,
            height=args.cam_height, fps=args.fps, fourcc=None, warmup_s=3,
        )
    else:
        agent_view = RealSenseCameraConfig(
            serial_number_or_name=args.agent_cam_serial, width=args.cam_width,
            height=args.cam_height, fps=args.fps, warmup_s=20,
        )
    wrist = OpenCVCameraConfig(
        index_or_path=wrist_index, width=args.cam_width, height=args.cam_height,
        fps=args.fps, fourcc=None, warmup_s=3,
    )
    # IMPORTANT (macOS): when agent_view is the RealSense (librealsense), connect it FIRST
    # so it seizes the color interface from the kernel UVC driver BEFORE OpenCV/AVFoundation
    # opens the wrist and locks the UVC subsystem. Wrist-first makes the RealSense color
    # seize time out (proven: RealSense delivers color in 0.5s alone, times out at 20s when
    # the wrist opens first; agent-first connects both). For an all-OpenCV agent_view there
    # is no librealsense seize, so wrist-first is fine.
    if args.agent_cam_opencv is not None:
        return {"wrist": wrist, "agent_view": agent_view}
    return {"agent_view": agent_view, "wrist": wrist}


def _write_frame(writer, rgb):
    import cv2
    writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def dry_run(args, media_dir):
    """Connect cameras only, save a snapshot + a few seconds of agent_view video."""
    import cv2
    from lerobot.cameras.utils import make_cameras_from_configs
    from lerobot.utils.utils import init_logging
    init_logging()

    cams = make_cameras_from_configs(build_camera_configs(args))
    for c in cams.values():
        c.connect()
    agent = cams["agent_view"]

    frame = agent.read()  # RGB HxWx3
    os.makedirs(media_dir, exist_ok=True)
    cv2.imwrite(os.path.join(media_dir, "snapshot_start.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    h, w = frame.shape[:2]
    writer = cv2.VideoWriter(os.path.join(media_dir, "episode.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
    t0 = time.perf_counter()
    n = 0
    while time.perf_counter() - t0 < args.dry_seconds:
        _write_frame(writer, agent.read())
        n += 1
        time.sleep(1.0 / args.fps)
    writer.release()
    for c in cams.values():
        c.disconnect()
    print(f"[dry-run] wrote {n} agent_view frames to episode.mp4 ({w}x{h}) over {args.dry_seconds}s")


def full_run(args, media_dir):
    import cv2
    from PIL import Image

    from lerobot.configs import PreTrainedConfig
    from lerobot.model.kinematics import RobotKinematics
    from lerobot.processor import (
        RobotProcessorPipeline, observation_to_transition, robot_action_observation_to_transition,
        transition_to_observation, transition_to_robot_action,
    )
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    from lerobot.robots.so_follower.robot_kinematic_processor import (
        ForwardKinematicsJointsToEE, InverseKinematicsEEToJoints,
    )
    from lerobot.rollout import BaseStrategyConfig, RolloutConfig, build_rollout_context
    from lerobot.rollout.inference import SyncInferenceConfig
    from lerobot.rollout.strategies.base import BaseStrategy
    from lerobot.rollout.strategies.core import send_next_action
    from lerobot.types import RobotAction, RobotObservation
    from lerobot.utils.constants import ACTION
    from lerobot.utils.device_utils import auto_select_torch_device, is_torch_device_available
    from lerobot.utils.process import ProcessSignalHandler
    from lerobot.utils.robot_utils import precise_sleep
    from lerobot.utils.utils import init_logging
    init_logging()

    device = args.device
    if device is None or not is_torch_device_available(device):
        device = auto_select_torch_device().type

    robot_config = SO101FollowerConfig(
        port=args.follower_port, id=args.follower_id,
        cameras=build_camera_configs(args), use_degrees=True,
    )
    temp_robot = SO101Follower(robot_config)
    motor_names = list(temp_robot.bus.motors.keys())
    kin = RobotKinematics(urdf_path=args.urdf, target_frame_name="gripper_frame_link", joint_names=motor_names)

    obs_proc = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[ForwardKinematicsJointsToEE(kinematics=kin, motor_names=motor_names)],
        to_transition=observation_to_transition, to_output=transition_to_observation)
    teleop_proc = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[ForwardKinematicsJointsToEE(kinematics=kin, motor_names=motor_names)],
        to_transition=robot_action_observation_to_transition, to_output=transition_to_robot_action)
    act_proc = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[InverseKinematicsEEToJoints(kinematics=kin, motor_names=motor_names, initial_guess_current_joints=True)],
        to_transition=robot_action_observation_to_transition, to_output=transition_to_robot_action)

    policy_config = PreTrainedConfig.from_pretrained(args.policy_path)
    policy_config.pretrained_path = args.policy_path
    policy_config.device = device

    cfg = RolloutConfig(
        robot=robot_config, policy=policy_config, strategy=BaseStrategyConfig(),
        inference=SyncInferenceConfig(), fps=args.fps, duration=args.duration,
        task=args.instruction, device=device,
    )

    signal_handler = ProcessSignalHandler(use_threads=True)
    ctx = build_rollout_context(
        cfg, signal_handler.shutdown_event,
        teleop_action_processor=teleop_proc, robot_action_processor=act_proc,
        robot_observation_processor=obs_proc)

    correct_action_keys = list(ctx.data.dataset_features[ACTION]["names"])
    ctx.data.ordered_action_keys[:] = correct_action_keys
    ctx.policy.inference._ordered_action_keys = correct_action_keys

    robot = ctx.hardware.robot_wrapper
    engine = ctx.policy.inference

    # --- Scene graph: seed once, then refresh in a BACKGROUND thread (variant b) ---
    # The VLM runs off the control loop so the arm never freezes; engine._task is
    # swapped whenever a fresh scene graph arrives. Cadence = max(--sg-min-interval, VLM latency).
    import threading
    base_instruction = args.instruction
    sg_log = []
    sg_stop = threading.Event()
    sg_generator = None
    frame_box = {"frame": None}
    frame_lock = threading.Lock()
    run_t0 = [0.0]

    if args.context_mode == "scene_graph":
        sys.path.insert(0, os.path.join(EMBODIMENT_SEMANTIC, "real_robot_deploy"))
        sys.path.insert(0, os.path.join(EMBODIMENT_SEMANTIC, "vlm_benchmarking"))
        from dotenv import load_dotenv
        load_dotenv(os.path.join(EMBODIMENT_SEMANTIC, "vlm_benchmarking", ".env"))
        from live_vlm_scene_graph import LiveVLMSceneGraphGenerator
        from vlm_bench.models.gemini import GeminiVLM
        from transformers import AutoTokenizer
        try:
            tok = AutoTokenizer.from_pretrained(getattr(policy_config, "vlm_model_name",
                                                        "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"))
        except Exception:
            tok = None
        vlm = GeminiVLM(args.vlm_model, temperature=None, max_new_tokens=4096, max_retries=6)
        sg_generator = LiveVLMSceneGraphGenerator(
            vlm=vlm, objects=args.objects, scene_graph_format=args.scene_graph_format,
            tokenizer=tok, max_prompt_tokens=getattr(policy_config, "tokenizer_max_length", 48))
        # Seed synchronously so the very first policy step already has a scene graph.
        obs0 = robot.get_observation()
        t0 = time.perf_counter()
        suffix = sg_generator.suffix_for_frame(Image.fromarray(obs0["agent_view"]), base_instruction, "scene_graph")
        engine._task = f"{base_instruction}{suffix}"
        sg_log.append({"t_s": 0.0, "latency_s": round(time.perf_counter() - t0, 2),
                       "raw_response": sg_generator._last_response, "prompt_sent": engine._task})
        print(f"[scene_graph seed] {sg_log[-1]['latency_s']}s ({args.vlm_model}) -> {engine._task!r}")

    def sg_worker():
        while not sg_stop.is_set():
            sg_stop.wait(args.sg_min_interval)
            if sg_stop.is_set():
                break
            with frame_lock:
                fr = frame_box["frame"]
            if fr is None:
                continue
            try:
                t0 = time.perf_counter()
                suffix = sg_generator.suffix_for_frame(Image.fromarray(fr), base_instruction, "scene_graph")
                engine._task = f"{base_instruction}{suffix}"
                sg_log.append({"t_s": round(time.perf_counter() - run_t0[0], 2),
                               "latency_s": round(time.perf_counter() - t0, 2),
                               "raw_response": sg_generator._last_response, "prompt_sent": engine._task})
            except Exception as e:
                sg_log.append({"t_s": round(time.perf_counter() - run_t0[0], 2), "error": str(e)[:200]})

    # --- Recording strategy: tap agent_view frames into episode.mp4 ---
    video_path = os.path.join(media_dir, "episode.mp4")
    snap_path = os.path.join(media_dir, "snapshot_start.png")
    motion_stats = {"max_travel_deg": 0.0, "per_joint": {}}

    class RecordingStrategy(BaseStrategy):
        def run(self, ctx):
            cfg = ctx.runtime.cfg
            robot = ctx.hardware.robot_wrapper
            interp = self._interpolator
            control_interval = interp.get_control_interval(cfg.fps)
            writer = None
            sg_thread = None
            pos_min, pos_max = {}, {}
            start_time = time.perf_counter()
            run_t0[0] = start_time
            self._engine.resume()
            try:
                while not ctx.runtime.shutdown_event.is_set():
                    loop_start = time.perf_counter()
                    if cfg.duration > 0 and (time.perf_counter() - start_time) >= cfg.duration:
                        break
                    obs = robot.get_observation()
                    for k, v in obs.items():
                        if isinstance(k, str) and k.endswith(".pos"):
                            pos_min[k] = min(pos_min.get(k, v), v)
                            pos_max[k] = max(pos_max.get(k, v), v)
                    frame = obs["agent_view"]
                    with frame_lock:
                        frame_box["frame"] = frame
                    if writer is None:
                        h, w = frame.shape[:2]
                        cv2.imwrite(snap_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                        writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), cfg.fps, (w, h))
                        if sg_generator is not None:
                            sg_thread = threading.Thread(target=sg_worker, daemon=True)
                            sg_thread.start()
                    _write_frame(writer, frame)
                    obs_processed = self._process_observation_and_notify(ctx.processors, obs)
                    if self._handle_warmup(cfg.use_torch_compile, loop_start, control_interval):
                        continue
                    action_dict = send_next_action(obs_processed, obs, ctx, interp)
                    self._log_telemetry(obs_processed, action_dict, ctx.runtime)
                    dt = time.perf_counter() - loop_start
                    if (sleep_t := control_interval - dt) > 0:
                        precise_sleep(sleep_t)
            finally:
                sg_stop.set()
                if sg_thread is not None:
                    sg_thread.join(timeout=15)
                if writer is not None:
                    writer.release()
                travel = {k.replace(".pos", ""): round(pos_max[k] - pos_min[k], 1) for k in pos_min}
                motion_stats["per_joint"] = travel
                motion_stats["max_travel_deg"] = max(travel.values()) if travel else 0.0

    strategy = RecordingStrategy(cfg.strategy)
    try:
        strategy.setup(ctx)
        strategy.run(ctx)
    finally:
        strategy.teardown(ctx)

    mt = motion_stats["max_travel_deg"]
    print(f"[motion] per-joint travel (deg): {motion_stats['per_joint']}  max={mt}")
    if mt < 5.0:
        print("\n  *** WARNING: arm barely moved (max joint travel < 5 deg) — likely DEGRADED/VIBRATING")
        print("      servos. Power-cycle and rerun. This result should NOT be logged. ***\n")

    if sg_log:
        with open(os.path.join(media_dir, "scene_graph_log.json"), "w") as f:
            json.dump({"task": args.task, "instruction": base_instruction, "vlm_model": args.vlm_model,
                       "objects": args.objects, "scene_graph_format": args.scene_graph_format,
                       "sg_min_interval_s": args.sg_min_interval, "refreshes": sg_log}, f, indent=2)
        print(f"[scene_graph] {len(sg_log)} refresh(es) logged")
    print(f"[full-run] video: {video_path}")
    return motion_stats


def main():
    args = parse_args()
    if args.objects is None:
        args.objects = ["black_cube_1", "black_cube_2", "blue_box", "drawer", "cylinder", "yellow_rectangle"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    media_dir = os.path.join(RESULTS_MEDIA, args.task, args.condition, f"trial_{timestamp}")
    os.makedirs(media_dir, exist_ok=True)

    print("=" * 72)
    print(f"  {'DRY-RUN' if args.dry_run else 'EPISODE'}   task={args.task}  condition={args.condition}  "
          f"mode={args.context_mode}  env={args.env}")
    print(f"  checkpoint : {args.policy_path}")
    print(f"  instruction: {args.instruction!r}")
    print(f"  duration   : {args.duration}s @ {args.fps} FPS   device=auto")
    print(f"  output     : {media_dir}")
    print("=" * 72)

    meta = {
        "timestamp": timestamp, "task": args.task, "condition": args.condition,
        "context_mode": args.context_mode, "env": args.env, "instruction": args.instruction,
        "policy_path": args.policy_path, "duration_s": args.duration, "fps": args.fps,
        "dry_run": args.dry_run,
    }

    if args.dry_run:
        dry_run(args, media_dir)
    else:
        motion = full_run(args, media_dir)
        meta["motion"] = motion
        meta["suspect_no_motion"] = bool(motion and motion.get("max_travel_deg", 0) < 5.0)

    with open(os.path.join(media_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Under sudo, hand the results tree back to the invoking user so exp_log.py
    # (run as that user) can write trials.csv / live_table.md without a chown.
    try:
        if hasattr(os, "geteuid") and os.geteuid() == 0 and os.environ.get("SUDO_UID"):
            uid = int(os.environ["SUDO_UID"])
            gid = int(os.environ.get("SUDO_GID", uid))
            results_root = os.path.join(ROOT_DIR, "experiments", "results")
            for dp, _dns, fns in os.walk(results_root):
                os.chown(dp, uid, gid)
                for fn in fns:
                    os.chown(os.path.join(dp, fn), uid, gid)
    except Exception as e:
        print(f"[warn] could not chown results back to user: {e}")

    print("\n--- log this trial (I will ask you grasp/success, then run) ---")
    print(f"python scripts/exp_log.py log --task {args.task} --condition {args.condition} "
          f"--context-mode {args.context_mode} --env {args.env} "
          f'--instruction "{args.instruction}" --grasp <0|1> --success <0|1> '
          f"--failure-mode <mode> --media-dir {os.path.relpath(media_dir, ROOT_DIR)}")


if __name__ == "__main__":
    main()
