# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Record SO100 demonstrations into prompt-specific LeRobot datasets.

Purpose:
    Use this script to collect teleoperated demonstrations from a SO100 leader
    arm to a SO100 follower arm. Each prompt/instruction is written to its own
    dataset folder so data stays grouped by task.

Common commands:
    Create a new video-backed dataset:
        python record_data.py --prompt "place black bowl in front of the drawer"

    Resume an existing dataset for the same prompt:
        python record_data.py --prompt "place black bowl in front of the drawer" --resume

    Overwrite an existing dataset and start fresh:
        python record_data.py --prompt "place black bowl in front of the drawer" --overwrite

    Store camera frames as PNG images instead of MP4 videos:
        python record_data.py --prompt "place black bowl in front of the drawer" --save-images

Keyboard controls while running:
    Space: start the next episode.
    Enter: end the current episode/reset loop early.
    Left arrow: discard the current episode and record it again.
    Esc: stop the whole recording session.
"""

import argparse
import os
import re
import shutil
from pathlib import Path
import select
import sys
import termios
import threading
import time
import tty

os.environ["HF_LEROBOT_HOME"] = "./"

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.datasets import LeRobotDataset, aggregate_pipeline_dataset_features, create_initial_features
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import (
    RobotProcessorPipeline,
    observation_to_transition,
    robot_action_observation_to_transition,
    transition_to_observation,
    transition_to_robot_action,
)
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    ForwardKinematicsJointsToEE,
    InverseKinematicsEEToJoints,
)
from lerobot.scripts.lerobot_record import record_loop
from lerobot.teleoperators.so_leader import SO100Leader, SO100LeaderConfig
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.feature_utils import combine_feature_dicts
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun


NUM_EPISODES = 2
FPS = 30
EPISODE_TIME_SEC = 60
RESET_TIME_SEC = 30

# Runtime keyboard controls. TerminalKeyboardListener implements these bindings.
START_EPISODE_KEY = "space"
STOP_RECORDING_KEY = "esc"
END_LOOP_KEY = "enter"
RERECORD_EPISODE_KEY = "left arrow"

HF_REPO_NAMESPACE = "dataset" # Use correct repo namespace

FOLLOWER_PORT = "/dev/ttyACM0" # Use correct follower port
LEADER_PORT = "/dev/ttyACM1" # Use correct leader port
FOLLOWER_ID = "follower_arm" # Use correct follower ID
LEADER_ID = "leader_arm" # Use correct leader ID
WRIST_CAMERA_PATH ="/dev/video8" # Use correct camera name
AGENT_CAMERA_PATH ="/dev/video10" # Use correct camera name
AGENT_DEPTH_CAMERA_PATH ="/dev/video5" # Use correct camera name


dataset_root = Path(os.environ["HF_LEROBOT_HOME"])


class TerminalKeyboardListener:
    """Small terminal keyboard listener compatible with LeRobot's events dict."""

    def __init__(self, events):
        self.events = events
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._old_settings = None

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=1.0)

    def _read_escape_sequence(self):
        sequence = "\x1b"
        readable, _, _ = select.select([sys.stdin], [], [], 0.15)
        if not readable:
            return sequence

        sequence += sys.stdin.read(1)
        if sequence != "\x1b[":
            return sequence

        readable, _, _ = select.select([sys.stdin], [], [], 0.15)
        if not readable:
            return sequence

        sequence += sys.stdin.read(1)
        return sequence

    def _handle_key(self, key):
        if key == " ":
            print("Space key pressed. Starting episode...")
            self.events["start_episode"] = True
        elif key in ("\r", "\n"):
            print("Enter key pressed. Exiting loop...")
            self.events["exit_early"] = True
        elif key == "\x1b[D":
            print("Left arrow key pressed. Exiting loop and rerecord the last episode...")
            self.events["rerecord_episode"] = True
            self.events["exit_early"] = True
        elif key == "\x1b":
            print("Escape key pressed. Stopping data recording...")
            self.events["stop_recording"] = True
            self.events["exit_early"] = True

    def _run(self):
        self._old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            while not self._stop_event.is_set():
                readable, _, _ = select.select([sys.stdin], [], [], 0.05)
                if not readable:
                    continue

                char = sys.stdin.read(1)
                key = self._read_escape_sequence() if char == "\x1b" else char
                self._handle_key(key)
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)


def init_terminal_keyboard_listener():
    events = {
        "start_episode": False,
        "exit_early": False,
        "rerecord_episode": False,
        "stop_recording": False,
    }

    if not sys.stdin.isatty():
        print("No interactive terminal detected. Keyboard controls are unavailable.")
        return None, events

    listener = TerminalKeyboardListener(events)
    listener.start()
    return listener, events


def wait_for_episode_start(events, episode_idx):
    """Block until TerminalKeyboardListener receives the start key."""
    events["start_episode"] = False
    events["exit_early"] = False
    log_say(
        f"Press {START_EPISODE_KEY} to start episode {episode_idx + 1}, "
        f"or {STOP_RECORDING_KEY} to stop recording"
    )

    if not sys.stdin.isatty():
        key_name = input("Type space to start or esc to stop, then press Enter: ").strip().lower()
        if key_name in (START_EPISODE_KEY, " "):
            events["start_episode"] = True
        elif key_name == STOP_RECORDING_KEY:
            events["stop_recording"] = True
            events["exit_early"] = True
        return events["start_episode"] and not events["stop_recording"]

    while not events["start_episode"] and not events["stop_recording"]:
        time.sleep(0.05)

    return events["start_episode"] and not events["stop_recording"]


def slugify_prompt(prompt):
    slug = re.sub(r"[^a-z0-9]+", "-", prompt.lower()).strip("-")
    return slug[:80].strip("-") or "instruction"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Record SO100 teleoperation data into a prompt-specific LeRobot dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Dataset behavior:
  default      Create a new dataset. Fails if the prompt dataset already exists.
  --resume    Append episodes to the existing dataset for the same prompt.
  --overwrite Delete the existing prompt dataset and start fresh.

Camera storage:
  default        Store cameras as MP4 videos under videos/.
  --save-images  Store cameras as PNG frames under images/ for newly created datasets.

Keyboard controls while running:
  {START_EPISODE_KEY:<10} Start the next episode.
  {END_LOOP_KEY:<10} End the current episode/reset loop early.
  {RERECORD_EPISODE_KEY:<10} Discard the current episode and record it again.
  {STOP_RECORDING_KEY:<10} Stop the whole recording session.

Examples:
  python record_data.py --prompt "place black bowl in front of the drawer"
  python record_data.py --prompt "place black bowl in front of the drawer" --resume
  python record_data.py --prompt "place black bowl in front of the drawer" --overwrite --save-images
""",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Task instruction to save with each episode. Its slug chooses the dataset folder.",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="For new datasets, store camera observations as PNG frames instead of MP4 videos.",
    )
    dataset_mode = parser.add_mutually_exclusive_group()

    dataset_mode.add_argument(
        "--resume",
        action="store_true",
        help="Append new episodes to the existing dataset for this prompt.",
    )
    dataset_mode.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the existing dataset for this prompt and start fresh.",
    )
    return parser.parse_args()


def open_dataset(args, dataset_features, robot_type, repo_id, dataset_path):
    use_videos = not args.save_images

    if args.overwrite and dataset_path.exists():
        print(f"Overwriting existing dataset at {dataset_path}")
        shutil.rmtree(dataset_path)

    if args.resume:
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Cannot resume because dataset does not exist at {dataset_path}. "
                "Run without --resume to create it, or use --overwrite to start fresh."
            )
        if args.save_images:
            print("--save-images is ignored with --resume; the existing dataset schema is reused.")
        print(f"Resuming dataset at {dataset_path}")
        return LeRobotDataset.resume(
            repo_id=repo_id,
            root=dataset_path,
            image_writer_threads=4,
        )

    if dataset_path.exists():
        raise FileExistsError(
            f"Dataset already exists at {dataset_path}. "
            "Use --resume to append episodes or --overwrite to delete it and start fresh."
        )

    print(f"Creating dataset at {dataset_path}")
    print(f"Camera storage: {'videos' if use_videos else 'images'}")
    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=FPS,
        features=dataset_features,
        robot_type=robot_type,
        use_videos=use_videos,
        image_writer_threads=4,
    )


def main():
    args = parse_args()
    task_prompt = args.prompt.strip()
    if not task_prompt:
        raise ValueError("--prompt cannot be empty.")
    repo_id = f"{HF_REPO_NAMESPACE}/{slugify_prompt(task_prompt)}"
    dataset_path = dataset_root / repo_id
    print(f"Task prompt: {task_prompt}")
    print(f"Dataset repo id: {repo_id}")
    print(f"Dataset path: {dataset_path}")
    print(
        "Controls: space=start episode, enter=end current loop, "
        "left arrow=rerecord episode, esc=stop recording"
    )

    # Create the robot and teleoperator configurations
    camera_config = {
    	"wrist": OpenCVCameraConfig(index_or_path=WRIST_CAMERA_PATH, width=640, height=480, fps=FPS, fourcc="MJPG"),
    	"agent_view": OpenCVCameraConfig(index_or_path=AGENT_CAMERA_PATH, width=640, height=480, fps=FPS, fourcc="MJPG"),
    	"agent_view_depth": OpenCVCameraConfig(index_or_path=AGENT_DEPTH_CAMERA_PATH, width=640, height=480, fps=FPS, fourcc="MJPG"),
    	}
    follower_config = SO100FollowerConfig(
        port=FOLLOWER_PORT,
        id=FOLLOWER_ID,
        cameras=camera_config,
        use_degrees=True,
    )
    leader_config = SO100LeaderConfig(port=LEADER_PORT, id=LEADER_ID)

    # Initialize the robot and teleoperator
    follower = SO100Follower(follower_config)
    leader = SO100Leader(leader_config)

    # NOTE: It is highly recommended to use the urdf in the SO-ARM100 repo:
    #   https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf
    follower_kinematics_solver = RobotKinematics(
        urdf_path="./SO101/so101_new_calib.urdf",
        target_frame_name="gripper_frame_link",
        joint_names=list(follower.bus.motors.keys()),
    )
    leader_kinematics_solver = RobotKinematics(
        urdf_path="./SO101/so101_new_calib.urdf",
        target_frame_name="gripper_frame_link",
        joint_names=list(leader.bus.motors.keys()),
    )

    # Build pipeline to convert follower joints to EE observation.
    follower_joints_to_ee = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[
            ForwardKinematicsJointsToEE(
                kinematics=follower_kinematics_solver, motor_names=list(follower.bus.motors.keys())
            ),
        ],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )

    # Build pipeline to convert leader joints to EE action.
    leader_joints_to_ee = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            ForwardKinematicsJointsToEE(
                kinematics=leader_kinematics_solver, motor_names=list(leader.bus.motors.keys())
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    # Build pipeline to convert EE action to follower joints (with safety bounds).
    ee_to_follower_joints = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            EEBoundsAndSafety(
                end_effector_bounds={"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},
                max_ee_step_m=0.10,
            ),
            InverseKinematicsEEToJoints(
                kinematics=follower_kinematics_solver,
                motor_names=list(follower.bus.motors.keys()),
                initial_guess_current_joints=True,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    use_videos = not args.save_images

    # Derive features from the pipelines so the on-disk schema matches exactly
    # what the pipelines produce at runtime.
    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=leader_joints_to_ee,
            initial_features=create_initial_features(action=leader.action_features),
            use_videos=use_videos,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=follower_joints_to_ee,
            initial_features=create_initial_features(observation=follower.observation_features),
            use_videos=use_videos,
        ),
    )

    dataset = open_dataset(args, dataset_features, follower.name, repo_id, dataset_path)

    # Connect the robot and teleoperator
    leader.connect()
    follower.connect()

    # Initialize the keyboard listener and rerun visualization
    listener, events = init_terminal_keyboard_listener()
    init_rerun(session_name="recording_so100_ee")

    try:
        if not leader.is_connected or not follower.is_connected:
            raise ValueError("Robot or teleop is not connected!")

        print("Starting record loop...")
        starting_episode_idx = dataset.num_episodes
        if starting_episode_idx > 0:
            print(f"Appending after {starting_episode_idx} existing episode(s).")
        episode_idx = 0
        while episode_idx < NUM_EPISODES and not events["stop_recording"]:
            if not wait_for_episode_start(events, episode_idx):
                break

            log_say(
                f"Recording new episode {episode_idx + 1} of {NUM_EPISODES} "
                f"(dataset episode {starting_episode_idx + episode_idx})"
            )

            # Main record loop
            record_loop(
                robot=follower,
                events=events,
                fps=FPS,
                teleop_action_processor=leader_joints_to_ee,
                robot_action_processor=ee_to_follower_joints,
                robot_observation_processor=follower_joints_to_ee,
                teleop=leader,
                dataset=dataset,
                control_time_s=EPISODE_TIME_SEC,
                single_task=task_prompt,
                display_data=True,
            )

            # Reset the environment if not stopping or re-recording
            if not events["stop_recording"] and (
                episode_idx < NUM_EPISODES - 1 or events["rerecord_episode"]
            ):
                log_say("Reset the environment")
                record_loop(
                    robot=follower,
                    events=events,
                    fps=FPS,
                    teleop_action_processor=leader_joints_to_ee,
                    robot_action_processor=ee_to_follower_joints,
                    robot_observation_processor=follower_joints_to_ee,
                    teleop=leader,
                    control_time_s=RESET_TIME_SEC,
                    single_task=task_prompt,
                    display_data=True,
                )

            if events["rerecord_episode"]:
                log_say("Re-recording episode")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue

            # Save episode
            dataset.save_episode()
            episode_idx += 1

    finally:
        # Clean up
        log_say("Stop recording")
        leader.disconnect()
        follower.disconnect()
        if listener is not None:
            listener.stop()

        dataset.finalize()
        # dataset.push_to_hub()


if __name__ == "__main__":
    main()
