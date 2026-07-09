#!/usr/bin/env python3
"""Pre-flight camera check for the SO101 deployment (no robot/motors involved).

Connects the wrist (OpenCV) and agent_view (RealSense) cameras exactly the way
deploy_ee_real_robot.py will, reads one frame from each, prints the shapes, and
saves JPEGs so you can eyeball that each stream points where it should.

On macOS the RealSense needs root to seize the device from the kernel UVC driver,
so run this with sudo, e.g.:

    sudo $(which python) scripts/preflight_cameras.py \
        --wrist-cam 0 --agent-cam-serial 234222301106 --out /tmp/preflight
"""

import argparse
import os

import cv2

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.cameras.utils import make_cameras_from_configs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wrist-cam", default="0")
    p.add_argument("--agent-cam-serial", default="234222301106")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--out", default="/tmp/preflight")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    configs = {
        "wrist": OpenCVCameraConfig(
            index_or_path=int(args.wrist_cam) if args.wrist_cam.isdigit() else args.wrist_cam,
            width=args.width,
            height=args.height,
            fps=args.fps,
            fourcc="MJPG",
        ),
        "agent_view": RealSenseCameraConfig(
            serial_number_or_name=args.agent_cam_serial,
            width=args.width,
            height=args.height,
            fps=args.fps,
        ),
    }

    cameras = make_cameras_from_configs(configs)
    for name, cam in cameras.items():
        print(f"[{name}] connecting...")
        cam.connect()
        frame = cam.read()
        print(f"[{name}] frame shape={frame.shape} dtype={frame.dtype}")
        # lerobot returns RGB; convert to BGR for cv2.imwrite.
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if frame.ndim == 3 else frame
        path = os.path.join(args.out, f"{name}.jpg")
        cv2.imwrite(path, bgr)
        print(f"[{name}] saved {path}")
        cam.disconnect()

    print("PREFLIGHT OK — both cameras connected and captured.")


if __name__ == "__main__":
    main()
