#!/usr/bin/env python3
"""Find the deployment bottleneck: camera reads vs. policy inference.

No robot/motors touched. Times each camera's async_read and a batch of policy
inferences so you can see which one is capping the control loop.

Run with sudo (RealSense needs root on macOS):

    sudo -E PYTORCH_ENABLE_MPS_FALLBACK=1 $(which python) scripts/profile_deploy.py \
        --policy-path outputs/train/smolvla_so101_two_cubes_arrange/checkpoints/last/pretrained_model \
        --task "pick up two black cubes and arrange them on either side of the blue box, one on the left"
"""

import argparse
import statistics
import time

import torch

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.configs import PreTrainedConfig
from lerobot.policies.factory import get_policy_class
from lerobot.utils.device_utils import auto_select_torch_device, is_torch_device_available


def timeit(fn, n, warmup=3):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return ts


def report(name, ts):
    med = statistics.median(ts)
    mean = statistics.mean(ts)
    hz = 1000.0 / mean if mean else float("inf")
    print(
        f"  {name:22s} median={med:7.1f} ms  mean={mean:7.1f} ms  max={max(ts):7.1f} ms"
        f"  -> {hz:5.1f} Hz (amortized)"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy-path", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--wrist-cam", default="0")
    p.add_argument("--agent-cam-serial", default="234222301106")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--device", default=None)
    p.add_argument("--iters", type=int, default=30)
    args = p.parse_args()

    device = args.device
    if device is None or not is_torch_device_available(device):
        device = auto_select_torch_device().type
    print(f"Device: {device}\n")

    # --- Cameras ---
    configs = {
        "wrist": OpenCVCameraConfig(
            index_or_path=int(args.wrist_cam) if args.wrist_cam.isdigit() else args.wrist_cam,
            width=args.width, height=args.height, fps=args.fps, fourcc="MJPG",
        ),
        "agent_view": RealSenseCameraConfig(
            serial_number_or_name=args.agent_cam_serial,
            width=args.width, height=args.height, fps=args.fps,
        ),
    }
    cams = make_cameras_from_configs(configs)
    for cam in cams.values():
        cam.connect()

    print("Camera async_read latency:")
    for name, cam in cams.items():
        report(f"{name}.async_read", timeit(lambda c=cam: c.async_read(), args.iters))

    # --- Policy inference ---
    print("\nLoading policy...")
    cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    cfg.pretrained_path = args.policy_path
    cfg.device = device
    policy = get_policy_class(cfg.type).from_pretrained(args.policy_path, config=cfg)
    policy = policy.to(device)
    policy.eval()
    if hasattr(policy, "reset"):
        policy.reset()

    def make_obs():
        img = torch.rand(1, 3, args.height, args.width, device=device)
        return {
            "observation.state": torch.rand(1, 7, device=device),
            "observation.images.wrist": img,
            "observation.images.agent_view": img.clone(),
            "task": [args.task],
        }

    @torch.no_grad()
    def one_infer():
        policy.select_action(make_obs())

    print("\nPolicy select_action latency (chunk_size=50 => most calls pop from queue, every ~50th re-runs the model):")
    report("select_action", timeit(one_infer, max(args.iters, 55), warmup=2))

    for cam in cams.values():
        cam.disconnect()
    print("\nDONE. The component with the highest ms / lowest Hz is your bottleneck.")


if __name__ == "__main__":
    main()
