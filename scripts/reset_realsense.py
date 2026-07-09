#!/usr/bin/env python3
"""Hardware-reset the RealSense to restore its COLOR stream to AVFoundation/OpenCV.

Background: on macOS the kernel UVC driver claims the RealSense COLOR interface, so
librealsense can't STREAM color (only depth/IR) — color must be read via OpenCV.
But a cascade of librealsense segfaults can leave the device exposing only IR to
AVFoundation. A hardware_reset restores the color stream. This uses the from-source
librealsense 2.57.6 build (which, unlike the pip wheel, doesn't segfault and has a
working hardware_reset). Run with sudo at session start, or whenever
find_agent_cam.py can't find a COLOR stream.
"""
import sys, time
sys.path.insert(0, "/Users/refinath/work/librealsense_257/build/Release")
import pyrealsense2 as rs

ctx = rs.context()
devs = ctx.query_devices()
print(f"RealSense devices: {len(devs)}")
for d in devs:
    sn = d.get_info(rs.camera_info.serial_number)
    print(f"  hardware_reset -> {d.get_info(rs.camera_info.name)} ({sn})")
    d.hardware_reset()
print("Reset issued. Wait ~10s for re-enumeration, then run: python scripts/find_agent_cam.py")
