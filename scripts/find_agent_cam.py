#!/usr/bin/env python3
"""Identify which OpenCV index is the RealSense COLOR stream.

The RealSense exposes color/IR/depth as separate AVFoundation devices whose
OpenCV indices shuffle per USB port. The COLOR stream is the one with real
color (differing B/G/R channels) and reasonable brightness; IR is grayscale
(often dotted by the emitter); depth is near-black. Point the camera at a lit,
colorful scene and run this to get the color index for --agent-cam-opencv.
"""
import cv2, time, numpy as np

best = None
for i in range(6):
    cap = cv2.VideoCapture(i)
    if not cap.isOpened():
        cap.release(); continue
    cap.set(3, 640); cap.set(4, 480); time.sleep(1.0)
    f = None
    for _ in range(15):
        r, f = cap.read(); time.sleep(0.03)
    cap.release()
    if f is None:
        print(f"idx {i}: no frame"); continue
    b, g, r = f[:, :, 0].mean(), f[:, :, 1].mean(), f[:, :, 2].mean()
    spread = float(np.std([b, g, r])); bright = float(f.mean())
    # Channel spread is the real color-vs-grayscale discriminator; IR/depth are
    # grayscale (B==G==R -> spread ~0) even when bright, while the color stream has
    # differing channels even in a dark room. Do NOT gate on brightness.
    is_color = spread > 2.0
    tag = "COLOR" if is_color else "grayscale/IR/depth"
    note = "  (dark - will brighten in a lit room)" if is_color and bright < 30 else ""
    print(f"idx {i}: bright={bright:.0f} channel_spread={spread:.1f} -> {tag}{note}")
    if is_color and (best is None or spread > best[1]):
        best = (i, spread)
print()
if best:
    print(f"==> COLOR stream is OpenCV index {best[0]}  (use: --agent-cam-opencv {best[0]})")
else:
    print("==> No clear COLOR stream found. Make sure the scene is lit and colorful, then retry.")
