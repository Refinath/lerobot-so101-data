# Deploying trained SmolVLA policies on the real SO101 (macOS)

This document explains, in detail, how we got a trained SmolVLA policy running on
the physical SO101 arm on an **Apple Silicon Mac (M1 Pro, macOS)**, how to find
the camera IDs (especially the Intel RealSense), every problem we hit and how we
fixed it, and how we sped the control loop up to the trained rate.

Read this top-to-bottom the first time. After that, the **Quick start** and the
**Troubleshooting** table are all you normally need.

---

## 0. TL;DR quick start

```bash
# From the repo root, with the robot + wrist cam + RealSense connected.
# .env line 1 must contain your macOS login password (used for sudo).
{ sed -n 1p .env; echo RUN; } | sudo -S -E env \
    PYTORCH_ENABLE_MPS_FALLBACK=1 HF_TOKEN=<your_hf_token> \
    $(which python) scripts/deploy_ee_real_robot.py \
      --policy-path outputs/train/<CHECKPOINT>/checkpoints/last/pretrained_model \
      --task "<EXACT task string from tasks.txt>" \
      --follower-port /dev/tty.usbmodem5B140303851 \
      --wrist-cam 0 \
      --agent-cam-serial 234222301106 \
      --fps 30 --duration 60
```

- `sudo` is **required** on macOS (the RealSense can only be opened as root — see §2).
- `--fps 30` runs the policy at the rate it was trained at (see §6). Use a lower
  value like `--fps 5` only to watch slow, stable motion.
- Replace `<CHECKPOINT>` and `--task` using the table in §4.

---

## 1. Hardware / software setup

| Thing | Value |
|---|---|
| Machine | Apple Silicon Mac (M1 Pro, 16-core GPU), macOS (arm64) |
| Python env | conda env at `/Users/refinath/work/envs/lerobot` (`$(which python)`) |
| lerobot | editable checkout at `/Users/refinath/work/lerobot` (v0.5.2) |
| Robot | SO101 follower, serial port `/dev/tty.usbmodem5B140303851` |
| Wrist camera | generic USB UVC webcam → **OpenCV index 0** |
| Agent-view camera | **Intel RealSense D455**, serial **234222301106** |
| Inference device | **mps** (Metal) — the checkpoint hard-codes `cuda`, which does not exist here |

The policy (`smolvla`) consumes **two 640×480 RGB image streams** named
`observation.images.wrist` and `observation.images.agent_view`, plus a 7-D EE
state, and outputs a 7-D end-effector action
(`ee.x, ee.y, ee.z, ee.wx, ee.wy, ee.wz, ee.gripper_pos`).

---

## 2. Finding the camera IDs

There are **two** cameras and they are found in **completely different ways**.

### 2a. The wrist camera (OpenCV / plain USB webcam)

The wrist cam is a normal UVC webcam addressed by an **integer OpenCV index**.
On macOS these indices are assigned by AVFoundation and **reshuffle whenever you
plug/unplug any USB device** — so re-verify it each session.

Probe the indices and save a frame from each working one:

```python
import cv2, time
for i in range(4):
    cap = cv2.VideoCapture(i)
    if not cap.isOpened():
        cap.release(); continue
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    time.sleep(0.5)
    ok = 0; f = None
    for _ in range(6):
        r, f = cap.read(); ok += r; time.sleep(0.05)
    if ok and f is not None:
        cv2.imwrite(f"/tmp/cam_{i}.jpg", f)      # open these and look
    print(f"index {i}: reads_ok={ok}/6")
    cap.release()
```

Then open the `/tmp/cam_*.jpg` files and pick the index that shows the **gripper
close-up** (the SO101's teal fingers). In our setup that was **index 0**.

Notes we learned the hard way:
- The **RealSense does NOT deliver frames through OpenCV on macOS** — it opens but
  returns 0 frames at every resolution/format. So a working OpenCV index that
  shows nothing / times out is usually the RealSense; skip it.
- The Mac's **built-in FaceTime camera** also shows up (it points at your face) —
  don't pick that one.

### 2b. The agent-view camera (Intel RealSense — this is the important one)

The RealSense will **not** work through OpenCV on macOS. It must be driven by
Intel's `librealsense` backend, and it is addressed by its **serial number**, not
an index — so it is stable no matter how many cameras you attach.

**Step 1 — install the backend** (Apple Silicon has no official wheel; use the
community macOS build, plus Homebrew's librealsense for the CLI tools):

```bash
pip install pyrealsense2-macosx        # provides `import pyrealsense2`
brew install librealsense              # provides rs-enumerate-devices, etc.
```

**Step 2 — find the serial number.** On macOS, `librealsense` must *seize* the
camera from the kernel's UVC driver, which **requires root**. So:

```bash
sudo rs-enumerate-devices
```

Look for the `Serial Number` field, e.g.:

```
Name              : Intel RealSense D455
Serial Number     : 234222301106
Firmware Version  : 5.13.1.53
Product Id        : 0B5C
Usb Type Descriptor : 3.2
...
Stream Profiles supported by RGB Camera
    Color   640x480   RGB8   @ 60/30/15/5 Hz     <- this is what the policy uses
```

That `Serial Number` is what you pass to `--agent-cam-serial`.

**Why sudo?** Without root, `rs-enumerate-devices` prints *"No device detected"*
and `librealsense` reports 0 devices even though the camera is clearly enumerated
at the USB level (visible in `ioreg -p IOUSB`). macOS binds the RealSense's video
interfaces to its own UVC driver, and only a **root** process can seize them back.
There is **no udev-style permission fix on macOS** (that only exists on Linux —
Intel's `99-realsense-libusb.rules`). Granting the terminal "Camera" permission
does *not* help, because that governs AVFoundation, not the raw USB access
librealsense needs. Practical consequence: **every RealSense run on this Mac must
be `sudo`.** To avoid typing the password each time, either prime it once with
`sudo -v`, or add a scoped `NOPASSWD` entry in `/etc/sudoers.d/`.

**Sanity-check the wiring before touching the robot** with the no-motor preflight:

```bash
sudo -E $(which python) scripts/preflight_cameras.py \
    --wrist-cam 0 --agent-cam-serial 234222301106 --out /tmp/preflight
# then open /tmp/preflight/wrist.jpg and /tmp/preflight/agent_view.jpg
```

`wrist.jpg` should show the gripper; `agent_view.jpg` should show the whole
workspace (arm + cubes + blue box). If they're swapped or wrong, fix the index
before running the policy.

---

## 3. What the deploy script does (and what we changed)

`scripts/deploy_ee_real_robot.py` wraps lerobot's rollout engine with the SO101
forward/inverse-kinematics processors so the joint-space robot talks to the
EE-space policy. Changes we made for macOS reliability:

- **`--agent-cam-serial`** — routes `agent_view` through the RealSense
  (`librealsense`) backend instead of OpenCV. Required on macOS.
- **`--wrist-cam`** now accepts an **integer index** (macOS) as well as a Linux
  `/dev/videoN` path.
- **Wrist camera uses auto video format (no MJPG) + 3 s warmup.** Forcing MJPG
  fails to set on macOS and could leave the camera unable to deliver its first
  frame (connect-time `TimeoutError`). `--wrist-fourcc` can override if needed.
- **RealSense warmup raised to 2 s** — 1 s sometimes timed out on the first frame;
  5 s increased exposure to an intermittent backend crash (see §7). 2 s was the
  sweet spot.
- **Device auto-select** — the checkpoint records `device="cuda"`, and lerobot's
  rollout trusted that string without checking availability. We force a real,
  available device, which resolves to **mps** here.

Two more no-motor helper scripts were added:
- `scripts/preflight_cameras.py` — connect both cameras, save a frame from each.
- `scripts/diagnose_policy.py` — read the robot + cameras, run the policy, and
  print the EE state, predicted action, IK targets, in-distribution checks, and
  **per-step timing** — all **without sending any motor command**.
- `scripts/profile_deploy.py` — time camera reads vs. policy inference in isolation.

---

## 4. Checkpoint ↔ task ↔ scene

The `--task` string **must match the training instruction exactly** (SmolVLA is
language-conditioned). The exact strings live in `tasks.txt`:

| Checkpoint (`outputs/train/…`) | Exact `--task` string | Scene must contain |
|---|---|---|
| `smolvla_so101_cube_to_drawer` | `Pick up the black cube on top of the blue box and place it inside the drawer` | black cube on blue box, drawer |
| `smolvla_so101_two_cubes_arrange` | `Pick up two black cubes and arrange them on either side of the blue box (one left, one right)` | 2 black cubes, blue box |
| `smolvla_so101_push_cube` | `Push the black cube farthest from the drawer toward the blue box` | cubes, drawer, blue box |
| `smolvla_so101_bowl_yellow_rectangle` | `Pick up the black bowl in the drawer and place it on the yellow rectangle` | bowl in drawer, yellow rectangle |

The physical table **must** be arranged like the training scene or the policy
cannot succeed.

---

## 5. Pre-flight checklist (every session)

1. **Motors** — power the arm. If a run fails with `Failed to write 'Lock' on
   id_=N … There is no status packet!`, that servo dropped off the bus:
   power-cycle the arm and reseat the daisy-chain cable near that motor.
2. **Wrist camera index** — re-verify (see §2a); it changes when USB devices move.
3. **RealSense serial** — stable, but confirm with `sudo rs-enumerate-devices` if
   you swapped cameras.
4. **Scene + start pose** — arrange the table for the task, and put the arm in a
   **raised, mid-workspace "ready" pose**, similar to where your teleop episodes
   started. Do **not** start from a tucked/edge pose — that is out-of-distribution
   and the first policy command will lurch.
5. **Keep the USB bus lean** — extra RealSense cameras add contention and raise
   the chance of the intermittent connect crash (§7).

---

## 6. Speeding up the control loop (the important part)

### The symptom
The loop printed warnings like *"Record loop is running slower (4.3 Hz) than the
target FPS (30 Hz) … 1) Camera FPS 2) Policy inference 3) CPU starvation"*, and
the arm approached objects but **missed the grasp**. The same warning appears on
the Linux+GPU box too.

### How we found the real bottleneck
We added **per-step timing** to `scripts/diagnose_policy.py` (no motor motion) and
measured each phase over many steps:

```
get_observation (camera read):   1.0 ms      <- NOT the bottleneck
obs_processor  (forward kine):    0.0 ms
build_frame:                      0.0 ms
get_action     (inference):       3.4 ms median  BUT  416 ms max
TOTAL / step:                     4.4 ms  ->  ~227 Hz ceiling
```

**Key insight:** cameras are *not* the problem (1 ms). The loop can do ~227 Hz.
SmolVLA predicts **50 actions per inference** (`chunk_size = n_action_steps = 50`),
so 49 of every 50 steps just pop a pre-computed action (~3 ms) and only the **50th
step re-runs the model synchronously and freezes for ~416 ms**.

That single periodic freeze is what trips the warning. It is *not* a uniform
slowdown — and it is *not* the camera or the GPU. On the Linux box the same
periodic synchronous re-plan is what shows up as "CPU starvation".

### The fix that worked: run at `--fps 30`
Because per-step work is only 4.4 ms, the loop happily sustains 30 Hz **between**
the re-plan hitches. Evidence: at `--fps 30` over 20 s we saw only **~11 warnings**
(one per action chunk ≈ every 1.7 s). If the loop were genuinely stuck at ~4 Hz,
*every* step would warn (600+). So between hitches it runs at the **full 30 Hz —
the trained rate** — with a brief ~220 ms hitch every ~1.7 s. Switching from
`--fps 5` to `--fps 30` made the arm visibly much faster and more decisive.

### Further speedups (optional)
- **RTC inference** (`RTCInferenceConfig`, "Real-Time Chunking") — overlaps the
  re-plan with motion so there is **no hitch at all**. This is the proper
  real-time fix and also cures the Linux "CPU starvation". Not yet wired into the
  deploy script (would be a `--rtc` flag).
- **Fewer flow-matching steps** — SmolVLA's 416 ms is `num_steps = 10` denoising
  steps. Dropping to ~5 roughly halves each hitch, at a small quality cost.
- **Linux + GPU** for the real grasp-success evaluation — faster inference shrinks
  the hitch further and hits a clean 30 Hz most easily.

### What is NOT the bottleneck (so don't chase these)
- Camera FPS / the RealSense — reads are 1 ms (frames are served from a background
  thread; you read the latest cached frame).
- The GPU — inference is chunked; a faster GPU only shrinks the periodic hitch.

---

## 7. Troubleshooting

| Symptom in the log | Cause | Fix |
|---|---|---|
| `No device detected` / 0 RealSense devices | Not running as root | Run with `sudo` (see §2b) |
| `Timed out waiting for frame from camera OpenCVCamera(0)` | Wrong wrist index, or MJPG left the cam in a bad state | Re-verify index (§2a); the script already avoids MJPG + uses 3 s warmup |
| `Timed out … RealSenseCamera(...) after 1000 ms` | RealSense first-frame slow | RealSense warmup is 2 s; just retry |
| `EXIT=139` (segfault) right after a camera connects | `cv2` and `av` each bundle their own `libavdevice` (duplicate Obj-C classes → intermittent crash) | Can't remove either (both are hard deps). It's a ~50/50 dice roll — **just retry**; keep only needed cameras attached |
| `Failed to write 'Lock' on id_=N … no status packet` | A servo dropped off the motor bus | Power-cycle the arm, reseat the daisy-chain cable near that motor |
| `Device 'cuda' is not available. Switching to 'mps'` | Checkpoint hard-codes cuda | Harmless — the script forces a valid device |
| Arm shakes / lurches, no task progress | Out-of-distribution **start pose** (e.g. tucked at workspace edge, `ee.z` at the training minimum) | Start from a raised, mid-workspace ready pose (§5) |
| Arm approaches the object but misses the grasp | Control rate too low for the timing-critical grasp | Use `--fps 30` (§6); for real success rate, run on Linux+GPU |
| Warning every ~1–2 s but motion is smooth otherwise | Normal — the periodic re-plan hitch | Optional: RTC inference or fewer flow steps (§6) |

---

## 8. Verified healthy (ruled out as causes)

During debugging we confirmed, with `scripts/diagnose_policy.py` (no motor motion):
- **Cameras are correct and not swapped** — verified visually and via the recorded
  per-camera brightness stats (wrist ≈ 0.65, agent_view ≈ 0.51) matching the live
  frames; also proven by the arm moving *toward* the correct object.
- **FK/IK is self-consistent** — IK of the current EE pose reproduces the current
  joint angles to within ~2°, so there is no calibration/URDF/rotation bug.
- **Policy outputs are in-distribution and coherent** — predicted actions fall
  inside the training action ranges and form sensible reach/grasp motions.

So the only remaining gap is **grasp precision**, which is a control-rate /
policy-training matter, not a setup bug. Approaching the correct object (the hard
perceptual part) already works.
