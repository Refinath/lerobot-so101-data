#!/usr/bin/env python3
"""Pre-episode SO101 arm health gate.

Detects the degraded/overload state that silently corrupted the campaign
(servos vibrate instead of moving after extended use). Run this BEFORE each
episode (or each env block). Exit code 0 = healthy, non-zero = power-cycle needed.

Checks:
  1. All 6 servos respond.
  2. Feetech `Status` error byte == 0 for every servo (non-zero = overload/overheat/etc. fault latched).
  3. Temperatures below a warn/fail threshold.
  4. Functional micro-move on wrist_roll: command a small move and confirm the
     encoder actually moved (catches "commanded but vibrating / no torque").

Run: sudo -E python scripts/check_arm.py --follower-port /dev/tty.usbmodem5B140303851
"""
import argparse
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--follower-port", default="/dev/tty.usbmodem5B140303851")
    p.add_argument("--follower-id", default="follower_arm")
    p.add_argument("--temp-warn", type=int, default=50)
    p.add_argument("--temp-fail", type=int, default=60)
    p.add_argument("--move-joint", default="wrist_roll")
    p.add_argument("--move-deg", type=float, default=5.0)
    p.add_argument("--no-move", action="store_true", help="skip the functional micro-move test")
    args = p.parse_args()

    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    cfg = SO101FollowerConfig(port=args.follower_port, id=args.follower_id, use_degrees=True)
    r = SO101Follower(cfg)
    problems = []
    try:
        r.bus.connect()
    except Exception as e:
        print(f"FAIL: motors not responding ({str(e)[:120]})")
        print("VERDICT: POWER-CYCLE the arm (unplug power ~10s, replug), then re-check.")
        sys.exit(2)

    motors = list(r.bus.motors)
    status = {m: r.bus.read("Status", m) for m in motors}
    temp = {m: r.bus.read("Present_Temperature", m) for m in motors}
    print("Status (0=ok):", status)
    print("Temp (C):     ", temp)

    faulted = [m for m, s in status.items() if s != 0]
    if faulted:
        problems.append(f"servo fault flag set on: {faulted}")
    hot = [m for m, t in temp.items() if t >= args.temp_fail]
    if hot:
        problems.append(f"overheating (>= {args.temp_fail}C): {hot}")
    warm = [m for m, t in temp.items() if args.temp_warn <= t < args.temp_fail]
    if warm:
        print(f"  [warn] warm servos (>= {args.temp_warn}C): {warm} — consider a cooldown")

    # Functional micro-move: does the arm actually move when commanded?
    if not args.no_move and not problems:
        j = args.move_joint
        try:
            start = r.bus.read("Present_Position", j)
            r.bus.write("Torque_Enable", j, 1)
            r.bus.write("Goal_Position", j, start + args.move_deg)
            time.sleep(0.6)
            moved = r.bus.read("Present_Position", j)
            r.bus.write("Goal_Position", j, start)   # move back
            time.sleep(0.6)
            r.bus.write("Torque_Enable", j, 0)
            delta = abs(moved - start)
            print(f"Micro-move {j}: commanded {args.move_deg:.1f}deg, actual moved {delta:.1f}deg")
            if delta < args.move_deg * 0.4:
                problems.append(f"{j} commanded {args.move_deg}deg but only moved {delta:.1f}deg "
                                f"(vibrating / degraded torque)")
        except Exception as e:
            problems.append(f"micro-move test error: {str(e)[:100]}")

    r.bus.disconnect()

    if problems:
        print("\nFAIL:")
        for pb in problems:
            print("  -", pb)
        print("VERDICT: POWER-CYCLE the arm, then re-check before running episodes.")
        sys.exit(1)
    print("\nVERDICT: HEALTHY — ok to run episodes.")
    sys.exit(0)


if __name__ == "__main__":
    main()
