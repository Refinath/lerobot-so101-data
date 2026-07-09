#!/usr/bin/env python3
"""Instrument the real control loop briefly (drives the arm ~6s) and log commanded
joint targets vs observed positions, to tell whether the vibrating is command-side
(policy/IK oscillation) or execution-side (mechanical/bus)."""
import argparse, os, time, numpy as np
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p=argparse.ArgumentParser()
p.add_argument("--policy-path",required=True); p.add_argument("--task",required=True)
p.add_argument("--follower-port",default="/dev/tty.usbmodem5B140303851")
p.add_argument("--wrist-cam",default="1"); p.add_argument("--agent-cam-opencv",default="0")
p.add_argument("--seconds",type=float,default=6.0); p.add_argument("--fps",type=int,default=30)
p.add_argument("--urdf",default=os.path.join(ROOT,"SO101","so101_new_calib.urdf"))
a=p.parse_args()
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.configs import PreTrainedConfig
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import (RobotProcessorPipeline, observation_to_transition, robot_action_observation_to_transition, transition_to_observation, transition_to_robot_action)
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.robots.so_follower.robot_kinematic_processor import ForwardKinematicsJointsToEE, InverseKinematicsEEToJoints
from lerobot.rollout import BaseStrategyConfig, RolloutConfig, build_rollout_context
from lerobot.rollout.inference import SyncInferenceConfig
from lerobot.rollout.strategies.base import BaseStrategy
from lerobot.rollout.strategies.core import send_next_action
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.constants import ACTION
from lerobot.utils.device_utils import auto_select_torch_device
from lerobot.utils.process import ProcessSignalHandler
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging
init_logging()
dev=auto_select_torch_device().type
cams={"wrist":OpenCVCameraConfig(index_or_path=int(a.wrist_cam),width=640,height=480,fps=a.fps,fourcc=None,warmup_s=3),
      "agent_view":OpenCVCameraConfig(index_or_path=int(a.agent_cam_opencv),width=640,height=480,fps=a.fps,fourcc=None,warmup_s=3)}
rc=SO101FollowerConfig(port=a.follower_port,id="follower_arm",cameras=cams,use_degrees=True)
mn=list(SO101Follower(rc).bus.motors.keys())
kin=RobotKinematics(urdf_path=a.urdf,target_frame_name="gripper_frame_link",joint_names=mn)
op=RobotProcessorPipeline[RobotObservation,RobotObservation](steps=[ForwardKinematicsJointsToEE(kinematics=kin,motor_names=mn)],to_transition=observation_to_transition,to_output=transition_to_observation)
tp=RobotProcessorPipeline[tuple[RobotAction,RobotObservation],RobotAction](steps=[ForwardKinematicsJointsToEE(kinematics=kin,motor_names=mn)],to_transition=robot_action_observation_to_transition,to_output=transition_to_robot_action)
ap=RobotProcessorPipeline[tuple[RobotAction,RobotObservation],RobotAction](steps=[InverseKinematicsEEToJoints(kinematics=kin,motor_names=mn,initial_guess_current_joints=True)],to_transition=robot_action_observation_to_transition,to_output=transition_to_robot_action)
pc=PreTrainedConfig.from_pretrained(a.policy_path); pc.pretrained_path=a.policy_path; pc.device=dev
cfg=RolloutConfig(robot=rc,policy=pc,strategy=BaseStrategyConfig(),inference=SyncInferenceConfig(),fps=a.fps,duration=a.seconds,task=a.task,device=dev,return_to_initial_position=False)
sh=ProcessSignalHandler(use_threads=True)
ctx=build_rollout_context(cfg,sh.shutdown_event,teleop_action_processor=tp,robot_action_processor=ap,robot_observation_processor=op)
keys=list(ctx.data.dataset_features[ACTION]["names"]); ctx.data.ordered_action_keys[:]=keys; ctx.policy.inference._ordered_action_keys=keys
robot=ctx.hardware.robot_wrapper; interp=ctx.hardware.__dict__.get("interpolator")
rows=[]
class S(BaseStrategy):
    def run(self,ctx):
        cfg=ctx.runtime.cfg; robot=ctx.hardware.robot_wrapper; interp=self._interpolator
        ci=interp.get_control_interval(cfg.fps); t0=time.perf_counter(); self._engine.resume()
        while not ctx.runtime.shutdown_event.is_set():
            ls=time.perf_counter()
            if time.perf_counter()-t0>=cfg.duration: break
            obs=robot.get_observation(); obsp=self._process_observation_and_notify(ctx.processors,obs)
            if self._handle_warmup(cfg.use_torch_compile,ls,ci): continue
            ad=send_next_action(obsp,obs,ctx,interp)
            if ad is not None:
                cmd=ctx.processors.robot_action_processor((ad,obs))
                row={"t":round(time.perf_counter()-t0,3)}
                for k in mn:
                    row[f"cmd_{k}"]=round(float(cmd.get(f"{k}.pos",float('nan'))),2)
                    row[f"obs_{k}"]=round(float(obs.get(f"{k}.pos",float('nan'))),2)
                rows.append(row)
            dt=time.perf_counter()-ls
            if (st:=ci-dt)>0: precise_sleep(st)
strat=S(cfg.strategy); strat.setup(ctx)
try: strat.run(ctx)
finally: strat.teardown(ctx)
print(f"\nlogged {len(rows)} steps over {a.seconds}s -> {len(rows)/a.seconds:.1f} Hz")
def reversals(seq):
    d=np.diff(seq); s=np.sign(d[np.abs(d)>0.5])
    return int(np.sum(s[1:]!=s[:-1])) if len(s)>1 else 0
print("\nper-joint: cmd_range  cmd_reversals  obs_range  obs_reversals  (reversals=oscillation count)")
for k in mn:
    cmd=[r[f"cmd_{k}"] for r in rows]; obs=[r[f"obs_{k}"] for r in rows]
    print(f"  {k:14s} cmd[{min(cmd):.0f},{max(cmd):.0f}] rev={reversals(cmd):3d}   obs[{min(obs):.0f},{max(obs):.0f}] rev={reversals(obs):3d}")
