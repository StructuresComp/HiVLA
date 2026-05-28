#!/usr/bin/env python3
"""
NaVILA Bridge — Simple script-level NaVILA -> ROS bridge.

Reads ZED camera frames, runs NaVILA inference, publishes action strings
to /hivla/vla_action for run.py to execute.

This script runs independently from run.py. Replace this file later
with your real VLN module.

Usage:
  # Free memory first
  sudo sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

  # Run (in a separate terminal from run.py, from HiVLA root)
  python3 models/vla/navila_bridge.py --instruction "Go to the trash can"
"""

import sys
import os
import json
import math
import time
import copy
import threading
import argparse

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image as RosImage
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from PIL import Image as PILImage
from cv_bridge import CvBridge

_HIVLA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _HIVLA_ROOT not in sys.path:
    sys.path.insert(0, _HIVLA_ROOT)
from models.vla.inference import NaVILAInference
from models.vla.config import HIVLA_BEST_CONFIG_PATH, NAVILA_EVAL_PATH

# Import ReplanningManager directly by file path to avoid pulling in
# vlnce_baselines/__init__.py which imports Habitat trainers.
import importlib.util as _ilu
_replan_path = os.path.join(
    NAVILA_EVAL_PATH, "vlnce_baselines", "hivla", "8_replanning", "replanning_manager.py"
)
_spec = _ilu.spec_from_file_location("replanning_manager", _replan_path)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
ReplanningManager = _mod.ReplanningManager
ScanAction = _mod.ScanAction


class NaVILABridge(Node):
    def __init__(self, instruction: str):
        super().__init__("navila_bridge")
        self.instruction = instruction
        self.cv_bridge = CvBridge()
        self._lock = threading.Lock()

        # NaVILA model
        self.navila = NaVILAInference()
        self.get_logger().info("Loading NaVILA model...")
        self.navila.load_model()
        self.get_logger().info("NaVILA model loaded")

        # HiVLA replanning
        self._replan_mgr, self._target_heads = self._init_replanning()
        self._step = 0
        self._agent_state = {"x": 0.0, "y": 0.0, "heading": 0.0}
        self._rollback_in_progress = False
        self._rollback_memory = []   # frame_history to restore on rollback_done
        self._latest_frame = None    # most recent ZED frame (not yet added to history)

        # Separate callback groups so ZED sub and inference timer
        # can run concurrently in the MultiThreadedExecutor.
        self._cb_zed = MutuallyExclusiveCallbackGroup()
        self._cb_infer = MutuallyExclusiveCallbackGroup()
        self._cb_odom = MutuallyExclusiveCallbackGroup()
        self._cb_rb = MutuallyExclusiveCallbackGroup()
        self._cb_mode = MutuallyExclusiveCallbackGroup()

        odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )
        view_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.sub_odom = self.create_subscription(
            Odometry,
            "/odometry/local",
            self._odom_callback,
            odom_qos,
            callback_group=self._cb_odom,
        )

        # Subscribe to pre-processed 384x384 image from publish_resize_rgb.py
        self.sub_zed = self.create_subscription(
            RosImage,
            "/hivla/navila_view",
            self._zed_callback,
            view_qos,
            callback_group=self._cb_zed,
        )

        # Publisher: action strings -> run.py listens on this topic
        self.pub_action = self.create_publisher(String, "/hivla/vla_action", 10)

        # Subscribe to rollback_done reply from run.py
        self.sub_rollback_done = self.create_subscription(
            String,
            "/hivla/rollback_done",
            self._rollback_done_callback,
            10,
            callback_group=self._cb_rb,
        )

        # Subscribe to replanning mode toggle from run.py ("true"/"false")
        # Use TRANSIENT_LOCAL so we receive the latched value published at run.py startup.
        self._replan_mgr_saved = self._replan_mgr
        self._target_heads_saved = self._target_heads
        _latched_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.sub_replanning_mode = self.create_subscription(
            String,
            "/hivla/replanning_mode",
            self._replanning_mode_callback,
            _latched_qos,
            callback_group=self._cb_mode,
        )

        # Inference timer (runs every 0.5s check, actual inference takes ~4s)
        self.create_timer(0.5, self._inference_tick, callback_group=self._cb_infer)

        self._busy = False
        self._infer_count = 0
        self.get_logger().info(f"NaVILA Bridge ready. Instruction: '{instruction}'")

    def _init_replanning(self):
        """Load best_config.json and return (ReplanningManager, target_heads)."""
        try:
            with open(HIVLA_BEST_CONFIG_PATH) as f:
                cfg = json.load(f)
            heads = [tuple(h) for h in cfg["heads"]]
            mgr = ReplanningManager(
                idiag_heads=heads,
                natural_threshold=cfg["tau"],
                patience=cfg["P"],
                window=cfg["W"],
            )
            self.get_logger().info(
                f"Replanning loaded: K={cfg['K']} W={cfg['W']} P={cfg['P']} tau={cfg['tau']:.5f}"
            )
            return mgr, heads
        except Exception as e:
            self.get_logger().warn(f"Replanning config not loaded ({e}); replanning disabled")
            return None, []

    def _odom_callback(self, msg):
        """Track robot pose for replanning agent_state."""
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        self._agent_state = {"x": p.x, "y": p.y, "heading": yaw}

    def _zed_callback(self, msg):
        """Receives pre-processed 384x384 image from publish_resize_rgb.py."""
        try:
            cv_img = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            pil_img = PILImage.fromarray(cv_img)
            if self._latest_frame is None:
                self.get_logger().info(f"First frame: {cv_img.shape} dtype={cv_img.dtype}")
            with self._lock:
                self._latest_frame = pil_img  # cache only; added to history at inference time
        except Exception as e:
            self.get_logger().error(f"Frame error: {e}", throttle_duration_sec=5.0)

    def _replanning_mode_callback(self, msg):
        enabled = msg.data.strip().lower() == "true"
        if enabled:
            self._replan_mgr = self._replan_mgr_saved
            self._target_heads = self._target_heads_saved
        else:
            self._replan_mgr = None
            self._target_heads = []
        self.get_logger().info(f"Replanning {'enabled' if enabled else 'disabled'}")

    def _rollback_done_callback(self, msg):
        """run.py signals rollback complete; restore frame_history and resume."""
        if not self._rollback_in_progress:
            return
        if self._rollback_memory:
            with self._lock:
                self.navila.frame_history.clear()
                for frame in self._rollback_memory:
                    self.navila.frame_history.append(frame)
        self._rollback_memory = []
        self._rollback_in_progress = False
        self.get_logger().info("Rollback done: frame history restored, resuming inference")

    def _inference_tick(self):
        if self._busy or self._rollback_in_progress:
            return
        if self._latest_frame is None:
            return

        self._busy = True
        try:
            # Add current frame to history at inference time, then sample
            with self._lock:
                self.navila.add_frame(self._latest_frame)
                images = self.navila._sample_frames()
                current_history = list(self.navila.frame_history)

            self._infer_count += 1
            n_frames = len(images)
            sizes = [f"{img.size[0]}x{img.size[1]}" for img in images[:2]] + ["..."]
            self.get_logger().info(
                f"NaVILA: inference #{self._infer_count}, {n_frames} frames, "
                f"history={len(current_history)}, sizes={sizes}"
            )

            t0 = time.monotonic()
            frame_instr = None
            if self._replan_mgr is not None and self._target_heads:
                try:
                    output, frame_instr = self.navila.infer_with_frames_and_attention(
                        self.instruction, images, self._target_heads
                    )
                    if output is None:
                        import traceback as _tb
                        self.get_logger().error(
                            "infer_with_frames_and_attention returned None — "
                            "falling back to standard inference. "
                            "Check stdout for [NaVILA] Inference error details."
                        )
                        output = self.navila.infer_with_frames(self.instruction, images)
                        frame_instr = None
                except Exception as e:
                    import traceback as _tb
                    self.get_logger().error(
                        f"infer_with_frames_and_attention raised: {e}\n{_tb.format_exc()}"
                    )
                    output = self.navila.infer_with_frames(self.instruction, images)
                    frame_instr = None
            else:
                output = self.navila.infer_with_frames(self.instruction, images)
            elapsed = time.monotonic() - t0
            self.get_logger().info(f"NaVILA: {elapsed:.2f}s -> '{output}'")

            if output is None:
                return

            # Replanning step
            if self._replan_mgr is not None and frame_instr is not None:
                import numpy as np
                attn_maps = {k: v for k, v in frame_instr.items()}
                action_type = self._output_to_action_type(output)
                result = self._replan_mgr.update(
                    step=self._step,
                    agent_state=copy.deepcopy(self._agent_state),
                    visual_buffer=current_history,
                    attention_maps=attn_maps,
                    action_type=action_type,
                )
                self._step += 1

                if isinstance(result, ScanAction):
                    s = result.rollback_agent_state
                    self.get_logger().warn(
                        f"Anomaly triggered (score={result.anomaly_score:.4f}): "
                        f"rolling back to ({s['x']:.2f}, {s['y']:.2f}, "
                        f"heading={math.degrees(s['heading']):.1f}deg)"
                    )
                    self._rollback_in_progress = True
                    self._rollback_memory = copy.deepcopy(result.rollback_memory)
                    goto_msg = String()
                    goto_msg.data = (
                        f"goto ({s['x']:.4f}, {s['y']:.4f}, {s['heading']:.4f})"
                    )
                    self.pub_action.publish(goto_msg)
                    self._busy = False
                    return  # skip normal action publish; wait for rollback_done

                self.get_logger().info(
                    f"Replanning: {result.state} (score={result.anomaly_score:.4f})"
                )

            msg = String()
            msg.data = output
            self.pub_action.publish(msg)
        except Exception as e:
            self.get_logger().error(f"NaVILA inference error: {e}")
        finally:
            self._busy = False

    @staticmethod
    def _output_to_action_type(output: str) -> str:
        """Map NaVILA output string to ReplanningManager action type."""
        t = output.lower()
        if "turn left" in t:
            return "TURN_LEFT"
        if "turn right" in t:
            return "TURN_RIGHT"
        if "stop" in t:
            return "STOP"
        return "MOVE_FORWARD"


def main():
    parser = argparse.ArgumentParser(description="NaVILA Bridge")
    parser.add_argument("--instruction", type=str, required=True,
                        help="Navigation instruction for NaVILA")
    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = NaVILABridge(args.instruction)

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
