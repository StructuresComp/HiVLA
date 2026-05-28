import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import numpy as np
import torch
import argparse
import re
import os
import math
import time
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


class ActionSmoother:
    def __init__(self, max_v=0.5, max_w=0.5):
        self.max_v = max_v
        self.max_w = max_w

        self.max_acc_v = 1.0   # m/s^2 (symmetric for forward/backward)
        self.max_acc_w = 5.0   # rad/s^2

        self.last_v = 0.0
        self.last_w = 0.0

    def process(self, target_v, target_w, dt):
        # 1. Velocity Saturation
        target_v = np.clip(target_v, -self.max_v, self.max_v)
        target_w = np.clip(target_w, -self.max_w, self.max_w)

        # 2. Acceleration Clamping
        if dt < 0.001:
            dt = 0.1

        current_acc = (target_v - self.last_v) / dt
        if abs(current_acc) > self.max_acc_v:
            current_acc = np.clip(current_acc, -self.max_acc_v, self.max_acc_v)
            target_v = self.last_v + current_acc * dt

        acc_w = (target_w - self.last_w) / dt
        if abs(acc_w) > self.max_acc_w:
            acc_w = np.clip(acc_w, -self.max_acc_w, self.max_acc_w)
            target_w = self.last_w + acc_w * dt

        # 3. Deadzone Suppression
        if abs(target_v) < 0.02: target_v = 0.0
        if abs(target_w) < 0.02: target_w = 0.0

        self.last_v = target_v
        self.last_w = target_w

        return target_v, target_w


def parse_instruction_for_path(instruction_str: str) -> list[tuple[float, float]] | None:
    pattern = r'[\(\[\{]\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*[\)\]\}]'
    matches = re.findall(pattern, instruction_str)

    if not matches:
        return None

    try:
        path = [(float(x), float(y)) for x, y in matches]
        return path
    except Exception as e:
        print(f"Error converting parsed coordinates to float: {e}")
        return None


from models.policy.config import DEVICE, GRID_RES
from models.policy.inference import RLNavigator


class HiVLARunner(Node):
    def __init__(self, instruction_str: str):
        super().__init__('hivla_runner_node')

        self.instruction = instruction_str
        self.get_logger().info(f"Instruction Received: '{self.instruction}'")
        self.callback_group = ReentrantCallbackGroup()
        self.pub_debug_log = self.create_publisher(String, '/hivla/debug_log', 10)
        self.pub_navila_log = self.create_publisher(String, '/hivla/navila_log', 10)
        _latched_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.pub_replanning_mode = self.create_publisher(String, '/hivla/replanning_mode', _latched_qos)
        self._last_vla_action = ""
        self._last_cmd_v = 0.0
        self._last_cmd_w = 0.0

        odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE
        )

        self.VLA_USE_RL = True  # True: NaVILA -> RL -> robot, False: NaVILA -> robot (open-loop)
        self.REPLANNING_ENABLED = True  # True: path deviation detection, False: pure NaVILA
        # Publish immediately so navila_bridge gets the mode as soon as it subscribes
        self.pub_replanning_mode.publish(String(data="true" if self.REPLANNING_ENABLED else "false"))
        self.WAYPOINT_THRESHOLD = 0.15   # waypoint nav: 50cm
        self.ROLLBACK_THRESHOLD = 0.10  # rollback: 10cm
        self.path_waypoints = []
        self.current_waypoint_index = 0
        self.local_goal = None

        self.smoother = ActionSmoother(max_v=0.5, max_w=0.5)

        self.last_control_time = self.get_clock().now()
        self.CONTROL_PERIOD = 0.1  # 10Hz

        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.is_pose_initialized = False

        # In-place turn state (open-loop, duration-based)
        self._is_turning = False
        self._turn_speed = 0.5         # rad/s
        self._turn_direction = 0
        self._turn_end_time = 0.0

        # Open-loop forward state
        self._is_forwarding = False
        self._fwd_speed = 0.3          # m/s
        self._fwd_end_time = 0.0

        # Rollback state: RL drives to checkpoint, then heading correction via w
        self._is_rollback = False
        self._rollback_target_heading = 0.0
        self._is_heading_correction = False

        # Once stop received, ignore all future VLA actions
        self._vla_stopped = False

        parsed_path = parse_instruction_for_path(self.instruction)

        if parsed_path and len(parsed_path) > 0:
            self.path_waypoints = parsed_path
            self.is_language_instruction = False
            self.get_logger().info(f"✅ Path Parsed: {len(self.path_waypoints)} waypoints found.")
            self._set_current_goal_global()
        else:
            self.local_goal = None
            self.is_language_instruction = True
            self.get_logger().info("Goal Type: Language/VLA. Waiting for VLA action...")
            mode = "true" if self.REPLANNING_ENABLED else "false"
            self.pub_replanning_mode.publish(String(data=mode))
            self.get_logger().info(f"Replanning mode: {'enabled' if self.REPLANNING_ENABLED else 'disabled'}")

        base_dir = os.path.dirname(os.path.abspath(__file__))
        checkpoint_path = os.path.join(base_dir, "models", "policy", "checkpoints", "rl_checkpoints.pt")

        self.get_logger().info(f"Loading RL Policy from: {checkpoint_path}")
        self.navigator = RLNavigator(checkpoint_path)

        self.cv_bridge = CvBridge()

        self.sub_odom = self.create_subscription(
            Odometry,
            '/odometry/local',
            self.odom_callback,
            odom_qos,
            callback_group=self.callback_group
        )
        self.sub_costmap = self.create_subscription(
            Image,
            '/local_costmap/raw',
            self.costmap_callback,
            odom_qos,
            callback_group=self.callback_group
        )

        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_rollback_reply = self.create_publisher(String, '/hivla/rollback_done', 10)

        self.sub_vla_action = self.create_subscription(
            String, '/hivla/vla_action', self.vla_action_callback,
            10, callback_group=self.callback_group
        )

        self.create_timer(0.1, self.vla_control_tick, callback_group=self.callback_group)

        self.get_logger().info("🚀 HiVLA Runner Started (Live Control Enabled)")

    def _set_current_goal_global(self):
        if self.current_waypoint_index < len(self.path_waypoints):
            gx, gy = self.path_waypoints[self.current_waypoint_index]
            self.local_goal = torch.tensor([[gx, gy]], device=DEVICE, dtype=torch.float32)
            self.get_logger().info(f"➡️ Global Goal Updated ({self.current_waypoint_index+1}/{len(self.path_waypoints)}): {self.local_goal.tolist()}")
        else:
            self.local_goal = None
            self.get_logger().info("🎉 All waypoints reached!")
            self.pub_cmd_vel.publish(Twist())
            self.create_timer(0.5, lambda: rclpy.try_shutdown(), callback_group=self.callback_group)

    def get_yaw_from_quaternion(self, q):
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def odom_callback(self, msg):
        """Updates robot pose from odometry. Runs independently of the control loop."""
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        self.robot_yaw = self.get_yaw_from_quaternion(msg.pose.pose.orientation)
        self.is_pose_initialized = True

    def check_and_update_goal(self, dist):
        if self._is_rollback:
            if dist < self.ROLLBACK_THRESHOLD:
                self.local_goal = None
                self.pub_cmd_vel.publish(Twist())
                self._start_heading_correction()
            return
        if dist < self.WAYPOINT_THRESHOLD:
            self.get_logger().info(f"📍 Waypoint {self.current_waypoint_index+1} Reached! (Dist: {dist:.2f}m)")
            self.current_waypoint_index += 1
            self._set_current_goal_global()

    def _start_heading_correction(self):
        heading_error = self._normalize_angle(
            self._rollback_target_heading - self.robot_yaw
        )
        if abs(heading_error) < 0.05:  # ~3 degrees
            self._finish_rollback()
            return
        self._turn_direction = 1 if heading_error > 0 else -1
        self._is_heading_correction = True
        self._is_turning = False  # use heading correction path, not duration path
        self.get_logger().info(
            f"Heading correction: error={math.degrees(heading_error):.1f}deg"
        )

    def _finish_rollback(self):
        self._is_rollback = False
        self._is_heading_correction = False
        self.pub_cmd_vel.publish(Twist())
        self.pub_rollback_reply.publish(String(data="rollback_done"))
        self.get_logger().info("Rollback complete, signalling navila_bridge")

    def vla_control_tick(self):
        """10Hz timer: handles open-loop turn/forward and feedback heading correction."""
        # Feedback-based heading correction (rollback phase 2)
        if self._is_heading_correction:
            if not self.is_pose_initialized:
                return
            heading_error = self._normalize_angle(
                self._rollback_target_heading - self.robot_yaw
            )
            if abs(heading_error) < 0.05:  # ~3 degrees
                self._finish_rollback()
            else:
                self._turn_direction = 1 if heading_error > 0 else -1
                cmd = Twist()
                cmd.angular.z = float(self._turn_direction * self._turn_speed)
                self.pub_cmd_vel.publish(cmd)
            return

        # Open-loop duration-based turn (VLA turn commands)
        if self._is_turning:
            if time.monotonic() >= self._turn_end_time:
                self._is_turning = False
                self.pub_cmd_vel.publish(Twist())
                self.get_logger().info("VLA turn completed")
            else:
                cmd = Twist()
                cmd.angular.z = float(self._turn_direction * self._turn_speed)
                self.pub_cmd_vel.publish(cmd)
            return

        if self._is_forwarding:
            if time.monotonic() >= self._fwd_end_time:
                self._is_forwarding = False
                self.pub_cmd_vel.publish(Twist())
                self.get_logger().info("VLA forward completed (open-loop)")
            else:
                cmd = Twist()
                cmd.linear.x = float(self._fwd_speed)
                self.pub_cmd_vel.publish(cmd)
            return

        # Bottom-center overlay: always publish at 10Hz
        if self._last_vla_action:
            goto_m = re.match(
                r'goto\s*\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)',
                self._last_vla_action
            )
            if goto_m:
                ox, oy, oh = float(goto_m.group(1)), float(goto_m.group(2)), float(goto_m.group(3))
                navila_msg = (
                    f"Recovery: rolling back to ({ox:.2f}, {oy:.2f}, {math.degrees(oh):.1f}°)<br>"
                    f"v = {self._last_cmd_v:.2f} m/s &nbsp; w = {self._last_cmd_w:.2f} rad/s"
                )
            else:
                navila_msg = (
                    f"VLA output: {self._last_vla_action}<br>"
                    f"v = {self._last_cmd_v:.2f} m/s &nbsp; w = {self._last_cmd_w:.2f} rad/s"
                )
        else:
            navila_msg = f"v = {self._last_cmd_v:.2f} m/s &nbsp; w = {self._last_cmd_w:.2f} rad/s"
        self.pub_navila_log.publish(String(data=navila_msg))

    @staticmethod
    def _normalize_angle(angle):
        """Normalize angle to [-pi, pi]."""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def vla_action_callback(self, msg):
        """Receive VLA action string (from navila_bridge.py or ros2 topic pub)."""
        text = msg.data.strip().lower()
        if not text:
            return

        if self._vla_stopped:
            return

        # Store latest VLA action for bottom-center overlay
        self._last_vla_action = msg.data.strip()

        if "stop" in text:
            self._vla_stopped = True
            self.local_goal = None
            self._is_turning = False
            self._is_forwarding = False
            self.pub_cmd_vel.publish(Twist())
            self.get_logger().info("VLA: stop — killing navila_bridge and self")
            import subprocess
            subprocess.Popen(["pkill", "-f", "navila_bridge.py --instruction"])
            subprocess.Popen(["pkill", "-f", "run.py --instruction"])
            return

        goto_match = re.match(
            r'goto\s*\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)', text
        )
        if goto_match:
            gx = float(goto_match.group(1))
            gy = float(goto_match.group(2))
            heading = float(goto_match.group(3))
            self._is_rollback = True
            self._rollback_target_heading = heading
            self._is_turning = False
            self._is_forwarding = False
            self._is_heading_correction = False
            self.local_goal = torch.tensor([[gx, gy]], device=DEVICE, dtype=torch.float32)
            self.get_logger().info(
                f"Rollback: RL → ({gx:.2f}, {gy:.2f}), "
                f"then heading={math.degrees(heading):.1f}deg"
            )
            return

        turn_match = re.search(r'turn\s+(left|right)\s+(\d+)', text)
        if turn_match:
            direction = turn_match.group(1)
            angle_deg = int(turn_match.group(2))
            angle_rad = math.radians(angle_deg)
            self._turn_direction = 1 if direction == "left" else -1
            turn_duration = angle_rad / self._turn_speed
            self._turn_end_time = time.monotonic() + turn_duration
            self._is_turning = True
            self._is_forwarding = False
            self.local_goal = None
            self.get_logger().info(f"VLA: turn {direction} {angle_deg}deg (open-loop {turn_duration:.2f}s)")
            return

        fwd_match = re.search(r'(?:forward|move forward)\s+(\d+)\s*cm', text)
        if fwd_match:
            dist_m = int(fwd_match.group(1)) / 100.0
            if self.VLA_USE_RL:
                # RL mode: set goal, let costmap_callback drive via policy
                yaw = self.robot_yaw
                gx = self.robot_x + dist_m * math.cos(yaw)
                gy = self.robot_y + dist_m * math.sin(yaw)
                self.local_goal = torch.tensor([[gx, gy]], device=DEVICE, dtype=torch.float32)
                self._is_turning = False
                self._is_forwarding = False
                self.get_logger().info(f"VLA: forward {dist_m:.2f}m -> goal ({gx:.2f}, {gy:.2f}) (RL mode)")
            else:
                # Open-loop mode: drive directly at fixed speed
                fwd_duration = dist_m / self._fwd_speed
                self._fwd_end_time = time.monotonic() + fwd_duration
                self._is_forwarding = True
                self._is_turning = False
                self.local_goal = None
                self.get_logger().info(f"VLA: forward {dist_m:.2f}m (open-loop {fwd_duration:.2f}s)")
            return

        self.get_logger().warn(f"VLA: unrecognized action '{text}'")

    def costmap_callback(self, msg):
        """Main Control Loop (Rate Limited to 10Hz)"""
        # 1. Rate Limiting
        current_time = self.get_clock().now()
        dt_nano = (current_time - self.last_control_time).nanoseconds
        dt_seconds = dt_nano / 1e9

        if dt_seconds < self.CONTROL_PERIOD:
            return

        self.last_control_time = current_time

        # 2. Robot State
        if not hasattr(self, '_costmap_cb_count'):
            self._costmap_cb_count = 0
        self._costmap_cb_count += 1
        if self._costmap_cb_count == 1:
            self.get_logger().info("Costmap CB: first costmap message received - callback is firing")

        if not self.is_pose_initialized:
            if self._costmap_cb_count % 50 == 1:
                self.get_logger().warn("Costmap CB: waiting for pose init (no odom on /odometry/local)")
            return

        current_x = self.robot_x
        current_y = self.robot_y
        current_yaw = self.robot_yaw

        if self.local_goal is None:
            self.pub_cmd_vel.publish(Twist())
            if self._costmap_cb_count % 50 == 1:
                self.get_logger().warn("Costmap CB: no active goal (local_goal is None, waiting for VLA forward cmd)")
            return

        # 3. Coordinate Transformation
        gx = self.local_goal[0, 0].item()
        gy = self.local_goal[0, 1].item()

        dx_global = gx - current_x
        dy_global = gy - current_y
        dist = math.sqrt(dx_global**2 + dy_global**2)

        if self._is_rollback:
            self.check_and_update_goal(dist)
            if self.local_goal is None: return
            gx = self.local_goal[0, 0].item()
            gy = self.local_goal[0, 1].item()
            dx_global = gx - current_x
            dy_global = gy - current_y
        elif self.is_language_instruction:
            self.get_logger().info(f"VLA dist to goal: {dist:.2f}m")
            if dist < 0.15:
                self.local_goal = None
                self.pub_cmd_vel.publish(Twist())
                self.get_logger().info(f"VLA forward arrived (dist={dist:.2f}m)")
                return
        else:
            self.check_and_update_goal(dist)
            if self.local_goal is None: return
            gx = self.local_goal[0, 0].item()
            gy = self.local_goal[0, 1].item()
            dx_global = gx - current_x
            dy_global = gy - current_y

        local_x = dx_global * math.cos(current_yaw) + dy_global * math.sin(current_yaw)
        local_y = -dx_global * math.sin(current_yaw) + dy_global * math.cos(current_yaw)

        state_input = torch.tensor([[local_x, local_y]], device=DEVICE, dtype=torch.float32)
        state_input = state_input / 20.0
        # RL policy trained with goals at 5-20m (state_input magnitude 0.25-1.0).
        # NaVILA sends 0.25-0.75m (magnitude 0.0125-0.0375) which is out of distribution.
        # Scale up to minimum training range, preserving direction only.
        # Arrival is handled by the 0.15m distance check, not by state magnitude.
        MIN_STATE_MAG = 0.25  # = 5m / 20 (minimum training distance)
        state_mag = torch.norm(state_input)
        if state_mag > 1e-6 and state_mag < MIN_STATE_MAG:
            state_input = state_input * (MIN_STATE_MAG / state_mag)
        state_input = torch.clamp(state_input, -1.0, 1.0)

        # 4. Visual Observation
        try:
            data_np = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
            costmap_tensor = torch.tensor(data_np, device=DEVICE, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

            # Matches training convention (nav_task.py:66-77):
            # permute swaps X/Y axes, flip rotates 180° -> top=forward for CNN
            costmap_tensor = costmap_tensor.permute(0, 1, 3, 2)
            costmap_tensor = torch.flip(costmap_tensor, dims=[-2, -1])

            costmap_tensor = torch.clamp(costmap_tensor, 0.0, 1.0)

        except Exception as e:
            self.get_logger().error(f"Error converting raw costmap image: {e}")
            return

        # 5. RL Inference & Smoothing
        v_raw, w_raw = self.navigator.get_action(costmap_tensor, state_input)

        v_raw_clamped = max(min(v_raw, 1.0), -1.0)
        w_raw_clamped = max(min(w_raw, 1.0), -1.0)

        target_v = float(v_raw_clamped * 0.5)
        W_SCALE = 0.5
        # Deadzone: suppress small w to eliminate residual bias on straight paths
        if abs(w_raw_clamped) < 0.15:
            w_raw_clamped = 0.0
        target_w = float(w_raw_clamped * W_SCALE)

        smooth_v, smooth_w = self.smoother.process(target_v, target_w, dt_seconds)

        cmd = Twist()
        cmd.linear.x = smooth_v
        cmd.angular.z = smooth_w

        self._last_cmd_v = cmd.linear.x
        self._last_cmd_w = cmd.angular.z
        self.pub_cmd_vel.publish(cmd)

        # 6. Debug Logging
        cm = costmap_tensor[0, 0]
        cm_min = cm.min().item()
        cm_max = cm.max().item()
        cm_mean = cm.mean().item()
        total_cells = cm.numel()
        obstacle_cells = (cm > 0.7).sum().item()
        path_cells = ((cm > 0.3) & (cm <= 0.7)).sum().item()
        unknown_cells = (cm <= 0.3).sum().item()

        self.get_logger().info(
            f"\n{'='*60}\n"
            f"[POLICY DEBUG] dt={dt_seconds:.3f}s\n"
            f"  Robot Pose   : x={current_x:.3f}, y={current_y:.3f}, yaw={math.degrees(current_yaw):.1f}deg\n"
            f"  Global Goal  : x={gx:.3f}, y={gy:.3f}, dist={dist:.3f}m\n"
            f"  Local Goal   : x={local_x:.3f}, y={local_y:.3f}\n"
            f"  State Input  : {state_input[0].tolist()} (after /20 + clamp)\n"
            f"  Costmap Stats: min={cm_min:.3f}, max={cm_max:.3f}, mean={cm_mean:.4f}\n"
            f"  Costmap Cells: obstacle={obstacle_cells}/{total_cells} ({100*obstacle_cells/total_cells:.1f}%), "
            f"path={path_cells} ({100*path_cells/total_cells:.1f}%), "
            f"unknown={unknown_cells} ({100*unknown_cells/total_cells:.1f}%)\n"
            f"  Raw Policy   : v={v_raw:.4f}, w={w_raw:.4f}\n"
            f"  Scaled       : v={target_v:.4f}, w={target_w:.4f}\n"
            f"  Smoothed Cmd : v={smooth_v:.4f}, w={smooth_w:.4f}\n"
            f"{'='*60}"
        )

        debug_msg = (
            f"Global Goal  : x={gx:.2f}, y={gy:.2f}\n"
            f"Local Remain : x={local_x:.2f}, y={local_y:.2f}\n"
            f"RL Output : v={v_raw:.2f}, w={w_raw:.2f}\n"
            f"Robot Action : v={cmd.linear.x:.2f}, w={cmd.angular.z:.2f}"
        )
        self.pub_debug_log.publish(String(data=debug_msg))



def main():
    parser = argparse.ArgumentParser(description="Run the HiVLA RL Navigator node.")
    parser.add_argument('--instruction', type=str, default="")
    known_args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = HiVLARunner(known_args.instruction)

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()

    except KeyboardInterrupt:
        pass

    except ExternalShutdownException:
        pass

    except Exception as e:
        if "context is not valid" in str(e):
            pass
        else:
            print(f"Unexpected Error: {e}")

    finally:
        try:
            node.destroy_node()
        except:
            pass

        if rclpy.ok():
            try:
                rclpy.shutdown()
            except:
                pass

if __name__ == '__main__':
    main()
