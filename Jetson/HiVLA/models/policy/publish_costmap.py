import sys
import os

_HIVLA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _HIVLA_ROOT not in sys.path:
    sys.path.insert(0, _HIVLA_ROOT)

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, Image
from sensor_msgs_py import point_cloud2
from nav_msgs.msg import OccupancyGrid, Odometry
from tf2_ros import Buffer, TransformListener
from cv_bridge import CvBridge

import numpy as np
import cv2
import torch
import math
import time

from models.policy.config import DEVICE, GRID_RES, COSTMAP_SIZE_M, RESOLUTION, DTYPE, Z_MIN_THRESHOLD, Z_MAX_THRESHOLD
from models.policy.costmap import BatchGPULocalCostmapCore

# ==============================================================================
# 2. Costmap Visualization Node
# ==============================================================================
class CostmapVisualizationNode(Node):
    def __init__(self):
        super().__init__('costmap_visualization_node')
        
        # ----------------------------------------------------------------------
        # 2.1 GPU Costmap Core Initialization
        # ----------------------------------------------------------------------
        self.get_logger().info(f"Initializing GPU Costmap Core on {DEVICE}...")
        self.costmap_core = BatchGPULocalCostmapCore(
            device=DEVICE, 
            num_envs=1, 
            grid_res=GRID_RES, 
            map_size_m=COSTMAP_SIZE_M, 
            resolution=RESOLUTION
        )

        # ----------------------------------------------------------------------
        # 2.2 ROS 2 Communication Setup
        # ----------------------------------------------------------------------
        # Subscriber: LiDAR Point Cloud
        qos_sensor = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.sub_lidar = self.create_subscription(
            PointCloud2, '/rslidar_points', self.lidar_callback, qos_sensor
        )
        
        # Publishers: Separate topics for RL (float32) and Web (Image, uint8)
        self.pub_costmap_raw = self.create_publisher(Image, '/local_costmap/raw', 10)
        self.pub_costmap_img = self.create_publisher(Image, '/local_costmap/image', 10)
        
        # Utils: CV Bridge & TF (TF still needed for lidar→base_link transform)
        self.cv_bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subscriber: EKF Odometry (direct, avoids TF tree conflict)
        self.sub_odom = self.create_subscription(
            Odometry, '/odometry/local', self._odom_callback, 10
        )

        # ----------------------------------------------------------------------
        # 2.3 State Variables
        # ----------------------------------------------------------------------
        # Ego-Motion Calculation State
        self.prev_pose_x = 0.0
        self.prev_pose_y = 0.0
        self.prev_pose_yaw = 0.0
        self.first_pose_received = False
        self._latest_odom = None
        self._tf_warn_count = 0
        self._last_img_pub_time = 0.0  # rate-limit costmap image to 10Hz

        self.get_logger().info(f"✅ Node Started! Resolution: {RESOLUTION:.3f}m/px")

    def lidar_callback(self, msg):
        """Processes LiDAR data: TF transform -> Footprint Filter -> Height Filter."""
        try:
            # 1. Lookup Transform (Sensor Frame -> Robot Base Frame)
            tf_sensor = self.tf_buffer.lookup_transform('base_link', msg.header.frame_id, rclpy.time.Time())
            
            # 2. Convert PointCloud2 to Nx3 float32
            pts_iter = point_cloud2.read_points(
                msg,
                field_names=("x", "y", "z"),
                skip_nans=True
            )

            pts_list = list(pts_iter)
            if len(pts_list) == 0:
                return
            
            # Downsample: Keep every 2nd point (50% sampling)
            # Reduces Convert time from ~114ms to ~57ms
            pts_list = pts_list[::2]

            xyz_np = np.empty((len(pts_list), 3), dtype=np.float32)
            for i, p in enumerate(pts_list):
                xyz_np[i, 0] = float(p[0])
                xyz_np[i, 1] = float(p[1])
                xyz_np[i, 2] = float(p[2])
            
            # 3. Transform Points to GPU Tensor
            points_raw = torch.from_numpy(xyz_np).to(DEVICE, dtype=DTYPE)
            
            # Distance filtering on GPU (parallel, very fast!)
            # Costmap is 6.4m x 6.4m, filter beyond 5m radius
            MAX_RANGE = 5.0
            dist_sq = points_raw[:, 0]**2 + points_raw[:, 1]**2
            mask_range = dist_sq < MAX_RANGE**2
            
            # Combine with NaN filtering
            mask_valid = ~torch.isnan(points_raw).any(dim=1) & mask_range
            points_raw = points_raw[mask_valid]
            if len(points_raw) == 0:
                return
            points_base = self.transform_points(points_raw, tf_sensor)
            
            # 4. Footprint Filtering (Remove Robot Body)
            x = points_base[:, 0]
            y = points_base[:, 1]

            # A. Footprint Filtering (Remove Robot Body)
            # Keep points OUTSIDE the robot box
            HALF_LEN = 0.47 + 0.06
            HALF_WID = 0.35 + 0.10
            mask_footprint = (torch.abs(x) > HALF_LEN) | (torch.abs(y) > HALF_WID)
            
            points_filtered = points_base[mask_footprint]

            # 5. Update Costmap Immediately (Event-Driven)
            if len(points_filtered) > 0:
                current_points = points_filtered.unsqueeze(0)
                
                # Get robot motion
                dx, dy, dtheta = self.get_robot_motion()

                # Update costmap with latest data
                final_costmap = self.costmap_core.update_costmap(current_points, dx, dy, dtheta)
                
                # Publish costmap
                self.publish_costmap(final_costmap)
                
        except Exception as e:
            self.get_logger().warn(f"LiDAR callback error: {e}")

    def _odom_callback(self, msg: Odometry):
        self._latest_odom = msg

    def get_robot_motion(self):
        """Calculates the relative motion (dx, dy, dtheta) since the last frame using /odometry/local."""
        if self._latest_odom is None:
            return torch.tensor([0.0], device=DEVICE, dtype=DTYPE), \
                   torch.tensor([0.0], device=DEVICE, dtype=DTYPE), \
                   torch.tensor([0.0], device=DEVICE, dtype=DTYPE)

        p = self._latest_odom.pose.pose
        tx = p.position.x
        ty = p.position.y
        q = p.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        if not self.first_pose_received:
            self.prev_pose_x = tx
            self.prev_pose_y = ty
            self.prev_pose_yaw = yaw
            self.first_pose_received = True
            return torch.tensor([0.0], device=DEVICE, dtype=DTYPE), \
                   torch.tensor([0.0], device=DEVICE, dtype=DTYPE), \
                   torch.tensor([0.0], device=DEVICE, dtype=DTYPE)

        dx_global = tx - self.prev_pose_x
        dy_global = ty - self.prev_pose_y
        dtheta = yaw - self.prev_pose_yaw
        dtheta = (dtheta + math.pi) % (2 * math.pi) - math.pi

        cos_yaw = math.cos(self.prev_pose_yaw)
        sin_yaw = math.sin(self.prev_pose_yaw)
        dx_local = dx_global * cos_yaw + dy_global * sin_yaw
        dy_local = -dx_global * sin_yaw + dy_global * cos_yaw

        self.prev_pose_x = tx
        self.prev_pose_y = ty
        self.prev_pose_yaw = yaw

        return torch.tensor([dx_local], device=DEVICE, dtype=DTYPE), \
               torch.tensor([dy_local], device=DEVICE, dtype=DTYPE), \
               torch.tensor([dtheta], device=DEVICE, dtype=DTYPE)

    def transform_points(self, points, tf_msg):
        """Applies a ROS Transform to a batch of points."""
        t = tf_msg.transform.translation
        q = tf_msg.transform.rotation
        t_vec = torch.tensor([t.x, t.y, t.z], device=DEVICE, dtype=DTYPE)

        # Rotation Matrix from Quaternion
        r00 = 1 - 2*(q.y**2 + q.z**2); r01 = 2*(q.x*q.y - q.z*q.w); r02 = 2*(q.x*q.z + q.y*q.w)
        r10 = 2*(q.x*q.y + q.z*q.w);   r11 = 1 - 2*(q.x**2 + q.z**2); r12 = 2*(q.y*q.z - q.x*q.w)
        r20 = 2*(q.x*q.z - q.y*q.w);   r21 = 2*(q.y*q.z + q.x*q.w);   r22 = 1 - 2*(q.x**2 + q.y**2)
        
        R = torch.tensor([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]], device=DEVICE, dtype=DTYPE)
        return points @ R.T + t_vec

    def publish_costmap(self, costmap_tensor):
        """Publishes the Costmap as RAW float32 Image and Display Image (uint8)."""
        
        # 1. Prepare Data: Convert Tensor to Numpy (range 0.0-1.0)
        costmap_float_cpu = costmap_tensor.squeeze().float().cpu().numpy()
        
        # ----------------------------------------------------------------------
        # A. Publish RAW Float32 Image (FOR RL POLICY) - Topic: /local_costmap/raw
        # ----------------------------------------------------------------------
        msg_raw_float = self.cv_bridge.cv2_to_imgmsg(
            costmap_float_cpu, 
            encoding="32FC1" # Float 32, 1 Channel - NO PRECISION LOSS
        )
        msg_raw_float.header.stamp = self.get_clock().now().to_msg()
        msg_raw_float.header.frame_id = "base_link"
        self.pub_costmap_raw.publish(msg_raw_float)

        # ----------------------------------------------------------------------
        # B. Publish Display Image (FOR Web Video Server) - Topic: /local_costmap/image
        # Rate-limited to 10Hz to reduce network usage
        # ----------------------------------------------------------------------
        now = time.monotonic()
        if now - self._last_img_pub_time >= 0.1:
            self._last_img_pub_time = now

            # Quantize the float data to int8 (0-100) for color mapping
            costmap_int8_cpu = (costmap_float_cpu * 100).astype(np.int8)
            grid_2d = costmap_int8_cpu.reshape(GRID_RES, GRID_RES)

            # Color Mapping (Uses quantized 0-100 data)
            img_display = np.zeros((GRID_RES, GRID_RES), dtype=np.uint8)
            img_display.fill(128)
            img_display[grid_2d == 0] = 255
            img_display[grid_2d == 100] = 0
            mask_dynamic = (grid_2d > 0) & (grid_2d < 100)
            img_display[mask_dynamic] = 255 - (grid_2d[mask_dynamic] * 2.5).astype(np.uint8)

            # Coordinate Transformation:
            img_display = cv2.rotate(img_display, cv2.ROTATE_90_COUNTERCLOCKWISE)
            img_display = cv2.flip(img_display, 1)

            # Convert to ROS Image Message
            msg_img = self.cv_bridge.cv2_to_imgmsg(img_display, encoding="mono8")
            msg_img.header = msg_raw_float.header

            self.pub_costmap_img.publish(msg_img)

# ==============================================================================
# 3. Main Execution
# ==============================================================================
def main():
    rclpy.init()
    node = CostmapVisualizationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()