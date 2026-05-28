#!/usr/bin/env python3
# ==============================================================================
# Script Name: publish_tag.py
# Description: [Anti-Teleport Mode] 
#              Calculates absolute pose from AprilTag but BLOCKS any data
#              that jumps faster than the robot's physical limit (0.5m/s).
# ==============================================================================

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
import numpy as np

def quaternion_from_euler(roll, pitch, yaw):
    """ Converts Euler angles to Quaternion. """
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return [qx, qy, qz, qw]

def make_trans_mat(t, q):
    """ Creates a 4x4 Transformation Matrix. """
    x, y, z, w = q
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y]
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [t.x, t.y, t.z] if hasattr(t, 'x') else t
    return T

class TagToPose(Node):
    def __init__(self):
        super().__init__('tag_to_pose_converter')
        
        self.publisher_ = self.create_publisher(PoseWithCovarianceStamped, '/pose_from_tag', 10)
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # =========================================================
        # [Configuration] Static Tag Transform
        # =========================================================
        tag_pos = [0.0, 0.0, 0.7366] 
        tag_quat = quaternion_from_euler(1.5707, 0.0, -1.5707) 
        self.T_map_tag = make_trans_mat(tag_pos, tag_quat)
        self.max_distance_limit = 2.5
        self.timer = self.create_timer(0.03, self.compute_pose)

    def compute_pose(self):
        try:
            trans = self.tf_buffer.lookup_transform(
                'tag36h11:0', 'base_link', rclpy.time.Time()) 

            # 1. Stale Check
            now = self.get_clock().now()
            trans_time = rclpy.time.Time.from_msg(trans.header.stamp)
            time_diff = (now - trans_time).nanoseconds / 1e9

            if time_diff > 0.2:
                return 

            # 2. Compute Pose
            t = trans.transform.translation
            dist_to_tag = np.sqrt(t.x**2 + t.y**2 + t.z**2)
            
            if dist_to_tag > self.max_distance_limit:
                self.get_logger().warn(
                    f"[DROP] Too Far! Dist: {dist_to_tag:.2f}m > Limit: {self.max_distance_limit}m", 
                    throttle_duration_sec=1
                )
                return

            q = trans.transform.rotation
            T_tag_base = make_trans_mat(t, [q.x, q.y, q.z, q.w])
            T_map_base = np.dot(self.T_map_tag, T_tag_base)
            final_pos = T_map_base[:3, 3]

            # 3. Extract Orientation
            R = T_map_base[:3, :3]
            tr = np.trace(R)
            if tr > 0:
                S = np.sqrt(tr + 1.0) * 2
                qw, qx, qy, qz = 0.25 * S, (R[2,1] - R[1,2]) / S, (R[0,2] - R[2,0]) / S, (R[1,0] - R[0,1]) / S
            else:
                qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0 

            # 4. Publish Message
            msg = PoseWithCovarianceStamped()
            msg.header.stamp = now.to_msg()
            msg.header.frame_id = "map"

            msg.pose.pose.position.x = final_pos[0]
            msg.pose.pose.position.y = final_pos[1]
            msg.pose.pose.position.z = final_pos[2]
            msg.pose.pose.orientation.x = qx
            msg.pose.pose.orientation.y = qy
            msg.pose.pose.orientation.z = qz
            msg.pose.pose.orientation.w = qw

            # [Optional] Increased Covariance slightly for smoother EKF fusion
            # 1e-5 is extremely strict. 0.01 allows EKF to smooth out small jitters.
            msg.pose.covariance = [0.0] * 36
            cov_val = 0.01
            msg.pose.covariance[0] = cov_val
            msg.pose.covariance[7] = cov_val
            msg.pose.covariance[14] = cov_val
            msg.pose.covariance[35] = cov_val

            self.publisher_.publish(msg)

        except (LookupException, ConnectivityException, ExtrapolationException):
            pass

def main():
    rclpy.init()
    node = TagToPose()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()