import os
import time
import threading
import shutil
import signal
import atexit
import subprocess
import re
from datetime import datetime

import psutil
from flask import Flask, render_template
from flask_socketio import SocketIO

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from nav_msgs.msg import Odometry

import tempfile

# ==============================================================================
# 1. Configuration & Setup
# ==============================================================================
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # HiVLA root
BAG_ROOT_DIR = os.path.join(_BASE_DIR, "rosbag")
WEB_LOG_PATH = os.path.join(_BASE_DIR, "logs", "web_video_server.log")

app = Flask(__name__, static_folder='static', template_folder='templates')
socketio = SocketIO(
    app,
    cors_allowed_origins='*',
    ping_interval=5,
    ping_timeout=1
)

# ==============================================================================
# 2. Global State Variables
# ==============================================================================
# Process Handles
record_process = None
run_process = None
navila_process = None

# Control State
down_keys = set()
should_update_twist = True
last_heartbeat_time = 0.0  # 0 means no heartbeat received yet; watchdog won't fire

# Robot Pose (Updated via TF)
robot_pose = {'x': 0.0, 'y': 0.0, 'z': 0.0}

# Speed Control Settings
linear_speed = 0.5
angular_speed = 0.5
MAX_LINEAR_SPEED = 1.5
MIN_LINEAR_SPEED = 0.1
MAX_ANGULAR_SPEED = 1.0
MIN_ANGULAR_SPEED = 0.1

# ==============================================================================
# 3. ROS 2 Initialization
# ==============================================================================
rclpy.init()
node = rclpy.create_node('web_control_node')
pub_cmd_vel = node.create_publisher(Twist, '/cmd_vel', 10)

os.makedirs(os.path.join(_BASE_DIR, "logs"), exist_ok=True)

_odom_log_file = None
_odom_last_log_time = 0.0

def _get_yaw(q):
    import math
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

def global_odom_callback(msg):
    global robot_pose, _odom_last_log_time, _odom_log_file
    robot_pose['x'] = msg.pose.pose.position.x
    robot_pose['y'] = msg.pose.pose.position.y
    robot_pose['z'] = msg.pose.pose.position.z

    if _odom_log_file is not None:
        now = time.time()
        if now - _odom_last_log_time >= 0.1:
            _odom_last_log_time = now
            yaw = _get_yaw(msg.pose.pose.orientation)
            _odom_log_file.write(f"{now:.3f},{robot_pose['x']:.4f},{robot_pose['y']:.4f},{yaw:.4f}\n")
            _odom_log_file.flush()

# node.create_subscription(Odometry, '/odometry/global', global_odom_callback, 10)
node.create_subscription(Odometry, '/odometry/local', global_odom_callback, 10)
# node.create_subscription(Odometry, '/fast_livo/odom', global_odom_callback, 10)
# node.create_subscription(Odometry, '/zed/odom', global_odom_callback, 10)

def debug_log_callback(msg):
    """Receives logs published by run.py and forwards them to the web client."""
    socketio.emit('debug_log', {'text': msg.data})

node.create_subscription(String, '/hivla/debug_log', debug_log_callback, 10)

def navila_log_callback(msg):
    """Receives NaVILA output logs and forwards to the bottom-center overlay."""
    socketio.emit('navila_log', {'text': msg.data})

node.create_subscription(String, '/hivla/navila_log', navila_log_callback, 10)

# ==============================================================================
# 4. Helper Functions: System Monitoring
# ==============================================================================
def get_orin_stats():
    """
    Parses 'tegrastats' for Jetson Orin hardware metrics.
    Uses 'timeout' to capture a snapshot without blocking indefinitely.
    """
    stats = {'cpu': 0, 'gpu': 0, 'mem': 0, 'temp': 0}
    cmd_output = ""
    
    try:
        # Run tegrastats for 0.2s. This will cause a timeout error (exit code 124),
        # but we catch it to retrieve the output captured up to that point.
        subprocess.check_output(
            ['timeout', '0.2', 'tegrastats', '--interval', '100'], 
            text=True, stderr=subprocess.STDOUT
        )
    except subprocess.CalledProcessError as e:
        # The output is stored in the exception object
        cmd_output = e.output
    except Exception as e:
        print(f"Error executing tegrastats: {e}")
        # Fallback to psutil if tegrastats fails completely
        stats['cpu'] = psutil.cpu_percent()
        stats['mem'] = psutil.virtual_memory().percent
        return stats

    # Parse the tegrastats output string
    try:
        # 1. GPU Load (e.g., GR3D_FREQ 63%@...)
        gpu_match = re.search(r'GR3D(?:_FREQ)?\s+(\d+)%', cmd_output)
        if gpu_match:
            stats['gpu'] = int(gpu_match.group(1))
        
        # 2. CPU Load (Average across cores)
        cpu_part = re.search(r'CPU \[(.*?)\]', cmd_output)
        if cpu_part:
            cores = re.findall(r'(\d+)%', cpu_part.group(1))
            if cores:
                stats['cpu'] = sum(map(int, cores)) / len(cores)

        # 3. Memory Usage
        mem_match = re.search(r'RAM (\d+)/(\d+)MB', cmd_output)
        if mem_match:
            mem_used = int(mem_match.group(1))
            mem_total = int(mem_match.group(2))
            if mem_total > 0:
                stats['mem'] = (mem_used / mem_total) * 100
                
        # 4. Temperature (Prioritize GPU, fallback to SoC)
        temp_match = re.search(r'gpu@([\d\.]+)C', cmd_output)
        if not temp_match:
            temp_match = re.search(r'soc0@([\d\.]+)C', cmd_output)
        if temp_match:
            stats['temp'] = float(temp_match.group(1))
            
    except Exception as e:
        print(f"Error parsing stats: {e}")

    return stats


def safe_ros_shutdown():
    """Checks if ROS is running before attempting to shut down."""
    if rclpy.ok():
        print("🛑 Safely shutting down ROS 2 context...")
        rclpy.shutdown()
    else:
        print("🛑 ROS 2 context already shut down.")

def cleanup_all_processes():
    """Cleanup handler for application exit."""
    clear_keys_and_stop()
    safe_ros_shutdown()

# Register cleanup function to run on exit
atexit.register(cleanup_all_processes)

# ==============================================================================
# 6. SocketIO Event Handlers
# ==============================================================================
@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    global should_update_twist
    should_update_twist = True
    print("🔄 Client connected")
    # Sync current run state so the button reflects reality on reconnect
    if run_process is not None:
        socketio.emit('instruction_status', {'status': 'running'})
    else:
        socketio.emit('instruction_status', {'status': 'idle'})

@socketio.on('disconnect')
def handle_disconnect():
    print("🔌 Client disconnected")
    clear_keys_and_stop()

@socketio.on('keydown')
def handle_keydown(data):
    global linear_speed, angular_speed, should_update_twist
    down_keys.add(data)
    should_update_twist = True
    
    # Adjust speed settings based on key input
    updated = False
    if data == ',': linear_speed = max(MIN_LINEAR_SPEED, linear_speed - 0.1); updated = True
    elif data == '.': linear_speed = min(MAX_LINEAR_SPEED, linear_speed + 0.1); updated = True
    elif data == '[': angular_speed = max(MIN_ANGULAR_SPEED, angular_speed - 0.1); updated = True
    elif data == ']': angular_speed = min(MAX_ANGULAR_SPEED, angular_speed + 0.1); updated = True

@socketio.on('keyup')
def handle_keyup(data):
    down_keys.discard(data)

@socketio.on('heartbeat')
def handle_heartbeat(data=None):
    """Sync key state from client heartbeat as safety fallback."""
    global last_heartbeat_time, should_update_twist
    last_heartbeat_time = time.time()
    if data and 'keys' in data:
        client_keys = set(data['keys'])
        # If server thinks keys are pressed but client says they're not, fix it
        stale = down_keys - client_keys
        if stale:
            down_keys.intersection_update(client_keys)
            should_update_twist = True

qos_temp_file_path = None

@socketio.on('start_recording')
def handle_start_recording(dirname):
    global record_process, qos_temp_file_path
    if record_process: return

    dirname = dirname or "hivla_data"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    bag_path = os.path.join(BAG_ROOT_DIR, dirname, timestamp)
    if not os.path.exists(os.path.dirname(bag_path)):
        os.makedirs(os.path.dirname(bag_path), exist_ok=True)

    print(f"📥 Start ROS2 Bag recording: {bag_path}")

    # 1. Create a temporary YAML file for QoS overrides
    # We explicitly define the content for /tf_static
    qos_content = """
    /tf_static:
      durability: transient_local
      reliability: reliable
      history: keep_last
      depth: 1
    """
    
    try:
        # Create a named temp file that persists until we manually delete it
        # (delete=False is important so the subprocess can read it)
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml') as tmp:
            tmp.write(qos_content)
            qos_temp_file_path = tmp.name
        
        print(f"📄 Created temporary QoS override file: {qos_temp_file_path}")

        # 2. Define topics to record (Added the ones from your command)
        topics_to_record = [
            "tf_static",
            "tf",
            "/scout/odom",
            "/rslidar_points",
            "/zed/zed_node/odom",
            "/zed/zed_node/imu/data",
            "/zed/zed_node/rgb/color/rect/camera_info",
            "/zed/zed_node/rgb/color/rect/image",
            "/zed/zed_node/depth/camera_info",
            "/zed/zed_node/depth/depth_registered",
        ]

        # 3. Construct the command
        cmd = [
            "ros2", "bag", "record", 
            "-s", "mcap", 
            "-o", bag_path,
            "--qos-profile-overrides-path", qos_temp_file_path
        ] + topics_to_record

        # 4. Start the process
        record_process = subprocess.Popen(cmd, cwd=BAG_ROOT_DIR)
        socketio.emit('rec_status', {'status': 'recording', 'path': bag_path})

    except Exception as e:
        print(f"❌ Record Fail: {e}")
        # Clean up if it failed immediately
        if qos_temp_file_path and os.path.exists(qos_temp_file_path):
            os.remove(qos_temp_file_path)
            qos_temp_file_path = None

@socketio.on('stop_recording')
def handle_stop_recording():
    global record_process, qos_temp_file_path
    if not record_process: return

    print("🛑 Stopping Recording...")
    record_process.send_signal(signal.SIGINT)
    
    try: 
        record_process.wait(timeout=5)
    except: 
        record_process.kill()
    
    record_process = None
    socketio.emit('rec_status', {'status': 'idle'})

    # 5. Clean up the temporary QoS file
    if qos_temp_file_path and os.path.exists(qos_temp_file_path):
        try:
            os.remove(qos_temp_file_path)
            print(f"🗑️ Deleted temporary QoS file: {qos_temp_file_path}")
        except Exception as e:
            print(f"⚠️ Failed to delete temp QoS file: {e}")
        qos_temp_file_path = None

# ==============================================================================
# Instruction Execution Handlers (run.py)
# ==============================================================================
@socketio.on('run_instruction')
def handle_run_instruction(instruction):
    global run_process, navila_process

    if run_process:
        socketio.emit('instruction_status', {'status': 'running', 'message': 'Already running'})
        return

    # [SAFETY] Stop the robot before handing over control to AI
    pub_cmd_vel.publish(Twist())
    print("Handing over control to AI...")

    cmd = ["python3", os.path.join(_BASE_DIR, "run.py"), "--instruction", instruction]
    print(f"▶️ Starting run.py with instruction: '{instruction}'")

    # Detect if instruction is (x,y) coordinates → skip NaVILA
    is_coordinate = bool(re.search(r'[\(\[\{]\s*-?\d+\.?\d*\s*,\s*-?\d+\.?\d*\s*[\)\]\}]', instruction))

    # Use the log directory created by the hivla startup script (same session)
    run_log_dir = os.environ.get("HIVLA_LOG_DIR", os.path.join(_BASE_DIR, "logs"))

    # Start odom logging for this run
    _odom_log_file = open(os.path.join(run_log_dir, "odom_waypoints.csv"), "w")
    _odom_log_file.write("timestamp,x,y,yaw\n")

    try:
        log_file = open(os.path.join(run_log_dir, "instruction_run.log"), "w")
        run_process = subprocess.Popen(
            cmd,
            preexec_fn=os.setsid,
            stdout=log_file,
            stderr=subprocess.STDOUT
        )

        # Only start navila_bridge.py for language instructions (not coordinates)
        if not is_coordinate:
            navila_cmd = ["python3", os.path.join(_BASE_DIR, "models", "vla", "navila_bridge.py"), "--instruction", instruction]
            navila_log_path = os.path.join(run_log_dir, "navila_bridge.log")
            navila_log = open(navila_log_path, "w")
            navila_process = subprocess.Popen(
                navila_cmd,
                preexec_fn=os.setsid,
                stdout=navila_log,
                stderr=subprocess.STDOUT
            )
            print(f"▶️ Starting navila_bridge.py with instruction: '{instruction}'")
        else:
            print(f"📍 Coordinate mode — skipping navila_bridge.py")

        socketio.emit('instruction_status', {'status': 'running'})
        threading.Thread(target=monitor_run_process, daemon=True).start()

    except Exception as e:
        print(f"❌ Failed to start run.py: {e}")
        socketio.emit('instruction_status', {'status': 'error', 'message': str(e)})

@socketio.on('stop_instruction')
def handle_stop_instruction():
    global run_process, navila_process

    if not run_process:
        socketio.emit('instruction_status', {'status': 'idle'})
        return

    # Stop navila_bridge.py first
    if navila_process:
        print("🛑 Stopping navila_bridge.py...")
        try:
            os.killpg(os.getpgid(navila_process.pid), signal.SIGTERM)
            navila_process.wait(timeout=5)
        except Exception as e:
            print(f"⚠️ Error killing navila_bridge.py: {e}")
            try: navila_process.kill()
            except: pass
        navila_process = None

    print("🛑 Stopping run.py...")
    try:
        os.killpg(os.getpgid(run_process.pid), signal.SIGTERM)
        run_process.wait(timeout=5)
    except Exception as e:
        print(f"⚠️ Error killing run.py: {e}")
        try: run_process.kill()
        except: pass

    run_process = None

    # [SAFETY] Stop the robot immediately after AI stops to regain manual control safely
    pub_cmd_vel.publish(Twist())
    print("✅ Manual Control Restored.")

    socketio.emit('instruction_status', {'status': 'idle'})

def monitor_run_process():
    """Monitors the run_process completion and updates the client."""
    global run_process, navila_process, _odom_log_file
    if run_process is None: return

    return_code = run_process.wait()

    # Double check global variable (might have been stopped manually)
    if run_process is None: return

    run_process = None

    # Close odom log for this run
    if _odom_log_file is not None:
        _odom_log_file.close()
        _odom_log_file = None

    # Kill navila_process if run.py exited naturally (e.g. all waypoints reached)
    if navila_process:
        try:
            os.killpg(os.getpgid(navila_process.pid), signal.SIGTERM)
            navila_process.wait(timeout=3)
        except Exception:
            try: navila_process.kill()
            except: pass
        navila_process = None

    # [SAFETY] Ensure robot stops when AI finishes its task
    pub_cmd_vel.publish(Twist())

    if return_code == 0:
        print("✅ run.py completed successfully.")
        socketio.emit('instruction_status', {'status': 'success'})
    else:
        print(f"❌ run.py terminated with error code: {return_code}")
        socketio.emit('instruction_status', {'status': 'error', 'message': f"Exit Code {return_code}"})

# ==============================================================================
# 7. Background Threads
# ==============================================================================
def clear_keys_and_stop():
    """Resets key inputs and stops the robot."""
    global down_keys, should_update_twist
    down_keys.clear()
    should_update_twist = False
    pub_cmd_vel.publish(Twist())

def update_loop():
    """Main control loop. Pauses publishing when run.py (AI) is active."""
    global should_update_twist, run_process, linear_speed, angular_speed

    while True:
        # 1. PRIORITY CHECK: Is HiVLA (AI) running?
        # If AI is running, the Web Controller MUST NOT publish to /cmd_vel.
        if run_process is not None:
            time.sleep(0.5)  # Check less frequently to save CPU
            continue

        # Watchdog: if no heartbeat from client for >1s, emergency stop
        # Only active after the first heartbeat is received (last_heartbeat_time > 0)
        if down_keys and last_heartbeat_time > 0 and (time.time() - last_heartbeat_time) > 3.0:
            print("⚠️ Heartbeat timeout - stopping robot")
            down_keys.clear()
            pub_cmd_vel.publish(Twist())
            should_update_twist = False
            time.sleep(0.1)
            continue

        if not down_keys:
            if should_update_twist:
                pub_cmd_vel.publish(Twist())
                should_update_twist = False
            time.sleep(0.1)
            continue

        twist = Twist()
        
        if 'ArrowUp' in down_keys: 
            twist.linear.x += linear_speed
        if 'ArrowDown' in down_keys: 
            twist.linear.x -= linear_speed
        if 'ArrowLeft' in down_keys: 
            twist.angular.z += angular_speed
        if 'ArrowRight' in down_keys: 
            twist.angular.z -= angular_speed
        
        pub_cmd_vel.publish(twist)
        should_update_twist = True  # Allow stop command on next loop iteration
        time.sleep(0.1)

def ros_spin():
    """Keeps the ROS 2 node alive."""
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)

def system_monitor():
    """Thread for monitoring system stats (CPU/GPU/Disk) and Robot Pose."""
    last_io = psutil.disk_io_counters()
    last_time = time.time()

    while True:
        time.sleep(1.0)

        # 1. Disk Activity
        write_mbps = 0.0
        try:
            current_io = psutil.disk_io_counters()
            current_time = time.time()
            dt = current_time - last_time
            if dt <= 0: dt = 1.0
            write_mbps = (current_io.write_bytes - last_io.write_bytes) / dt / (1024 * 1024)
            last_io = current_io
            last_time = current_time
        except: pass

        # 2. Hardware Stats (Orin)
        stats = get_orin_stats()

        # 3. Disk Usage
        try:
            usage = shutil.disk_usage(BAG_ROOT_DIR)
            used_tb = round(usage.used / (1024**4), 2)
            total_tb = round(usage.total / (1024**4), 2)
        except:
            used_tb, total_tb = 0, 0

        # 4. WiFi
        try:
            wifi = subprocess.check_output(["iwgetid", "-r"], text=True, timeout=1).strip() or "N/A"
        except:
            wifi = "N/A"

        # 5. Emit Data to Client
        payload = {
            'cpu': round(stats['cpu'], 1),
            'mem': round(stats['mem'], 1),
            'gpu': round(stats['gpu'], 1),
            'temp': round(stats['temp'], 1),
            'disk_busy_pct': round(write_mbps, 1),
            'disk_used_tb': used_tb,
            'disk_total_tb': total_tb,
            'wifi': wifi,
            'odom_x': robot_pose['x'],
            'odom_y': robot_pose['y'],
            'linear': round(linear_speed, 1),
            'angular': round(angular_speed, 1)
        }

        socketio.emit('sysmon', payload)

# ==============================================================================
# 8. Main Entry Point
# ==============================================================================
if __name__ == '__main__':
    # Signal handling for graceful shutdown
    signal.signal(signal.SIGHUP, lambda s, f: cleanup_all_processes())
    signal.signal(signal.SIGINT, lambda s, f: cleanup_all_processes())

    # Start background threads
    threading.Thread(target=system_monitor, daemon=True).start()
    threading.Thread(target=ros_spin, daemon=True).start()
    threading.Thread(target=update_loop, daemon=True).start()

    os.makedirs(BAG_ROOT_DIR, exist_ok=True)
    
    print("✅ Server running on http://0.0.0.0:5000")
    socketio.run(app, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True, use_reloader=False)