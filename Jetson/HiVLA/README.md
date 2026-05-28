## NaVILA Integration

HiVLA extends [NaVILA](https://navila-bot.github.io/) with a custom replanning module (`vlnce_baselines/hivla/`).

After installing NaVILA and running the hyperparameter search (navigation heads & thresholds) on the host PC, copy the `hivla/` directory into NaVILA's evaluation folder on the Jetson:

```bash
cp -r NaVILA/evaluation/vlnce_baselines/hivla/ third_party/NaVILA/evaluation/vlnce_baselines/
```

> Only `vlnce_baselines/hivla/` is required for real-robot deployment.

---

## Build

### Prerequisites: Livox SDK2 (required for FAST-LIV2)

Clone and install Livox SDK2 into `third_party/` before building the ROS 2 workspace:

```bash
cd third_party
git clone https://github.com/Livox-SDK/Livox-SDK2.git
cd Livox-SDK2
mkdir build && cd build
cmake .. && make -j$(nproc)
sudo make install
```

### ROS 2 Workspace

> **Build in order:** Run step 1 first, then step 2.

**Step 1** — Build `livox_ros_driver2` first (required before building the full workspace):
```bash
cd ros2_ws
colcon build --symlink-install --packages-select livox_ros_driver2 --cmake-args -DROS_EDITION=ROS2 -DHUMBLE_ROS=humble
```

**Step 2** — Build the rest of the workspace:
```bash
colcon build --symlink-install
```