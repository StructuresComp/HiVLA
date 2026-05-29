## NaVILA Integration

HiVLA extends [NaVILA](https://navila-bot.github.io/) with a custom replanning module (`vlnce_baselines/hivla/`).

After installing NaVILA and running the hyperparameter search (navigation heads & thresholds) on the host PC, copy the `hivla/` directory into NaVILA's evaluation folder on the Jetson:

```bash
cp -r NaVILA/evaluation/vlnce_baselines/hivla/ third_party/NaVILA/evaluation/vlnce_baselines/
```

> Only `vlnce_baselines/hivla/` is required for real-robot deployment.

---

## Build

### Cloning with Submodules

This repository uses git submodules for three third-party dependencies under `third_party/`:

| Submodule | Source |
|-----------|--------|
| `Livox-SDK2` | https://github.com/Livox-SDK/Livox-SDK2.git |
| `janus-gateway` | https://github.com/meetecho/janus-gateway.git |
| `libnice` | https://gitlab.freedesktop.org/libnice/libnice |

Clone the repository with all submodules in one command:

```bash
git clone --recurse-submodules https://github.com/StructuresComp/HiVLA.git
```

If you already cloned without `--recurse-submodules`, initialize them manually:

```bash
git submodule update --init --recursive
```

---

### Prerequisites: gs_usb Kernel Module (required for AgileX Scout 2.0)

The AgileX Scout 2.0 communicates over CAN bus via a USB-CAN adapter that relies on the `gs_usb` kernel module. This module is **not included by default in L4T R36.4.4**, so it must be built and installed manually.

> **Optional:** Only needed if you are using the AgileX Scout 2.0 (or any platform that requires the `gs_usb` CAN-over-USB kernel module).

Run the provided build script from the `third_party/gs_usb/` directory. On Jetson (native ARM64), it will automatically download the kernel sources and generate the kernel config:

```bash
cd third_party/gs_usb
bash jetson-gs_usb-kernel-builder.sh
```

After installation, reboot the Jetson to load the module.

---

### Prerequisites: Livox SDK2 (required for FAST-LIVO2)

Install Livox SDK2 from `third_party/Livox-SDK2/` (already available after cloning with submodules):

```bash
cd third_party/Livox-SDK2
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