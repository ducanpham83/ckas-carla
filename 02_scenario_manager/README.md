# Module 2: Scenario Manager

**Mô phỏng di chuyển xe 2D/3D và quản lý dữ liệu động học**

## 📋 Mô Tả

Module này tải kế hoạch đường đi từ Module 1 và thực hiện mô phỏng di chuyển của xe trong CARLA. Nó cung cấp tính năng lưu video, chỉnh sửa dữ liệu động học và áp dụng các bộ lọc để chuẩn bị dữ liệu cho Module 3.

## 🎯 Chức Năng Chính

- ✅ Load quãng đường di chuyển từ Module 1
- ✅ Load spawn points
- ✅ Mô phỏng di chuyển xe 2D trên CARLA
- ✅ Mô phỏng di chuyển xe 3D (visual)
- ✅ Lưu video di chuyển
- ✅ Chế độ chỉnh sửa dữ liệu động học
- ✅ Bộ lọc và cắt sửa dữ liệu

## 🔧 Kiến Trúc

```
02_scenario_manager/
├── src/
│   ├── main.py              # Entry point
│   ├── scenario_runner.py   # Main scenario execution
│   ├── carla_client.py      # CARLA simulation interface
│   ├── vehicle_controller.py # Vehicle control logic
│   ├── data_processor.py    # Data filtering & processing
│   ├── video_recorder.py    # Video recording
│   └── utils.py             # Utility functions
├── config/
│   ├── config.yaml          # Main configuration
│   └── filters.yaml         # Filter configurations
├── tests/
│   ├── test_scenario.py
│   ├── test_controller.py
│   └── test_processor.py
└── README.md
```

## 📥 Input

**From Module 1:**
- `shared_data/routes_plan.csv` - Route waypoints and parameters

## 📤 Output

**File:** `shared_data/dynamics_data.csv`

**Format:**
```csv
timestamp,vehicle_x,vehicle_y,vehicle_z,velocity_x,velocity_y,velocity_z,acceleration_x,acceleration_y,acceleration_z,yaw_angle,steering_angle,throttle,brake
0.0,100.5,200.3,0.5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
0.1,100.6,200.4,0.5,1.0,1.0,0.0,10.0,10.0,0.0,5.2,0.1,0.5,0.0
```

**Video Output:**
- `outputs/simulation_run_*.mp4`

## 🚀 Cách Sử Dụng

```bash
cd 02_scenario_manager

# Chạy scenario simulation
python src/main.py

# Chạy tests
python -m pytest tests/

# Chế độ interactive (chỉnh sửa dữ liệu)
python src/main.py --interactive
```

## 🔗 Dependencies

- carla >= 0.9.13
- opencv-python
- numpy
- pandas
- scipy
- pyyaml
- pytest

## 📊 Data Processing Pipeline

```
routes_plan.csv 
    ↓
CARLA Simulation (2D/3D)
    ↓
Raw Dynamics Data
    ↓
Filtering (Lowpass, Kalman, etc.)
    ↓
Data Editing Mode
    ↓
dynamics_data.csv
```

## 🎬 Video Recording

- Resolution: 1920x1080
- FPS: 30
- Format: MP4 (H.264)
- Location: `outputs/`

## 🔧 Bộ Lọc Được Hỗ Trợ

- **Lowpass Filter** - Loại nhiễu tần số cao
- **Kalman Filter** - Ước tính trạng thái tối ưu
- **Moving Average** - Làm mịn dữ liệu
- **Butterworth Filter** - Lọc Butterworth

## 🔄 Kết Nối với Module Khác

← **Module 1 (Offline 2D Planner)** cung cấp routes_plan.csv
→ **Module 3 (MCA CKAS COM)** sẽ sử dụng dynamics_data.csv

---

**Last Updated:** 2026-05-18
