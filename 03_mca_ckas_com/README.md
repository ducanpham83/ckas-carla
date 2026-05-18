# Module 3: MCA CKAS COM

**Xử lý dữ liệu động học cho bệ mô phỏng CKAS**

## 📋 Mô Tả

Module này thực hiện các phép xử lý tín hiệu nâng cao bao gồm bộ lọc Washout và tạo ra file M code tương ứng cho bệ mô phỏng chuyển động CKAS (Computer Assisted Research and Education in Simulation - Motion Platform).

## 🎯 Chức Năng Chính

- ✅ Áp dụng bộ lọc Washout (Washout Filter)
- ✅ Xử lý dữ liệu động học
- ✅ Tạo file M code tương ứng
- ✅ Tương thích với bệ mô phỏng CKAS
- ✅ Validation dữ liệu output
- ✅ Optimization tín hiệu

## 🔧 Kiến Trúc

```
03_mca_ckas_com/
├── src/
│   ├── main.py              # Entry point
│   ├── washout_filter.py    # Washout filter implementation
│   ├── signal_processor.py  # Signal processing
│   ├── mcode_generator.py   # M-code file generator
│   ├── ckas_interface.py    # CKAS system interface
│   └── utils.py             # Utility functions
├── config/
│   ├── config.yaml          # Main configuration
│   ├── washout_params.yaml  # Washout filter parameters
│   └── ckas_commands.template # M-code template
├── tests/
│   ├── test_washout.py
│   ├── test_processor.py
│   └── test_generator.py
└── README.md
```

## 📥 Input

**From Module 2:**
- `shared_data/dynamics_data.csv` - Vehicle dynamics data

## 📤 Output

**File:** `shared_data/ckas_commands.m`

**Format (MATLAB M-code):**
```matlab
% CKAS Motion Platform Commands
% Generated from CKAS-CARLA Simulation
% Timestamp: 2026-05-18 10:00:00

% Motion platform parameters
platform_dof = 6;  % Degrees of freedom
max_displacement = [0.5, 0.5, 0.3, 0.2, 0.2, 0.3]; % X, Y, Z, Roll, Pitch, Yaw

% Washout filtered motion commands
time = [0.0, 0.1, 0.2, ...];
motion_x = [0.0, 0.05, 0.10, ...];
motion_y = [0.0, 0.02, 0.04, ...];
motion_z = [0.0, 0.01, 0.02, ...];
rotation_roll = [0.0, 0.001, 0.002, ...];
rotation_pitch = [0.0, 0.003, 0.006, ...];
rotation_yaw = [0.0, 0.002, 0.004, ...];

% Send to CKAS platform
ckas_send_motion_commands(time, motion_x, motion_y, motion_z, ...
                         rotation_roll, rotation_pitch, rotation_yaw);
```

## 🚀 Cách Sử Dụng

```bash
cd 03_mca_ckas_com

# Xử lý và tạo M-code
python src/main.py --input shared_data/dynamics_data.csv

# Chạy tests
python -m pytest tests/

# Validation output
python src/main.py --validate shared_data/ckas_commands.m
```

## 🔗 Dependencies

- numpy
- scipy
- pandas
- matplotlib (for visualization)
- pyyaml
- pytest

## 📊 Bộ Lọc Washout

**Công thức Washout Filter:**

```
H(s) = (s * T_w) / (s + 1/T_w) * K

Tham số:
- T_w: Time constant (thường 1-5 giây)
- K: Gain factor (0.0-1.0)
```

**Mục đích:**
- Loại bỏ các chuyển động quá chậm (DC components)
- Giảm các chuyển động quá nhanh không cần thiết
- Tối ưu hóa trải nghiệm trên bệ mô phỏng
- Giảm sự mệt mỏi của người tham gia

## 🎯 Quá Trình Xử Lý

```
dynamics_data.csv
    ↓
Tách tín hiệu (X, Y, Z, Roll, Pitch, Yaw)
    ↓
Áp dụng Washout Filter
    ↓
Chuẩn hóa dữ liệu
    ↓
Kiểm tra giới hạn bệ
    ↓
Tạo M-code
    ↓
ckas_commands.m
```

## ✅ Validation Criteria

- ✓ Amplitude không vượt quá limit bệ
- ✓ Frequency nằm trong dải cho phép
- ✓ Signal continuity (không có jumps)
- ✓ Timing consistency

## 🔄 Kết Nối với Module Khác

← **Module 2 (Scenario Manager)** cung cấp dynamics_data.csv
→ **CKAS Motion Platform** nhận M-code commands

---

**Last Updated:** 2026-05-18
