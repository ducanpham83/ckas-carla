import carla
import time
import math
import socket
import csv
import threading
import queue
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import json
import os
import numpy as np

class GenericIIRFilter:
    """Bộ lọc số tỷ lệ tuyến tính đáp ứng mọi bậc cấu hình (Direct Form I)"""
    def __init__(self, b, a):
        self.b = np.array(b)
        self.a = np.array(a)
        self.x = np.zeros(len(b))
        self.y = np.zeros(len(a))

    def process(self, x_in):
        # Dịch bộ nhớ tín hiệu vào
        self.x = np.roll(self.x, 1)
        self.x[0] = x_in
        
        # Tính toán phương trình sai phân tổng quát N-bậc
        val = np.sum(self.b * self.x) - np.sum(self.a[1:] * self.y[1:])
        val /= self.a[0]
        
        # Dịch bộ nhớ tín hiệu ra
        self.y = np.roll(self.y, 1)
        self.y[0] = val
        return val

class AdvancedWashoutFilter:
    def __init__(self, dt=0.01, order=3, params=None):
        self.dt = dt
        self.order = order # 2 hoặc 3
        self.g = 9.81
        
        # Cấu trúc tham số: [Gain (K), Omega_n, Zeta, Omega_b (Washout Freq)]
        self.p = params if params else {
            'hp_x': [1.0, 1.5, 1.0, 1.0], 'hp_y': [1.0, 1.5, 1.0, 1.0], 'hp_z': [1.0, 2.0, 1.0, 1.0],
            'hp_roll': [1.0, 1.0, 1.0, 1.0], 'hp_pitch': [1.0, 1.0, 1.0, 1.0], 'hp_yaw': [1.0, 0.8, 1.0, 1.0],
            'lp_tilt_x': [1.0, 0.5, 1.0, 1.0], 'lp_tilt_y': [1.0, 0.5, 1.0, 1.0]
        }
        
        self.pos = np.zeros(3)
        self.rot = np.zeros(3)
        self._build_filter_coefficients()

    def _poly_pow(self, p, n):
        """Hàm hỗ trợ: Tính lũy thừa của đa thức p^n"""
        if n == 0:
            return np.array([1.0])
        res = p
        for _ in range(1, n):
            res = np.polymul(res, p)
        return res

    def _bilinear_transform(self, num_s, den_s):
        c = 2.0 / self.dt
        z_p = np.array([1.0, 1.0])   # (z + 1)
        z_m = np.array([1.0, -1.0])  # (z - 1)
        
        deg = max(len(num_s), len(den_s)) - 1
        b_z = np.zeros(deg + 1)
        a_z = np.zeros(deg + 1)
        
        for i, coeff in enumerate(reversed(num_s)):
            p = self._poly_pow(z_m, i)
            q = self._poly_pow(z_p, deg - i)
            poly = np.polymul(p, q) * coeff * (c ** i)
            b_z = np.polyadd(b_z, poly)
            
        for i, coeff in enumerate(reversed(den_s)):
            p = self._poly_pow(z_m, i)
            q = self._poly_pow(z_p, deg - i)
            poly = np.polymul(p, q) * coeff * (c ** i)
            a_z = np.polyadd(a_z, poly)
            
        return b_z, a_z

    def _build_filter_coefficients(self):
        self.filters = {}
        for key, val in self.p.items():
            k, wn, zeta, wb = val[0], val[1], val[2], val[3]
            
            if self.order == 2:
                if 'hp' in key: 
                    # LHP/AHP Bậc 2: H(s) = K*s^2 / (s^2 + 2zw_n s + w_n^2)
                    num_s = [k, 0.0, 0.0]
                    den_s = [1.0, 2 * zeta * wn, wn ** 2]
                else: 
                    # LP Tilt Bậc 2: H(s) = K*w_n^2 / (s^2 + 2zw_n s + w_n^2)
                    num_s = [0.0, 0.0, k * (wn ** 2)]
                    den_s = [1.0, 2 * zeta * wn, wn ** 2]
            else: # Bậc 3 (Classical 3rd Order)
                if 'hp' in key:
                    # LHP/AHP Bậc 3: H(s) = K*s^3 / [ (s^2 + 2zw_n s + w_n^2)(s + w_b) ]
                    num_s = [k, 0.0, 0.0, 0.0]
                    den_s = [1.0, (2 * zeta * wn) + wb, (wn ** 2) + (2 * zeta * wn * wb), (wn ** 2) * wb]
                else:
                    # LP Tilt Bậc 3: H(s) = K*w_n^2*w_b / [ (s^2 + 2zw_n s + w_n^2)(s + w_b) ]
                    num_s = [0.0, 0.0, 0.0, k * (wn ** 2) * wb]
                    den_s = [1.0, (2 * zeta * wn) + wb, (wn ** 2) + (2 * zeta * wn * wb), (wn ** 2) * wb]
                    
            b, a = self._bilinear_transform(num_s, den_s)
            self.filters[key] = GenericIIRFilter(b, a)

    def process(self, accel, ang_vel):
        self.pos[0] = self.filters['hp_x'].process(accel[0])
        self.pos[1] = self.filters['hp_y'].process(accel[1])
        self.pos[2] = self.filters['hp_z'].process(accel[2])
        
        hp_roll = self.filters['hp_roll'].process(ang_vel[0])
        hp_pitch = self.filters['hp_pitch'].process(ang_vel[1])
        hp_yaw = self.filters['hp_yaw'].process(ang_vel[2])
        
        tilt_pitch = self.filters['lp_tilt_x'].process(accel[0] / self.g)
        tilt_roll = self.filters['lp_tilt_y'].process(-accel[1] / self.g)
        
        self.rot[0] = hp_roll + tilt_roll
        self.rot[1] = hp_pitch + tilt_pitch
        self.rot[2] = hp_yaw
        
        return [self.pos[0], self.pos[1], self.pos[2], self.rot[0], self.rot[1], self.rot[2]]

class FilterParamDialog(tk.Toplevel):
    def __init__(self, parent, current_params, callback):
        super().__init__(parent)
        self.title("Cấu hình Ma trận Tham số CWF Bậc 2/3")
        self.geometry("750x450") # Nới rộng bề ngang
        self.callback = callback
        self.params = current_params.copy()
        self.entries = {}
        
        self._build_layout()
        
    def _build_layout(self):
        main_frame = ttk.Frame(self, padding=15)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header danh mục 5 cột
        ttk.Label(main_frame, text="Kênh Xử Lý", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky="w", padx=5)
        ttk.Label(main_frame, text="Hệ số Gain (K)", font=("Arial", 10, "bold")).grid(row=0, column=1, padx=5)
        ttk.Label(main_frame, text="Tần số Tự nhiên ωn", font=("Arial", 10, "bold")).grid(row=0, column=2, padx=5)
        ttk.Label(main_frame, text="Hệ số Cản ζ", font=("Arial", 10, "bold")).grid(row=0, column=3, padx=5)
        ttk.Label(main_frame, text="Tần số Washout ωb", font=("Arial", 10, "bold")).grid(row=0, column=4, padx=5)
        
        # Sinh lưới nhập liệu 4 chiều
        for idx, (key, val) in enumerate(self.params.items(), start=1):
            ttk.Label(main_frame, text=f"{key.upper()}:").grid(row=idx, column=0, sticky="w", pady=6, padx=5)
            
            k_var = tk.DoubleVar(value=val[0])
            ttk.Spinbox(main_frame, from_=0.01, to=10.0, increment=0.05, textvariable=k_var, width=10).grid(row=idx, column=1, padx=5)
            
            wn_var = tk.DoubleVar(value=val[1])
            ttk.Spinbox(main_frame, from_=0.01, to=20.0, increment=0.05, textvariable=wn_var, width=10).grid(row=idx, column=2, padx=5)
            
            zeta_var = tk.DoubleVar(value=val[2])
            ttk.Spinbox(main_frame, from_=0.01, to=5.0, increment=0.05, textvariable=zeta_var, width=10).grid(row=idx, column=3, padx=5)

            wb_var = tk.DoubleVar(value=val[3])
            ttk.Spinbox(main_frame, from_=0.01, to=20.0, increment=0.05, textvariable=wb_var, width=10).grid(row=idx, column=4, padx=5)
            
            self.entries[key] = (k_var, wn_var, zeta_var, wb_var)
            
        btn_frame = ttk.Frame(self, padding=10)
        btn_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        ttk.Button(btn_frame, text="💾 Lưu JSON", command=self._save_to_json).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="📂 Tải JSON", command=self._load_from_json).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="✔ Áp dụng Cấu hình", command=self._apply_params).pack(side=tk.RIGHT, padx=5)

    def _apply_params(self):
        for key in self.entries:
            self.params[key] = [
                self.entries[key][0].get(), self.entries[key][1].get(),
                self.entries[key][2].get(), self.entries[key][3].get()
            ]
        self.callback(self.params)
        self.destroy()

    def _save_to_json(self):
        current_data = {key: [e[0].get(), e[1].get(), e[2].get(), e[3].get()] for key, e in self.entries.items()}
        file_path = tk.filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON Files", "*.json")])
        if file_path:
            with open(file_path, 'w') as f: json.dump(current_data, f, indent=4)
            messagebox.showinfo("Thành công", "Đã xuất file thông số cấu hình bộ lọc!")

    def _load_from_json(self):
        file_path = tk.filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
        if file_path:
            with open(file_path, 'r') as f: loaded_data = json.load(f)
            for key, val in loaded_data.items():
                if key in self.entries and len(val) == 4:
                    for i in range(4): self.entries[key][i].set(val[i])
            messagebox.showinfo("Thành công", "Đã nạp bộ dữ liệu thông số!")

# --- MODULE CẤU HÌNH & TRUYỀN THÔNG ---
UDP_IP = "192.168.10.2"
UDP_PORT = 1288
HZ = 100
DT = 1.0 / HZ

class AsyncCSVLogger:
    def __init__(self, filename, headers):
        self.q = queue.Queue()
        self.filename = filename
        self.headers = headers
        self.running = True
        self.thread = threading.Thread(target=self._writer_thread, daemon=True)
        self.thread.start()

    def _writer_thread(self):
        with open(self.filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)
            while self.running or not self.q.empty():
                try:
                    row = self.q.get(timeout=0.1)
                    writer.writerow(row)
                except queue.Empty:
                    continue
    def log(self, data): self.q.put(data)
    def stop(self): self.running = False; self.thread.join()


class CarlaNode:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("PC1: CARLA Pro Simulation Testbench")
        
        self.vars = {
            "pos": tk.BooleanVar(value=True), "vel": tk.BooleanVar(value=True),
            "accel": tk.BooleanVar(value=True), "rot": tk.BooleanVar(value=True),
            "ang_vel": tk.BooleanVar(value=True), "ang_accel": tk.BooleanVar(value=True)
        }
        
        self.town_var = tk.StringVar(value="Town04")
        self.weather_var = tk.StringVar(value="ClearNoon")
        self.spawn_var = tk.IntVar(value=10)
        self.dist_var = tk.DoubleVar(value=500.0)
        self.vmax_var = tk.DoubleVar(value=90.0)
        self.amax_var = tk.DoubleVar(value=5.0)
        self.jmax_var = tk.DoubleVar(value=2.0)
        self.exact_stop_var = tk.BooleanVar(value=True) 
        
        self.cam_x = tk.DoubleVar(value=0.4)
        self.cam_y = tk.DoubleVar(value=-0.3) 
        self.cam_z = tk.DoubleVar(value=1.3)
        self.cam_fov = tk.DoubleVar(value=90.0)
        
        self.display_mode = tk.StringVar(value="3D_CARLA") 
        self.plot_mode = tk.StringVar(value="2D")
        self.dof_mode = tk.IntVar(value=3)
        self.udp_mode = tk.StringVar(value="15_param")
        
        # Cấu hình chướng ngại vật
        self.slalom_enabled = tk.BooleanVar(value=False)
        self.slalom_spacing = 30.0
        self.slalom_offset = 2.0

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.config = {}
        
        # Phục vụ tính năng Preview
        self.preview_active = False
        self.preview_client = None
        self.preview_vehicle = None
        self.preview_camera = None
        # Thêm biến theo dõi cấu hình bộ lọc nâng cao
        self.filter_order_var = tk.IntVar(value=3) # Mặc định bậc 3 chuyên sâu
       # Cấu trúc: [Gain(K), Omega_n, Zeta, Omega_b]
        self.mca_params = {
            'hp_x': [1.0, 1.5, 1.0, 1.0], 'hp_y': [1.0, 1.5, 1.0, 1.0], 'hp_z': [1.0, 2.0, 1.0, 1.0],
            'hp_roll': [1.0, 1.0, 1.0, 1.0], 'hp_pitch': [1.0, 1.0, 1.0, 1.0], 'hp_yaw': [1.0, 0.8, 1.0, 1.0],
            'lp_tilt_x': [1.0, 0.5, 1.0, 1.0], 'lp_tilt_y': [1.0, 0.5, 1.0, 1.0]
        }
        self._build_gui()
        self.root.mainloop()

    def _build_gui(self):
        # 1. Khung Môi trường & Động học
        f1 = ttk.LabelFrame(self.root, text="Thiết lập Môi trường & Động học", padding=10)
        f1.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(f1, text="Bản đồ (Town):").grid(row=0, column=0, sticky="w")
        ttk.Combobox(f1, textvariable=self.town_var, values=["Town01", "Town02", "Town03", "Town04", "Town05", "Town10HD"], width=15).grid(row=0, column=1)
        ttk.Label(f1, text="Thời tiết:").grid(row=0, column=2, sticky="e", padx=5)
        ttk.Combobox(f1, textvariable=self.weather_var, values=["ClearNoon", "CloudyNoon", "WetNoon", "HardRainNoon"], width=12).grid(row=0, column=3)

        ttk.Label(f1, text="Quãng đường (m):").grid(row=1, column=0, sticky="w")
        ttk.Entry(f1, textvariable=self.dist_var, width=18).grid(row=1, column=1)
        ttk.Label(f1, text="Vận tốc Max (km/h):").grid(row=1, column=2, sticky="e", padx=5)
        ttk.Entry(f1, textvariable=self.vmax_var, width=15).grid(row=1, column=3)

        ttk.Label(f1, text="Gia tốc Max (m/s²):").grid(row=2, column=0, sticky="w")
        ttk.Entry(f1, textvariable=self.amax_var, width=18).grid(row=2, column=1)
        ttk.Label(f1, text="Jerk Max (m/s³):").grid(row=2, column=2, sticky="e", padx=5)
        ttk.Entry(f1, textvariable=self.jmax_var, width=15).grid(row=2, column=3)
        ttk.Checkbutton(f1, text="Tự động Giảm tốc Khẩn cấp dừng CÓ VẠCH ĐÍCH", variable=self.exact_stop_var).grid(row=3, column=0, columnspan=2, pady=5, sticky="w")
        
        btn_slalom = ttk.Button(f1, text="🚧 Thiết lập Zigzag (Slalom)", command=self.setup_slalom)
        btn_slalom.grid(row=3, column=2, columnspan=2, pady=5)

        # 2. Khung Góc nhìn Camera
        f_cam = ttk.LabelFrame(self.root, text="Cấu hình Góc nhìn (Ghế lái) - Cố định Rigid", padding=10)
        f_cam.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(f_cam, text="Mắt X (m):").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(f_cam, from_=-2.0, to=2.0, increment=0.1, textvariable=self.cam_x, width=8, command=self.update_preview_cam).grid(row=0, column=1, padx=5)
        ttk.Label(f_cam, text="Mắt Y (m):").grid(row=0, column=2, sticky="w")
        ttk.Spinbox(f_cam, from_=-1.0, to=1.0, increment=0.1, textvariable=self.cam_y, width=8, command=self.update_preview_cam).grid(row=0, column=3, padx=5)
        ttk.Label(f_cam, text="Mắt Z (m):").grid(row=0, column=4, sticky="w")
        ttk.Spinbox(f_cam, from_=0.5, to=3.0, increment=0.1, textvariable=self.cam_z, width=8, command=self.update_preview_cam).grid(row=0, column=5, padx=5)
        
        ttk.Label(f_cam, text="FOV (Độ):").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Spinbox(f_cam, from_=30.0, to=120.0, increment=5.0, textvariable=self.cam_fov, width=8, command=self.update_preview_cam).grid(row=1, column=1, padx=5, pady=5)
        
        self.btn_preview = ttk.Button(f_cam, text="👁 Bật Preview 3D Live", command=self.toggle_preview)
        self.btn_preview.grid(row=1, column=3, columnspan=3, pady=5)

        # 3. Khung Chế độ
        f3 = ttk.LabelFrame(self.root, text="Hiển thị, Bản đồ & UDP", padding=10)
        f3.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(f3, text="Trực tiếp lúc chạy:").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(f3, text="3D CARLA", variable=self.display_mode, value="3D_CARLA").grid(row=0, column=1, sticky="w")
        ttk.Radiobutton(f3, text="Radar 2D", variable=self.display_mode, value="2D_Live").grid(row=0, column=2, sticky="w")
        
        ttk.Label(f3, text="Vẽ Biểu đồ sau cùng:").grid(row=1, column=0, sticky="w")
        ttk.Radiobutton(f3, text="Vẽ 2D + V/A(xyz)", variable=self.plot_mode, value="2D").grid(row=1, column=1, sticky="w")
        ttk.Radiobutton(f3, text="Vẽ 3D + V/A(xyz)", variable=self.plot_mode, value="3D").grid(row=1, column=2, sticky="w")
        
        ttk.Label(f3, text="Thuật toán & Phân cấp bậc:").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(f3, text="Bypass (Truyền thẳng trực tiếp)", variable=self.dof_mode, value=0).grid(row=0, column=1, sticky="w")
        ttk.Radiobutton(f3, text="Bộ lọc số tích hợp", variable=self.dof_mode, value=1).grid(row=0, column=2, sticky="w")

        ttk.Label(f3, text="Bậc bộ lọc số (CWF Order):").grid(row=1, column=0, sticky="w")
        ttk.Radiobutton(f3, text="Bậc 2 (Hàm truyền chuẩn)", variable=self.filter_order_var, value=2).grid(row=1, column=1, sticky="w")
        ttk.Radiobutton(f3, text="Bậc 3 (Butterworth chống rung tốt)", variable=self.filter_order_var, value=3).grid(row=1, column=2, sticky="w")

        # Nút gọi Dialog ngoài
        ttk.Button(f3, text="⚙ Hiệu chỉnh ma trận tham số lọc", command=self._open_param_dialog).grid(row=2, column=1, columnspan=2, pady=5, sticky="ew")

        f_btns = ttk.Frame(self.root, padding=10)
        f_btns.pack(fill="x")
        ttk.Button(f_btns, text="▶ BẮT ĐẦU MÔ PHỎNG", command=self.save_and_start).pack(side=tk.LEFT, padx=10, expand=True, fill="x")
        ttk.Button(f_btns, text="🔄 Xem lại (Replay Log)", command=self.replay_simulation).pack(side=tk.RIGHT, padx=10, expand=True, fill="x")

    def setup_slalom(self):
        """Hộp thoại cấu hình chướng ngại vật"""
        ans = messagebox.askyesno("Zigzag Setup", "Bạn có muốn bật chướng ngại vật Nón Giao Thông (Slalom) không?")
        self.slalom_enabled.set(ans)
        if ans:
            sp = simpledialog.askfloat("Khoảng cách", "Nhập khoảng cách giữa 2 nón (m):", initialvalue=30.0)
            off = simpledialog.askfloat("Độ lệch", "Nhập độ lệch so với tâm đường (m):", initialvalue=2.0)
            if sp: self.slalom_spacing = sp
            if off: self.slalom_offset = off
            messagebox.showinfo("Đã lưu", f"Đã bật Slalom: Khoảng cách {self.slalom_spacing}m, lệch {self.slalom_offset}m")

    # ==========================================
    # CÁC HÀM PREVIEW CAMERA (LIVE UPDATE)
    # ==========================================
    def toggle_preview(self):
        if not self.preview_active:
            self.start_preview()
            self.btn_preview.config(text="⏹ Tắt Preview 3D")
        else:
            self.stop_preview()
            self.btn_preview.config(text="👁 Bật Preview 3D Live")

    def start_preview(self):
        self.preview_active = True
        self.preview_client = carla.Client('127.0.0.1', 2000)
        self.preview_client.set_timeout(60.0)
        
        # --- BỔ SUNG ĐOẠN NÀY ĐỂ PHÁ DEADLOCK ---
        try:
            print("[HỆ THỐNG] Đang kiểm tra trạng thái Server...")
            current_world = self.preview_client.get_world()
            settings = current_world.get_settings()
            # Nếu server đang bị kẹt ở chế độ đồng bộ từ lần chạy trước -> Tắt nó đi
            if settings.synchronous_mode:
                print("[HỆ THỐNG] Phát hiện Server bị kẹt Đồng bộ. Đang giải phóng...")
                settings.synchronous_mode = False
                current_world.apply_settings(settings)
        except Exception as e:
            pass # Bỏ qua nếu không kết nối được
        # ----------------------------------------
        
        print(f"[Preview] Đang tải bản đồ {self.town_var.get()}... Vui lòng đợi.")
        world = self.preview_client.load_world(self.town_var.get())
        world.set_weather(getattr(carla.WeatherParameters, self.weather_var.get()))
        
        bp_lib = world.get_blueprint_library()
        vehicle_bp = bp_lib.filter('vehicle.tesla.model3')[0]
        sp = world.get_map().get_spawn_points()[self.spawn_var.get() % len(world.get_map().get_spawn_points())]
        self.preview_vehicle = world.spawn_actor(vehicle_bp, sp)
        
        # Tạo cảm biến camera cố định
        cam_bp = bp_lib.find('sensor.camera.rgb')
        cam_bp.set_attribute('fov', str(self.cam_fov.get()))
        cam_transform = carla.Transform(carla.Location(x=self.cam_x.get(), y=self.cam_y.get(), z=self.cam_z.get()))
        
        # AttachmentType.Rigid: Đảm bảo camera KHÔNG bị rung bởi hệ thống treo của xe
        self.preview_camera = world.spawn_actor(cam_bp, cam_transform, attach_to=self.preview_vehicle, attachment_type=carla.AttachmentType.Rigid)
        
        world.get_spectator().set_transform(self.preview_camera.get_transform())
        
        # Khởi động vòng lặp tick ngầm bằng root.after để GUI không bị đơ
        self.root.after(100, self.preview_loop)

    def preview_loop(self):
        if self.preview_active and self.preview_client:
            world = self.preview_client.get_world()
            # Liên tục ép spectator nhìn qua camera
            world.get_spectator().set_transform(self.preview_camera.get_transform())
            self.root.after(100, self.preview_loop)

    def update_preview_cam(self):
        """Được gọi mỗi khi bấm nút mũi tên trên Spinbox"""
        if self.preview_active and self.preview_camera:
            cam_transform = carla.Transform(carla.Location(x=self.cam_x.get(), y=self.cam_y.get(), z=self.cam_z.get()))
            self.preview_camera.set_transform(cam_transform)
            # Không thể đổi FOV của actor đang sống, phải tạo lại. Tạm thời bỏ qua FOV live update để tránh lag.

    def stop_preview(self):
        self.preview_active = False
        if self.preview_camera: self.preview_camera.destroy()
        if self.preview_vehicle: self.preview_vehicle.destroy()
        self.preview_client = None
    def _open_param_dialog(self):
        FilterParamDialog(self.root, self.mca_params, self._update_mca_params)

    def _update_mca_params(self, new_params):
        self.mca_params = new_params
        print("[HỆ THỐNG] Đã cập nhật tham số lọc từ Dialog ngoại.")
    # ==========================================
    # LUỒNG MÔ PHỎNG CHÍNH
    # ==========================================
    def save_and_start(self):
        if self.preview_active: self.stop_preview()
      # --- CẬP NHẬT LẠI KHỐI NÀY ---
        self.config = {
            "town": self.town_var.get(), "weather": getattr(carla.WeatherParameters, self.weather_var.get()),
            "spawn_idx": self.spawn_var.get(), "max_dist": self.dist_var.get(),
            "v_max_mps": self.vmax_var.get() / 3.6, "a_max": self.amax_var.get(),
            "j_max": self.jmax_var.get(), "exact_stop": self.exact_stop_var.get(),
            "cam_x": self.cam_x.get(), "cam_y": self.cam_y.get(), 
            "cam_z": self.cam_z.get(), "cam_fov": self.cam_fov.get(),
            "display_mode": self.display_mode.get(), "plot_mode": self.plot_mode.get(),
            "dof_mode": self.dof_mode.get(),
            "udp_mode": self.udp_mode.get(),                  # Thêm dòng này
            "filter_order": self.filter_order_var.get()       # Thêm dòng này
        }
        
        self.root.withdraw()
        self.root.update()
        
        try:
            self.run_carla()
        except Exception as e:
            print(f"\n[LỖI CARLA] {e}")
        finally:
            self.root.deiconify()

    def replay_simulation(self):
        """Phát lại file log mô phỏng gần nhất"""
        client = carla.Client('127.0.0.1', 2000)
        client.set_timeout(10.0)
        print("\n[REPLAY] Đang phát lại mô phỏng trước đó...")
        # Lệnh replay_file tự động tua lại mọi actor, xe, chướng ngại vật như cũ
        client.replay_file("latest_sim.log", 0, 0, 0)
        messagebox.showinfo("Replay", "Đang phát lại trong cửa sổ CARLA. Xem xong bấm OK.")
        client.stop_replayer()

    def plot_trajectory_offline(self, h_t, h_x, h_y, h_z, h_v, h_ax, h_ay, h_az):
        """Vẽ biểu đồ tích hợp với 3 thành phần gia tốc rõ ràng"""
        mode = self.config['plot_mode']
        if mode == "None" or len(h_x) == 0: return
        
        fig = plt.figure(figsize=(15, 8))
        fig.suptitle(f"PHÂN TÍCH ĐỘNG HỌC CARLA - {self.config['town']}", fontsize=16, fontweight='bold')
        
        h_a_mag = [math.sqrt(x**2 + y**2 + z**2) for x, y, z in zip(h_ax, h_ay, h_az)]

        if mode == "3D":
            ax1 = fig.add_subplot(1, 2, 1, projection='3d')
            sc = ax1.scatter(h_x, h_y, h_z, c=h_a_mag, cmap='coolwarm', s=15, edgecolor='none')
            ax1.set_zlabel('Độ cao Z (m)')
        else:
            ax1 = fig.add_subplot(1, 2, 1)
            sc = ax1.scatter(h_x, h_y, c=h_a_mag, cmap='coolwarm', s=15, edgecolor='none')
            ax1.axis('equal') 

        ax1.set_title("Quỹ đạo di chuyển (Màu = Độ lớn Gia tốc)", fontsize=12)
        ax1.set_xlabel('Tọa độ X (m)'); ax1.set_ylabel('Tọa độ Y (m)')
        ax1.grid(True, linestyle='--', alpha=0.6)
        cbar = fig.colorbar(sc, ax=ax1, shrink=0.7)

        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(h_t, h_v, 'k-', lw=2)
        ax2.set_title('Đồ thị Vận tốc / Thời gian', fontsize=12)
        ax2.set_ylabel('Vận tốc (m/s)')
        ax2.grid(True, linestyle=':', alpha=0.7)

        # Đồ thị 3 trục gia tốc
        ax3 = fig.add_subplot(2, 2, 4, sharex=ax2)
        ax3.plot(h_t, h_ax, 'r-', lw=1.5, label='Ax (Surge/Phanh)')
        ax3.plot(h_t, h_ay, 'g-', lw=1.5, label='Ay (Sway/Cua)')
        ax3.plot(h_t, h_az, 'b-', lw=1.5, alpha=0.5, label='Az (Heave/Nhấp nhô)')
        ax3.axhline(y=self.config['a_max'], color='k', linestyle='--', alpha=0.5)
        ax3.axhline(y=-self.config['a_max'], color='k', linestyle='--', alpha=0.5)
        
        ax3.set_title('Gia tốc Tịnh tiến (X, Y, Z)', fontsize=12)
        ax3.set_xlabel('Thời gian mô phỏng (s)')
        ax3.set_ylabel('Gia tốc (m/s²)')
        ax3.legend(loc='upper right')
        ax3.grid(True, linestyle=':', alpha=0.7)

        plt.tight_layout()
        plt.show()

    def spawn_track_elements(self, world, vehicle):
        """Hàm sinh vạch đích và vật cản zigzag"""
        c_map = world.get_map()
        start_wp = c_map.get_waypoint(vehicle.get_location())
        bp_cone = world.get_blueprint_library().find('static.prop.constructioncone')
        
        # Sinh vạch đích
        finish_wps = start_wp.next(self.config['max_dist'])
        if finish_wps:
            f_wp = finish_wps[0]
            # Tạo một hàng nón chắn ngang đường làm vạch đích
            right_v = f_wp.transform.get_right_vector()
            for i in range(-3, 4):
                loc = f_wp.transform.location + right_v * (i * 1.0)
                world.spawn_actor(bp_cone, carla.Transform(loc))

        # Sinh Zigzag
        if self.slalom_enabled.get():
            curr_wp = start_wp
            dist_accum = 0.0
            side = 1.0
            while dist_accum < self.config['max_dist'] - 20: # Chừa 20m cuối để phanh
                next_wps = curr_wp.next(self.slalom_spacing)
                if not next_wps: break
                curr_wp = next_wps[0]
                dist_accum += self.slalom_spacing
                
                # Đặt nón lệch sang trái hoặc phải
                right_v = curr_wp.transform.get_right_vector()
                cone_loc = curr_wp.transform.location + right_v * (self.slalom_offset * side)
                world.spawn_actor(bp_cone, carla.Transform(cone_loc))
                side *= -1.0 # Đảo chiều zigzag

    def run_carla(self):
        client = carla.Client('127.0.0.1', 2000)
        client.set_timeout(30.0)
        world = client.load_world(self.config['town']) 
        world.set_weather(self.config['weather'])
        
        # Kích hoạt Recorder để phục vụ Replay
        client.start_recorder("latest_sim.log")

        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = DT
        settings.no_rendering_mode = (self.config['display_mode'] != "3D_CARLA")
        world.apply_settings(settings)

        tm = client.get_trafficmanager(8000)
        tm.set_synchronous_mode(True)
        tm.global_percentage_speed_difference(0.0) 
        
        bp_lib = world.get_blueprint_library()
        vehicle_bp = bp_lib.filter('vehicle.tesla.model3')[0]
        sp_list = world.get_map().get_spawn_points()
        vehicle = world.spawn_actor(vehicle_bp, sp_list[self.config['spawn_idx'] % len(sp_list)])
        vehicle.set_autopilot(True, tm.get_port())
        
        # Sinh vạch đích và zigzag
        self.spawn_track_elements(world, vehicle)
        
        # TẠO CAMERA CỐ ĐỊNH CHỐNG RUNG (RIGID) CHO CHẾ ĐỘ 3D
        drive_cam = None
        spectator = world.get_spectator()
        if self.config['display_mode'] == "3D_CARLA":
            cam_bp = bp_lib.find('sensor.camera.rgb')
            cam_bp.set_attribute('fov', str(self.config['cam_fov']))
            cam_trans = carla.Transform(carla.Location(x=self.config['cam_x'], y=self.config['cam_y'], z=self.config['cam_z']))
            drive_cam = world.spawn_actor(cam_bp, cam_trans, attach_to=vehicle, attachment_type=carla.AttachmentType.Rigid)
        
        headers = ["t_sim", "x", "y", "z", "vx", "vy", "vz", "ax", "ay", "az", "roll", "pitch", "yaw", "wx", "wy", "wz", "alpha_x", "alpha_y", "alpha_z"]
        logger = AsyncCSVLogger("carla_telemetry.csv", headers)
        
        # ĐÃ SỬA THÀNH ADVANCED WASHOUT FILTER
        washout = AdvancedWashoutFilter(dt=DT, order=self.config['filter_order'], params=self.mca_params)
        
        t_sim, total_dist = 0.0, 0.0
        start_loc, prev_loc = None, None
        prev_a_filtered = np.zeros(3)
        prev_ang_vel = carla.Vector3D(0,0,0)
        is_braking = False
        loop_count = 0
        
        hist_t, hist_x, hist_y, hist_z, hist_v, hist_ax, hist_ay, hist_az = [], [], [], [], [], [], [], []
        live_x, live_y = [], []

        fig_live, ax_live, line_live, marker_live = None, None, None, None
        if self.config['display_mode'] == "2D_Live":
            plt.ion() 
            fig_live, ax_live = plt.subplots(figsize=(6, 6))
            ax_live.set_title("Live Radar 2D")
            line_live, = ax_live.plot([], [], 'b-', linewidth=2, alpha=0.5)
            marker_live, = ax_live.plot([], [], 'ro', markersize=8)
            plt.show(block=False)

        print("[CARLA] Bắt đầu mô phỏng 100Hz...")
        
        try:
            while True:
                world.tick()
                t_sim += DT; loop_count += 1
                
                t = vehicle.get_transform()
                v = vehicle.get_velocity()
                a_raw = vehicle.get_acceleration()
                w = vehicle.get_angular_velocity()
                loc = t.location
                speed = math.sqrt(v.x**2 + v.y**2 + v.z**2)

                if start_loc is None: start_loc = loc
                if prev_loc is not None: total_dist += loc.distance(prev_loc)
                prev_loc = loc

                # Clamper giới hạn động học
                alpha_x = (w.x - prev_ang_vel.x) / DT
                alpha_y = (w.y - prev_ang_vel.y) / DT
                alpha_z = (w.z - prev_ang_vel.z) / DT
                prev_ang_vel = w
                
                a_current = np.array([a_raw.x, a_raw.y, a_raw.z])
                da_clamped = np.clip(a_current - prev_a_filtered, -self.config['j_max'] * DT, self.config['j_max'] * DT)
                a_final = np.clip(prev_a_filtered + da_clamped, -self.config['a_max'], self.config['a_max'])
                prev_a_filtered = a_final
                ax_c, ay_c, az_c = a_final[0], a_final[1], a_final[2]

                # Lưu mảng vẽ biểu đồ
                hist_t.append(t_sim); hist_v.append(speed)
                hist_ax.append(ax_c); hist_ay.append(ay_c); hist_az.append(az_c)
                hist_x.append(loc.x); hist_y.append(loc.y); hist_z.append(loc.z)

                # CẬP NHẬT CAMERA CỐ ĐỊNH (Không cần tự tính, chỉ bắt spectator nhìn qua camera)
                if self.config['display_mode'] == "3D_CARLA" and drive_cam:
                    spectator.set_transform(drive_cam.get_transform())

                if self.config['display_mode'] == "2D_Live" and loop_count % 10 == 0:
                    live_x.append(loc.x); live_y.append(loc.y)
                    line_live.set_data(live_x, live_y)
                    marker_live.set_data([loc.x], [loc.y])
                    ax_live.relim(); ax_live.autoscale_view()
                    fig_live.canvas.draw_idle(); fig_live.canvas.flush_events()

                # Phanh khẩn cấp trước vạch đích
                if self.config['exact_stop'] and not is_braking:
                    dist_rem = self.config['max_dist'] - total_dist
                    if self.config['a_max'] > 0:
                        d_brake = (speed**2) / (2.0 * self.config['a_max'])
                        if dist_rem <= d_brake and speed > 0.5:
                            is_braking = True
                            vehicle.set_autopilot(False)

                if is_braking:
                    control = vehicle.get_control()
                    control.throttle = 0.0; control.brake = 1.0
                    vehicle.apply_control(control)
                    if speed < 0.1: break
                elif total_dist >= self.config['max_dist']:
                    break
                
                logger.log([t_sim, loc.x, loc.y, loc.z, v.x, v.y, v.z, ax_c, ay_c, az_c, t.rotation.roll, t.rotation.pitch, t.rotation.yaw, w.x, w.y, w.z, alpha_x, alpha_y, alpha_z])
                
                # ========================================================
                # 4. ĐIỀU PHỐI PAYLOAD MẠNG UDP THEO CHUẨN CKAS ICD
                # ========================================================
                send_timestamp = time.time() # Chốt thời gian gửi để PC2 tính Jitter
                
                # Tính toán qua bộ lọc Washout (nếu không chọn Bypass)
                if self.config['dof_mode'] != 0:
                    wo_out = washout.process([ax_c, ay_c, az_c], [math.radians(w.x), math.radians(w.y), math.radians(w.z)])
                
                # KIỂM TRA TÙY CHỌN ĐỊNH DẠNG GÓI TIN TRÊN GUI
                if self.config['udp_mode'] == "15_param":
                    # --------------------------------------------------------
                    # CHẾ ĐỘ 1: TRUYỀN RAW TELEMETRY TỪ CARLA (15 Tham số)
                    # Thường dùng để PC2 nhận dữ liệu thô và tự tính toán Washout bên ngoài
                    # --------------------------------------------------------
                    payload = (
                        f"{t_sim:.3f},"                           # 1. Thời gian mô phỏng (s)
                        f"{loc.x:.3f},{loc.y:.3f},{loc.z:.3f},"   # 2,3,4. Vị trí tuyệt đối X, Y, Z (m)
                        f"{v.x:.3f},{v.y:.3f},{v.z:.3f},"         # 5,6,7. Vận tốc X, Y, Z (m/s)
                        f"{ax_c:.3f},{ay_c:.3f},{az_c:.3f},"      # 8,9,10. Gia tốc tịnh tiến sau Clamper (m/s^2)
                        f"{t.rotation.roll:.3f},{t.rotation.pitch:.3f},{t.rotation.yaw:.3f}," # 11,12,13. Góc xoay Euler (Độ)
                        f"{w.x:.3f},{w.y:.3f},"                   # 14,15. Vận tốc góc Roll rate, Pitch rate (Độ/s)
                        f"{send_timestamp:.6f}"                   # [Ẩn] Timestamp để đo độ trễ mạng
                    )
                else:
                    # --------------------------------------------------------
                    # CHẾ ĐỘ 2: TRUYỀN DỮ LIỆU ĐIỀU KHIỂN BỆ CKAS (9 Tham số)
                    # Gửi trực tiếp tọa độ để các xy-lanh của bệ thực thi
                    # --------------------------------------------------------
                    if self.config['dof_mode'] == 0:
                        # Trường hợp ngoại lệ: Chọn gửi 9 tham số nhưng lại bắt Bypass (Không qua Washout)
                        # -> Gửi độ lệch vị trí tương đối so với điểm xuất phát và góc thật của xe
                        start_to_current_x = loc.x - start_loc.x
                        start_to_current_y = loc.y - start_loc.y
                        start_to_current_z = loc.z - start_loc.z
                        payload = (
                            f"{start_to_current_x:.4f},{start_to_current_y:.4f},{start_to_current_z:.4f}," # 1,2,3. Dịch chuyển (m)
                            f"{math.radians(t.rotation.roll):.4f},{math.radians(t.rotation.pitch):.4f},{math.radians(t.rotation.yaw):.4f}," # 4,5,6. Góc (Rad)
                            f"{v.x:.4f},{v.y:.4f},{v.z:.4f}," # 7,8,9. Vận tốc hành trình (m/s)
                            f"{send_timestamp:.6f}"
                        )
                    else:
                        # Chuẩn: Gửi 9 thông số đầu ra từ thuật toán Washout Filter
                        payload = (
                            f"{wo_out[0]:.4f},{wo_out[1]:.4f},{wo_out[2]:.4f}," # 1,2,3. Dịch chuyển bù trừ LHP (m)
                            f"{wo_out[3]:.4f},{wo_out[4]:.4f},{wo_out[5]:.4f}," # 4,5,6. Góc nghiêng AHP & Tilt Coordination (Rad)
                            f"{wo_out[6]:.4f},{wo_out[7]:.4f},{wo_out[8]:.4f}," # 7,8,9. Vận tốc bệ (m/s)
                            f"{send_timestamp:.6f}"
                        )

                # Bắn gói tin dạng string ASCII qua port 1288
                self.sock.sendto(payload.encode('utf-8'), (UDP_IP, UDP_PORT))
                
        except KeyboardInterrupt:
            pass
        finally:
            logger.stop()
            client.stop_recorder() # Dừng ghi log để chuẩn bị cho Replay
            
            if drive_cam: drive_cam.destroy()
            vehicle.destroy()
            
            settings.no_rendering_mode = False 
            settings.synchronous_mode = False
            world.apply_settings(settings)
            
            if self.config['display_mode'] == "2D_Live":
                plt.ioff(); plt.close(fig_live)
            
            self.plot_trajectory_offline(hist_t, hist_x, hist_y, hist_z, hist_v, hist_ax, hist_ay, hist_az)

if __name__ == "__main__":
    app = CarlaNode()
