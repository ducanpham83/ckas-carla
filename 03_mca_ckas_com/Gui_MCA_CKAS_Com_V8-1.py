import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import signal
from scipy.fft import fft, fftfreq
import json, os, socket, threading, time, re, subprocess

try: import serial; SERIAL_AVAILABLE = True
except ImportError: SERIAL_AVAILABLE = False

try:
    import cv2
    from PIL import Image, ImageTk
    VIDEO_AVAILABLE = True
except ImportError:
    VIDEO_AVAILABLE = False

# =====================================================================
# 1. LOGIC TOÁN HỌC & WASHOUT CORE
# =====================================================================
class WashoutCore:
    @staticmethod
    def get_tf(fs, num, den):
        return signal.TransferFunction(num, den).to_discrete(dt=1.0/fs, method='bilinear')

    @classmethod
    def run(cls, fs, gains, p_mat, is_3rd, df_in, thr):
        dt, g = 1.0/fs, 9.81
        ax = df_in['Surge_ms2'].values * gains[0]
        ay = df_in['Sway_ms2'].values * gains[1]
        az = df_in['Heave_ms2'].values * gains[2]
        wx_r = df_in['RollRate_rads'].values * gains[3]
        wy_r = df_in['PitchRate_rads'].values * gains[4]
        wz_r = df_in['YawRate_rads'].values * gains[5]

        wx = np.where(np.abs(wx_r) < thr, 0, wx_r)
        wy = np.where(np.abs(wy_r) < thr, 0, wy_r)
        wz = np.where(np.abs(wz_r) < thr, 0, wz_r)
        
        pos, tilt, rot = [], [], []
        sigs = [ax, ay, az, wx, wy, wz]
        
        for i in range(6):
            w, z, wb, lw, lz = p_mat[i]
            if is_3rd:
                num = [1, 0] if i < 3 else [1, 0, 0]
                den = [1, wb + 2*z*w, 2*z*w*wb + w**2, wb * w**2]
            else:
                num = [1] if i < 3 else [1, 0]
                den = [1, 2*z*w, w**2]
            
            sys_hp = cls.get_tf(fs, num, den)
            if i < 3:
                pos.append(signal.lfilter(sys_hp.num, sys_hp.den, sigs[i]))
                if i < 2:
                    sys_lp = cls.get_tf(fs, [lw**2/g], [1, 2*lz*lw, lw**2])
                    tilt.append((1 if i==0 else -1) * signal.lfilter(sys_lp.num, sys_lp.den, sigs[i]))
            else:
                rot.append(signal.lfilter(sys_hp.num, sys_hp.den, sigs[i]))

        r_deg = np.degrees(tilt[1] + rot[0]) if len(tilt)>1 else np.zeros_like(ax)
        p_deg = np.degrees(tilt[0] + rot[1]) if len(tilt)>0 else np.zeros_like(ax)
        y_deg = np.degrees(rot[2])
        
        acc_x = np.gradient(np.gradient(pos[0], dt), dt)
        acc_y = np.gradient(np.gradient(pos[1], dt), dt)
        acc_z = np.gradient(np.gradient(pos[2], dt), dt)
        
        r_rad, p_rad = np.radians(r_deg), np.radians(p_deg)
        sf_x = acc_x + g * np.sin(p_rad)
        sf_y = acc_y - g * np.cos(p_rad) * np.sin(r_rad)
        sf_z = acc_z + g * np.cos(p_rad) * np.cos(r_rad)
        
        return pos, (r_deg, p_deg, y_deg), (sf_x, sf_y, sf_z), (acc_x, acc_y, acc_z)

# =====================================================================
# 2. KHỞI TẠO APP VÀ GIAO DIỆN
# =====================================================================
class IMUAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CKAS 6-DOF Auto-Tuning Washout Filter V8 (Final Edition)")
        self.root.geometry("1450x950")
        
        self.df_original, self.df_processed = None, None
        self.fs, self.plot_counter = 100.0, 0
        self.ch_names = ["Surge (X)", "Sway (Y)", "Heave (Z)", "Roll", "Pitch", "Yaw"]
        self.cols_process = ['Surge_ms2', 'Sway_ms2', 'Heave_ms2', 'RollRate_rads', 'PitchRate_rads', 'YawRate_rads']
        
        self.is_transmitting = False
        self.video_playing = False
        self.cap = None
        self.serial_port, self.udp_socket = None, None
        self.fullscreen_win = None
        
        # UI Variables
        self.ckas_format_var = tk.StringVar(value="6 Tham số")
        self.export_dt_var = tk.StringVar(value="0.0100")
        self.video_path_var = tk.StringVar()
        self.video_mode_var = tk.StringVar(value="Trong App (Test)")
        
        self.setup_ui()

    def setup_ui(self):
        style = ttk.Style(); style.theme_use('clam')
        self.notebook = ttk.Notebook(self.root); self.notebook.pack(fill=tk.BOTH, expand=True)
        
        self.tab_process = ttk.Frame(self.notebook); self.notebook.add(self.tab_process, text="1. Dữ liệu & Washout")
        self.tab_pso = ttk.Frame(self.notebook); self.notebook.add(self.tab_pso, text="2. ⚡ Auto-Tune (PSO)")
        self.tab_plots = ttk.Frame(self.notebook); self.notebook.add(self.tab_plots, text="3. 📊 Đồ thị & Dữ liệu")
        self.tab_comp = ttk.Frame(self.notebook); self.notebook.add(self.tab_comp, text="4. ⚖️ Benchmark")
        self.tab_connect = ttk.Frame(self.notebook); self.notebook.add(self.tab_connect, text="5. Kết nối CKAS")
        
        self.setup_tab_process()
        self.setup_tab_pso()
        self.setup_tab_plots()
        self.setup_tab_comp()
        self.setup_tab_connect()
        # ---> THÊM 2 DÒNG NÀY:
        self.tab_vr = ttk.Frame(self.notebook); self.notebook.add(self.tab_vr, text="6. 🥽 VR Experience")
        self.setup_tab_vr()

        self.lbl_status = ttk.Label(self.root, text="Hệ thống Sẵn sàng...", foreground="blue")
        self.lbl_status.pack(side=tk.BOTTOM, anchor=tk.W, padx=10, pady=2)

    # --- TAB 1: DỮ LIỆU & WASHOUT ---
    def setup_tab_process(self):
        parent = self.tab_process
        f_top = tk.Frame(parent); f_top.pack(fill=tk.X, padx=10, pady=5)
        
        # 1. Dữ liệu & Xử lý thô
        f_file = ttk.LabelFrame(f_top, text="1. Dữ liệu & Xử lý thô")
        f_file.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        f_f1 = tk.Frame(f_file); f_f1.pack(fill=tk.X, pady=2)
        self.entry_path = ttk.Entry(f_f1, width=25); self.entry_path.pack(side=tk.LEFT, padx=5)
        ttk.Button(f_f1, text="Mở CSV", command=self.load_csv).pack(side=tk.LEFT, padx=2)
        ttk.Button(f_f1, text="FFT", command=self.plot_fft, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Button(f_f1, text="Reset", command=self.reset_data, width=6).pack(side=tk.LEFT, padx=2)

        f_f2 = tk.Frame(f_file); f_f2.pack(fill=tk.X, pady=2)
        ttk.Label(f_f2, text="LP:").pack(side=tk.LEFT); self.e_lp = ttk.Entry(f_f2, width=4); self.e_lp.insert(0, "5.0"); self.e_lp.pack(side=tk.LEFT)
        ttk.Label(f_f2, text="HP:").pack(side=tk.LEFT); self.e_hp = ttk.Entry(f_f2, width=4); self.e_hp.insert(0, "0.5"); self.e_hp.pack(side=tk.LEFT)
        ttk.Button(f_f2, text="LP", command=lambda: self.apply_basic_filter('low'), width=4).pack(side=tk.LEFT, padx=1)
        ttk.Button(f_f2, text="HP", command=lambda: self.apply_basic_filter('high'), width=4).pack(side=tk.LEFT, padx=1)
        ttk.Button(f_f2, text="Band", command=lambda: self.apply_basic_filter('band'), width=5).pack(side=tk.LEFT, padx=1)
        
        # 2. Cắt & Đánh giá Step
        f_edit = ttk.LabelFrame(f_top, text="2. Cắt & Đánh giá Step")
        f_edit.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        f_e1 = tk.Frame(f_edit); f_e1.pack(fill=tk.X, pady=2)
        ttk.Label(f_e1, text="Từ:").pack(side=tk.LEFT); self.e_t1 = ttk.Entry(f_e1, width=4); self.e_t1.insert(0, "2.0"); self.e_t1.pack(side=tk.LEFT)
        ttk.Label(f_e1, text="Đến:").pack(side=tk.LEFT); self.e_t2 = ttk.Entry(f_e1, width=4); self.e_t2.insert(0, "5.0"); self.e_t2.pack(side=tk.LEFT)
        tk.Button(f_e1, text="✂️ Cắt", command=self.crop_data, width=6).pack(side=tk.LEFT, padx=2)
        
        f_e2 = tk.Frame(f_edit); f_e2.pack(fill=tk.X, pady=2)
        ttk.Label(f_e2, text="Kênh:").pack(side=tk.LEFT)
        self.step_ch_var = tk.StringVar(value="Surge (X)")
        ttk.Combobox(f_e2, textvariable=self.step_ch_var, values=self.ch_names, state="readonly", width=10).pack(side=tk.LEFT, padx=2)
        ttk.Label(f_e2, text="Amp:").pack(side=tk.LEFT); self.e_step = ttk.Entry(f_e2, width=4); self.e_step.insert(0, "1.0"); self.e_step.pack(side=tk.LEFT)
        tk.Button(f_e2, text="📈 Step", bg="#FFEB3B", command=self.eval_step_response, font=("Arial", 8, "bold")).pack(side=tk.LEFT, padx=2)

        # 3. Washout Filter Configuration
        f_wo = ttk.LabelFrame(parent, text="3. Washout Filter Configuration"); f_wo.pack(fill=tk.X, padx=10, pady=5)
        
        f_wo_t = tk.Frame(f_wo); f_wo_t.pack(fill=tk.X)
        self.wo_type_var = tk.StringVar(value="CKAS Ref (Bậc 2)")
        self.combo_wo_type = ttk.Combobox(f_wo_t, textvariable=self.wo_type_var, values=["CKAS Ref (Bậc 2)", "Custom (Bậc 3)"], state="readonly", width=25)
        self.combo_wo_type.pack(side=tk.LEFT, padx=5, pady=5)
        self.combo_wo_type.bind("<<ComboboxSelected>>", self.update_wo_formula)
        
        f_formula = tk.Frame(f_wo_t, bg="#F5F5F5", relief=tk.RIDGE, bd=1)
        f_formula.pack(side=tk.LEFT, padx=10)
        self.lbl_wo_formula = ttk.Label(f_formula, text="HP(s) = s² / (s² + 2ζwns + wn²)", font=("Courier", 8, "bold"), foreground="#D84315")
        self.lbl_wo_formula.pack(padx=10)
        tk.Button(f_wo_t, text="⚡ CHẠY AUTO-TUNE (PSO)", bg="#E91E63", fg="white", font=("Arial", 9, "bold"), command=self.run_pso_18d).pack(side=tk.RIGHT, padx=10)

        f_grid = tk.Frame(f_wo); f_grid.pack(pady=5)
        headers = ["Kênh", "Gain", "HP wn", "HP zeta", "HP wb", "LP wn", "LP zeta"]
        for c, h in enumerate(headers): ttk.Label(f_grid, text=h, font=("Arial", 8, "bold")).grid(row=0, column=c, padx=8)
        
        self.params_ui = {}
        for r, ch in enumerate(self.ch_names):
            ttk.Label(f_grid, text=ch).grid(row=r+1, column=0, sticky=tk.W)
            ents = []
            for c in range(6):
                e = ttk.Entry(f_grid, width=8); e.insert(0, "1.0"); e.grid(row=r+1, column=c+1, padx=4, pady=1)
                if ch in ["Heave (Z)", "Roll", "Pitch", "Yaw"] and c >= 4: e.config(state=tk.DISABLED)
                ents.append(e)
            self.params_ui[ch] = ents

        f_act = tk.Frame(parent); f_act.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(f_act, text="Thresh:").pack(side=tk.LEFT); self.e_thr = ttk.Entry(f_act, width=6); self.e_thr.insert(0, "0.01"); self.e_thr.pack(side=tk.LEFT, padx=5)
        tk.Button(f_act, text="▶ CHẠY WASHOUT & VẼ ĐỒ THỊ", bg="#FF9800", font=("Arial", 10, "bold"), command=self.apply_washout).pack(side=tk.LEFT, padx=20)
        tk.Button(f_act, text="🚀 XUẤT SANG STEWART 3D", bg="#2196F3", fg="white", font=("Arial", 9, "bold"), command=self.launch_stewart_dashboard).pack(side=tk.RIGHT, padx=5)
        tk.Button(f_act, text="💾 LƯU DYNAMIC SCRIPT", bg="#4CAF50", fg="white", font=("Arial", 9, "bold"), command=self.export_ckas_accel).pack(side=tk.RIGHT, padx=5)

        # 4. Trình xem Script & Quản lý File
        f_script = ttk.LabelFrame(parent, text="4. Trình xem Script & Quản lý File")
        f_script.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        f_sc_ctrl = tk.Frame(f_script); f_sc_ctrl.pack(fill=tk.X, pady=2)
        ttk.Label(f_sc_ctrl, text="dt(s):").pack(side=tk.LEFT, padx=2)
        ttk.Entry(f_sc_ctrl, textvariable=self.export_dt_var, state="readonly", width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(f_sc_ctrl, text="Format:").pack(side=tk.LEFT, padx=5)
        self.combo_ckas_format = ttk.Combobox(f_sc_ctrl, textvariable=self.ckas_format_var, values=["6 Tham số", "15 Tham số"], state="readonly", width=15)
        self.combo_ckas_format.pack(side=tk.LEFT)
        
        tk.Button(f_sc_ctrl, text="TẠO MÃ XEM TRƯỚC", bg="#2196F3", fg="white", command=self.generate_script_to_viewer).pack(side=tk.LEFT, padx=5)
        tk.Button(f_sc_ctrl, text="XUẤT SCRIPT", bg="#4CAF50", fg="white", command=self.export_ckas_script).pack(side=tk.LEFT, padx=5)
        tk.Button(f_sc_ctrl, text="LƯU CẤU HÌNH", bg="#607D8B", fg="white", command=self.save_config_txt).pack(side=tk.RIGHT, padx=5)
        tk.Button(f_sc_ctrl, text="TẢI CẤU HÌNH", command=self.load_config_txt).pack(side=tk.RIGHT, padx=2)
        
        self.text_script = tk.Text(f_script, wrap=tk.NONE, font=("Consolas", 9), bg="#1E1E1E", fg="#00FF00", height=8)
        sy = ttk.Scrollbar(f_script, orient=tk.VERTICAL, command=self.text_script.yview)
        sx = ttk.Scrollbar(f_script, orient=tk.HORIZONTAL, command=self.text_script.xview)
        self.text_script.config(yscrollcommand=sy.set, xscrollcommand=sx.set)
        sx.pack(side=tk.BOTTOM, fill=tk.X); sy.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_script.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        # --- THÊM NÚT GỌI CÔNG CỤ CONVERT (Giữ nguyên các chức năng cũ ở trên) ---
        f_conv = tk.Frame(f_script)
        f_conv.pack(fill=tk.X, pady=5)
        tk.Button(f_conv, text="🛠 MỞ CÔNG CỤ CONVERT / NỘI SUY TẦN SỐ (Độc lập)", bg="#9C27B0", fg="white", font=("Arial", 9, "bold"), command=self.open_converter_tool).pack(side=tk.LEFT, padx=5)

    # --- TAB 2: PSO AUTO-TUNE ---
    # --- TAB 2: PSO AUTO-TUNE (ĐÃ THÊM LƯU/TẢI CẤU HÌNH) ---
    def setup_tab_pso(self):
        parent = self.tab_pso
        f_left = tk.Frame(parent); f_left.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        f_ws = ttk.LabelFrame(f_left, text="Giới hạn Không gian làm việc (Limits)"); f_ws.pack(fill=tk.X, pady=5)
        self.ws_entries = []; ws_defaults = ["0.15", "0.15", "0.15", "23", "22.0", "25.0"]
        for i, text in enumerate(["Max X(m):", "1ax Y(m):", "Max Z(m):", "Max Roll(°):", "Max Pitch(°):", "Max Yaw(°):"]):
            row, col = i // 2, (i % 2) * 2
            ttk.Label(f_ws, text=text).grid(row=row, column=col, padx=2, pady=2)
            e = ttk.Entry(f_ws, width=6); e.insert(0, ws_defaults[i]); e.grid(row=row, column=col+1, padx=2, pady=2)
            self.ws_entries.append(e)
            
        f_weights = ttk.LabelFrame(f_left, text="Trọng số Hàm Cost (Weights)"); f_weights.pack(fill=tk.BOTH, expand=True, pady=5)
        self.weight_entries = {}
        default_weights = [
            ("Đ.Dạng Lực", "w_p_fx", 2.0), ("Đ.Dạng Lực", "w_p_fy", 2.0), ("Đ.Dạng Lực", "w_p_fz", 1.0),
            ("S.Số Lực", "w_e_fx", 1.0), ("S.Số Lực", "w_e_fy", 1.0), ("S.Số Lực", "w_e_fz", 1.0),
            ("S.Số Góc", "w_wx", 1.0), ("S.Số Góc", "w_wy", 1.0), ("S.Số Góc", "w_wz", 1.0),
            ("Phạt Vị trí", "w_px", 0.1), ("Phạt Vị trí", "w_py", 0.1), ("Phạt Vị trí", "w_pz", 0.1),
            ("Phạt Góc", "w_r", 0.1), ("Phạt Góc", "w_p", 0.1), ("Phạt Góc", "w_yaw", 0.1),
            ("Phạt Gain", "w_gain", 0.5)
        ]
        for r, (group, var_name, default_val) in enumerate(default_weights):
            ttk.Label(f_weights, text=var_name, foreground="blue").grid(row=r//2, column=(r%2)*2, sticky=tk.W, padx=5)
            e = ttk.Entry(f_weights, width=5); e.insert(0, str(default_val)); e.grid(row=r//2, column=(r%2)*2+1, padx=5)
            self.weight_entries[var_name] = e

        f_swarm = ttk.LabelFrame(f_left, text="Tham số Bầy đàn"); f_swarm.pack(fill=tk.X, pady=5)
        ttk.Label(f_swarm, text="Hạt:").grid(row=0, column=0); self.e_p = ttk.Entry(f_swarm, width=5); self.e_p.insert(0, "20"); self.e_p.grid(row=0, column=1)
        ttk.Label(f_swarm, text="Vòng:").grid(row=0, column=2); self.e_i = ttk.Entry(f_swarm, width=5); self.e_i.insert(0, "30"); self.e_i.grid(row=0, column=3)
        ttk.Label(f_swarm, text="w:").grid(row=1, column=0); self.e_w = ttk.Entry(f_swarm, width=5); self.e_w.insert(0, "0.5"); self.e_w.grid(row=1, column=1)
        ttk.Label(f_swarm, text="c1:").grid(row=1, column=2); self.e_c1 = ttk.Entry(f_swarm, width=5); self.e_c1.insert(0, "1.5"); self.e_c1.grid(row=1, column=3)
        ttk.Label(f_swarm, text="c2:").grid(row=2, column=0); self.e_c2 = ttk.Entry(f_swarm, width=5); self.e_c2.insert(0, "1.5"); self.e_c2.grid(row=2, column=1)
        
        # ---> MỚI: Nút chức năng Lưu/Tải thông số PSO
        f_pso_ctrl = tk.Frame(f_left); f_pso_ctrl.pack(fill=tk.X, pady=10)
        tk.Button(f_pso_ctrl, text="💾 LƯU CẤU HÌNH PSO", bg="#607D8B", fg="white", font=("Arial", 8, "bold"), command=self.save_pso_config).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        tk.Button(f_pso_ctrl, text="📂 TẢI CẤU HÌNH PSO", bg="#795548", fg="white", font=("Arial", 8, "bold"), command=self.load_pso_config).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        f_right = ttk.LabelFrame(parent, text="Lập trình Hàm Mục Tiêu (Tương thích 18 chiều)"); f_right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.text_code = tk.Text(f_right, wrap=tk.NONE, font=("Consolas", 10), bg="#1E1E1E", fg="#FFFFFF")
        self.text_code.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        pso_code = """cost = 0.0
# 1. PHẠT CỨNG NẾU VƯỢT GIỚI HẠN CƠ KHÍ
if max_x > lim_x or max_y > lim_y or max_z > lim_z or max_r > lim_r or max_p > lim_p or max_yaw > lim_yaw: cost = 999999.0
else:
    cost += w_p_fx * (1.0 - pearson_x) + w_p_fy * (1.0 - pearson_y) + w_p_fz * (1.0 - pearson_z)
    cost += w_e_fx * err_fx + w_e_fy * err_fy + w_e_fz * err_fz
    cost += w_wx * err_wx + w_wy * err_wy + w_wz * err_wz
    cost += w_px * max_x + w_py * max_y + w_pz * max_z + w_r * max_r + w_p * max_p + w_yaw * max_yaw
    cost += w_gain * ((1.0 - gain_x)**2 + (1.0 - gain_y)**2 + (1.0 - gain_z)**2 + 
                      (1.0 - gain_r)**2 + (1.0 - gain_p)**2 + (1.0 - gain_yaw)**2)
"""
        self.text_code.insert(tk.END, pso_code)
        
        self.btn_pso = tk.Button(f_right, text="🚀 BẮT ĐẦU AUTO-TUNE PSO (18 CHIỀU)", bg="#E91E63", fg="white", font=("Arial", 10, "bold"), command=self.run_pso_18d)
        self.btn_pso.pack(fill=tk.X, pady=5)
        self.pso_progress = ttk.Progressbar(f_right, maximum=100); self.pso_progress.pack(fill=tk.X)
    # --- TAB 3: ĐỒ THỊ CHUYÊN SÂU & DATA TABLE ---
    # --- TAB 3: ĐỒ THỊ CHUYÊN SÂU & DATA TABLE ---
    def setup_tab_plots(self):
        parent = self.tab_plots
        f_ctrl = tk.Frame(parent); f_ctrl.pack(fill=tk.X, padx=10, pady=5)
        
        # 1. Các Checkbox hiển thị mặc định
        self.plot_vars = {
            "Surge": tk.BooleanVar(value=True), "Sway": tk.BooleanVar(value=True), "Heave": tk.BooleanVar(value=True),
            "Roll": tk.BooleanVar(value=True), "Pitch": tk.BooleanVar(value=True), "Yaw": tk.BooleanVar(value=True)
        }
        f_chk = ttk.LabelFrame(f_ctrl, text="Bật/Tắt đại lượng hiển thị (Mặc định)"); f_chk.pack(side=tk.LEFT, padx=5)
        for i, (ch, var) in enumerate(self.plot_vars.items()):
            tk.Checkbutton(f_chk, text=ch, variable=var, font=("Arial", 9, "bold"), fg="#1565C0").grid(row=0, column=i, padx=5, pady=2)
            
        tk.Button(f_ctrl, text="🔄 VẼ LẠI ĐỒ THỊ", bg="#4CAF50", fg="white", font=("Arial", 9, "bold"), command=self.update_plots).pack(side=tk.LEFT, padx=15, pady=5, ipady=3)
        
        # 2. Tùy chọn Xuất File và Vẽ Tín hiệu khác (Listbox chọn nhiều)
        f_custom = ttk.LabelFrame(f_ctrl, text="Lựa chọn Dữ liệu Xuất CSV & Vẽ (Tín hiệu khác)")
        f_custom.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        
        # Danh sách chọn (Listbox hỗ trợ bôi đen nhiều dòng)
        self.list_custom = tk.Listbox(f_custom, selectmode=tk.MULTIPLE, height=3, exportselection=False)
        scroll = ttk.Scrollbar(f_custom, orient=tk.VERTICAL, command=self.list_custom.yview)
        self.list_custom.config(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.list_custom.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        tk.Button(f_custom, text="💾 XUẤT CSV (Đã chọn)", bg="#607D8B", fg="white", font=("Arial", 8, "bold"), command=self.export_custom_csv).pack(side=tk.RIGHT, padx=5)

        # 3. Tạo các Notebook Tabs
        self.nb_plots = ttk.Notebook(parent); self.nb_plots.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.f_plot_imu = ttk.Frame(self.nb_plots); self.nb_plots.add(self.f_plot_imu, text="1. Tín hiệu IMU")
        self.f_plot_pos = ttk.Frame(self.nb_plots); self.nb_plots.add(self.f_plot_pos, text="2. Vị trí & Góc bệ")
        self.f_plot_sf = ttk.Frame(self.nb_plots); self.nb_plots.add(self.f_plot_sf, text="3. Lực cảm nhận (SF) & Sai số")
        self.f_plot_data = ttk.Frame(self.nb_plots); self.nb_plots.add(self.f_plot_data, text="4. Bảng dữ liệu chi tiết")
        self.f_plot_custom = ttk.Frame(self.nb_plots); self.nb_plots.add(self.f_plot_custom, text="5. Tín Hiệu khác")
        
        # Khởi tạo bảng dữ liệu
        self.tree_data = ttk.Treeview(self.f_plot_data, show='headings'); self.tree_data.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb = ttk.Scrollbar(self.f_plot_data, orient="vertical", command=self.tree_data.yview); vsb.pack(side=tk.RIGHT, fill='y')
        self.tree_data.configure(yscrollcommand=vsb.set)
        
        self.canvas_imu = None; self.canvas_pos = None; self.canvas_sf = None; self.canvas_custom = None

    # --- TAB 4: BENCHMARK (ĐÃ NÂNG CẤP) ---
    def setup_tab_comp(self):
        parent = self.tab_comp
        f_ctrl = tk.Frame(parent); f_ctrl.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Button(f_ctrl, text="📁 Tải / Khởi tạo Cấu hình CKAS Mẫu", command=self.load_ref_config).pack(side=tk.LEFT, padx=10)
        
        # Bổ sung lựa chọn vẽ đồ thị theo từng kênh
        ttk.Label(f_ctrl, text="Chọn Kênh:").pack(side=tk.LEFT, padx=5)
        self.comp_ch_var = tk.StringVar(value="Surge (X)")
        ttk.Combobox(f_ctrl, textvariable=self.comp_ch_var, values=self.ch_names, state="readonly", width=12).pack(side=tk.LEFT, padx=5)
        
        tk.Button(f_ctrl, text="⚖️ CHẠY SO SÁNH 3 TÍN HIỆU", bg="#673AB7", fg="white", font=("Arial", 9, "bold"), command=self.run_comparison).pack(side=tk.LEFT, padx=10)
        self.lbl_ref_stt = ttk.Label(f_ctrl, text="Chưa nạp tham chiếu.", foreground="red"); self.lbl_ref_stt.pack(side=tk.LEFT, padx=10)
        
        # Vùng hiển thị trực quan thông số tham chiếu đang dùng
        self.lbl_ref_params = ttk.Label(parent, text="[Chưa có thông số CKAS mẫu]", foreground="blue", justify=tk.LEFT)
        self.lbl_ref_params.pack(anchor=tk.W, padx=20)
        
        self.ref_params, self.canvas_comp = None, None
        self.f_comp_plot = tk.Frame(parent); self.f_comp_plot.pack(fill=tk.BOTH, expand=True)
    # --- TAB 5: KẾT NỐI & VIDEO SYNC ---
    def setup_tab_connect(self):
        parent = self.tab_connect
        paned = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        left_col, right_col = ttk.Frame(paned), ttk.Frame(paned)
        paned.add(left_col, weight=1); paned.add(right_col, weight=2)
        
        f_conn = ttk.LabelFrame(left_col, text="Cấu hình Kết nối"); f_conn.pack(fill=tk.X, pady=10)
        self.conn_mode = tk.StringVar(value="UDP")
        ttk.Radiobutton(f_conn, text="Serial", variable=self.conn_mode, value="Serial").grid(row=0, column=0, sticky=tk.W)
        self.e_com = ttk.Entry(f_conn, width=10); self.e_com.insert(0, "COM3"); self.e_com.grid(row=0, column=1)
        ttk.Radiobutton(f_conn, text="UDP", variable=self.conn_mode, value="UDP").grid(row=1, column=0, sticky=tk.W)
        self.e_ip = ttk.Entry(f_conn, width=15); self.e_ip.insert(0, "192.168.1.100"); self.e_ip.grid(row=1, column=1)
        tk.Button(f_conn, text="KẾT NỐI", command=self.connect_hardware).grid(row=2, column=0, pady=10)
        tk.Button(f_conn, text="NGẮT", command=self.disconnect_hardware).grid(row=2, column=1, pady=10)
        
        f_tx = ttk.LabelFrame(left_col, text="Truyền Dữ liệu (Real-time)"); f_tx.pack(fill=tk.BOTH, expand=True, pady=10)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(f_tx, variable=self.progress_var, maximum=100); self.progress_bar.pack(fill=tk.X, padx=10, pady=10)
        self.lbl_tx_prog = ttk.Label(f_tx, text="0 / 0 lệnh (0%)"); self.lbl_tx_prog.pack()
        f_bx = tk.Frame(f_tx); f_bx.pack(pady=10)
        self.btn_tx = tk.Button(f_bx, text="▶ START STREAMING", bg="#2196F3", fg="white", font=("Arial", 10, "bold"), command=self.start_transmission)
        self.btn_tx.pack(side=tk.LEFT, padx=5)
        self.btn_stop = tk.Button(f_bx, text="⏹ STOP", command=self.stop_transmission, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=5)

        f_vid = ttk.LabelFrame(right_col, text="Màn hình Video (Đồng bộ thời gian thực)")
        f_vid.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        f_v_ctrl = tk.Frame(f_vid); f_v_ctrl.pack(fill=tk.X, padx=10, pady=5)
        ttk.Button(f_v_ctrl, text="Chọn Video...", command=self.browse_video).pack(side=tk.LEFT, padx=5)
        ttk.Entry(f_v_ctrl, textvariable=self.video_path_var, state="readonly", width=25).pack(side=tk.LEFT, padx=5)
        ttk.Combobox(f_v_ctrl, textvariable=self.video_mode_var, values=["Trong App (Test)", "Fullscreen (Màn 1)", "Fullscreen (Màn 2)"], state="readonly", width=18).pack(side=tk.LEFT, padx=5)
        
        self.lbl_video = tk.Label(f_vid, text="[MÀN HÌNH VIDEO]\nTự động phát khi bấm START STREAMING", bg="black", fg="white", font=("Arial", 12))
        self.lbl_video.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
# --- TAB 6: VR EXPERIENCE MODULE ---
    def setup_tab_vr(self):
        parent = self.tab_vr
        
        # Tiêu đề & Cảnh báo
        lbl_title = tk.Label(parent, text="TRUNG TÂM ĐIỀU KHIỂN THỰC TẾ ẢO (VR EXPERIENCE HUB)", font=("Arial", 14, "bold"), fg="#1565C0")
        lbl_title.pack(pady=(15, 5))
        lbl_warn = tk.Label(parent, text="Lưu ý: Để tránh hiện tượng Simulator Sickness (Say mô phỏng), độ trễ giữa hình ảnh VR và chuyển động bệ phải < 20ms.", font=("Arial", 10, "italic"), fg="#E65100")
        lbl_warn.pack(pady=(0, 15))

        paned = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=20, pady=5)
        
        f_left = ttk.Frame(paned); paned.add(f_left, weight=1)
        f_right = ttk.Frame(paned); paned.add(f_right, weight=1)

        # PHƯƠNG ÁN 1: DIRECT CARLA VR
        f_carla = ttk.LabelFrame(f_left, text="Phương án 1: Real-time CARLA VR (Cần PC cấu hình mạnh)")
        f_carla.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        ttk.Label(f_carla, text="Đường dẫn file CarlaUE4.exe:", font=("Arial", 9, "bold")).pack(anchor=tk.W, padx=10, pady=(10,2))
        f_c1 = tk.Frame(f_carla); f_c1.pack(fill=tk.X, padx=10)
        self.carla_path_var = tk.StringVar(value=r"C:\CARLA\CarlaUE4.exe")
        ttk.Entry(f_c1, textvariable=self.carla_path_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(f_c1, text="Browse...", command=lambda: self.carla_path_var.set(filedialog.askopenfilename(filetypes=[("Executable", "*.exe")]))).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(f_carla, text="Tham số khởi chạy (Launch Arguments):").pack(anchor=tk.W, padx=10, pady=(10,2))
        self.carla_args_var = tk.StringVar(value="-vr -quality-level=Low")
        ttk.Entry(f_carla, textvariable=self.carla_args_var).pack(fill=tk.X, padx=10)
        
        txt_desc1 = "Hệ thống sẽ gọi SteamVR hoặc Oculus Runtime.\nKính VR sẽ theo dõi chuyển động đầu thực tế của người lái."
        tk.Label(f_carla, text=txt_desc1, justify=tk.LEFT, fg="gray").pack(anchor=tk.W, padx=10, pady=10)
        
        tk.Button(f_carla, text="🚀 KHỞI CHẠY CARLA TRONG CHẾ ĐỘ VR", bg="#673AB7", fg="white", font=("Arial", 10, "bold"), command=self.launch_carla_vr).pack(pady=20, ipady=5)

        # PHƯƠNG ÁN 2: VR VIDEO SYNC
        f_vid = ttk.LabelFrame(f_right, text="Phương án 2: Sync Pre-rendered Video (Nhẹ máy, tối ưu trễ)")
        f_vid.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        ttk.Label(f_vid, text="File Video từ CARLA (MP4/MKV):", font=("Arial", 9, "bold")).pack(anchor=tk.W, padx=10, pady=(10,2))
        f_v1 = tk.Frame(f_vid); f_v1.pack(fill=tk.X, padx=10)
        self.vr_vid_path = tk.StringVar()
        ttk.Entry(f_v1, textvariable=self.vr_vid_path, state="readonly").pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(f_v1, text="Browse...", command=lambda: self.vr_vid_path.set(filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.mkv")]))).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(f_vid, text="Định dạng kính VR:").pack(anchor=tk.W, padx=10, pady=(10,2))
        self.vr_vid_format = tk.StringVar(value="Stereoscopic (Side-by-Side)")
        ttk.Combobox(f_vid, textvariable=self.vr_vid_format, values=["Stereoscopic (Side-by-Side)", "Monoscopic (180/360)"], state="readonly").pack(fill=tk.X, padx=10)
        
        txt_desc2 = "Sử dụng trình phát phụ (VD: DeoVR, VLC 360).\nVideo sẽ tự động kích hoạt cùng lúc khi bấm [START STREAMING] ở Tab 5."
        tk.Label(f_vid, text=txt_desc2, justify=tk.LEFT, fg="gray").pack(anchor=tk.W, padx=10, pady=10)
        
        tk.Button(f_vid, text="🔗 LIÊN KẾT VIDEO VÀO TAB STREAMING", bg="#009688", fg="white", font=("Arial", 10, "bold"), command=self.link_vr_video).pack(pady=20, ipady=5)

    # --- LOGIC CHO TAB VR ---
    def launch_carla_vr(self):
        carla_exe = self.carla_path_var.get()
        args = self.carla_args_var.get().split()
        if not os.path.exists(carla_exe):
            return messagebox.showerror("Lỗi", "Không tìm thấy file CarlaUE4.exe theo đường dẫn!")
        try:
            # Gọi CARLA dưới dạng tiến trình độc lập (không làm treo App V8)
            subprocess.Popen([carla_exe] + args)
            messagebox.showinfo("Khởi chạy", "Đã gửi lệnh gọi CARLA VR. Vui lòng đeo kính VR để kiểm tra SteamVR/Oculus.")
        except Exception as e:
            messagebox.showerror("Lỗi chạy CARLA", str(e))

    # --- CẬP NHẬT LOGIC VR 3D ---
    def link_vr_video(self):
        vid_path = self.vr_vid_path.get()
        if not vid_path: return messagebox.showwarning("Lỗi", "Hãy chọn video!")
        
        # Thiết lập cấu hình hiển thị 3D
        is_3d = "Stereoscopic" in self.vr_vid_format.get()
        
        # Gửi cấu hình sang Tab 5 (Streaming)
        self.video_path_var.set(vid_path)
        self.video_mode_var.set("Fullscreen (Màn 2)")
        
        msg = "Đã liên kết Video 3D (Stereo SBS)." if is_3d else "Đã liên kết Video 2D."
        messagebox.showinfo("Cấu hình VR", f"{msg}\nLưu ý: Hãy đặt trình phát VR ở chế độ Side-by-Side để có độ sâu 3D.")
    # =====================================================================
    # 3. LOGIC XỬ LÝ DỮ LIỆU
    # =====================================================================
    def update_wo_formula(self, event=None):
        if "Bậc 3" in self.wo_type_var.get():
            txt = "HP(s) = s³ / [ (s + wb)(s² + 2ζ·wn·s + wn²) ]\nLP(s) = (wn² / g) / (s² + 2ζ·wn·s + wn²)"
        else: txt = "HP(s) = s² / (s² + 2ζ·wn·s + wn²)            \nLP(s) = (wn² / g) / (s² + 2ζ·wn·s + wn²)"
        if hasattr(self, 'lbl_wo_formula'): self.lbl_wo_formula.config(text=txt)

    def load_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if path:
            try:
                self.df_original = pd.read_csv(path); self.df_processed = self.df_original.copy()
                self.entry_path.delete(0, tk.END); self.entry_path.insert(0, path)
                self.fs = 1.0 / np.mean(np.diff(self.df_processed['Time_s']))
                self.export_dt_var.set(f"{1.0/self.fs:.4f}")
                self.lbl_status.config(text=f"Đã nạp file. Fs = {self.fs:.2f} Hz"); self.update_data_table()
            except Exception as e: messagebox.showerror("Lỗi", str(e))

    def reset_data(self):
        if self.df_original is not None:
            self.df_processed = self.df_original.copy(); self.update_data_table()
            self.lbl_status.config(text="Đã Reset dữ liệu.")

    def crop_data(self):
        if self.df_processed is None: return
        try:
            t1, t2 = float(self.e_t1.get()), float(self.e_t2.get())
            mask = (self.df_processed['Time_s'] >= t1) & (self.df_processed['Time_s'] <= t2)
            self.df_processed = self.df_processed[mask].reset_index(drop=True)
            self.df_processed['Time_s'] -= self.df_processed['Time_s'].iloc[0]
            self.update_data_table()
            self.lbl_status.config(text=f"Đã giữ lại dữ liệu từ {t1}s đến {t2}s.")
        except Exception as e: messagebox.showerror("Lỗi", str(e))

    def apply_basic_filter(self, btype):
        if self.df_processed is None: return
        try:
            cutoff = float(self.e_lp.get()) if btype == 'low' else float(self.e_hp.get())
            b, a = signal.butter(4, cutoff / (0.5 * self.fs), btype=btype, analog=False)
            if btype == 'band': b, a = signal.butter(4, [float(self.e_hp.get()) / (0.5 * self.fs), float(self.e_lp.get()) / (0.5 * self.fs)], btype='band', analog=False)
            for col in self.cols_process: self.df_processed[col] = signal.filtfilt(b, a, self.df_processed[col])
            self.update_data_table(); self.lbl_status.config(text=f"Đã áp dụng Filter.")
        except Exception as e: messagebox.showerror("Lỗi", str(e))

    def plot_fft(self):
        if self.df_processed is None: return
        self.plot_counter += 1; N, T = len(self.df_processed), 1.0 / self.fs; xf = fftfreq(N, T)[:N//2]
        fig, axs = plt.subplots(2, 3, figsize=(12, 6)); fig.canvas.manager.set_window_title("FFT Analysis")
        fig.suptitle(f'[Lần {self.plot_counter}] Phân tích Phổ tần số (0-20Hz)', fontweight='bold')
        for i, col in enumerate(self.cols_process):
            axs[i//3, i%3].plot(xf, 2.0/N * np.abs(fft(self.df_processed[col].values)[0:N//2]), 'b')
            axs[i//3, i%3].set_xlim(0, 20); axs[i//3, i%3].grid(True); axs[i//3, i%3].set_title(col)
        fig.tight_layout(); plt.show(block=False)

    def update_data_table(self):
        """Cập nhật bảng dữ liệu và danh sách chọn tín hiệu tùy chỉnh"""
        if self.df_processed is None or not hasattr(self, 'tree_data'): return
        
        # Xóa dữ liệu cũ trong Treeview
        for i in self.tree_data.get_children(): self.tree_data.delete(i)
        
        # Hiển thị Treeview
        cols = ['Time_s'] + self.cols_process + ['Platform_Surge_m', 'Platform_Sway_m', 'Platform_Heave_m', 'Platform_Roll_deg', 'Platform_Pitch_deg', 'Platform_Yaw_deg', 'SF_Plat_Surge', 'SF_Plat_Sway', 'SF_Plat_Heave']
        exist = [c for c in cols if c in self.df_processed.columns]
        self.tree_data["columns"] = exist
        for col in exist:
            self.tree_data.heading(col, text=col.replace('Platform_', 'P_').replace('_deg', '(°)').replace('_ms2', '').replace('_m', '(m)'))
            self.tree_data.column(col, width=85, anchor=tk.CENTER)
        for _, row in self.df_processed.head(500).iterrows():
            self.tree_data.insert("", "end", values=[f"{row[c]:.4f}" if isinstance(row[c], float) else row[c] for c in exist])
            
        # ---> CẬP NHẬT LISTBOX TÍN HIỆU KHÁC (TỰ ĐỘNG)
        if hasattr(self, 'list_custom'):
            # Giữ lại các mục đang chọn để không bị mất khi load lại
            selected_names = [self.list_custom.get(i) for i in self.list_custom.curselection()]
            self.list_custom.delete(0, tk.END)
            
            # Đưa toàn bộ các cột (trừ Time_s) vào danh sách
            cols_to_insert = [c for c in self.df_processed.columns if c != 'Time_s']
            for col in cols_to_insert:
                self.list_custom.insert(tk.END, col)
                
            # Phục hồi trạng thái chọn
            for name in selected_names:
                if name in cols_to_insert:
                    self.list_custom.selection_set(cols_to_insert.index(name))

    def export_custom_csv(self):
        """Xuất file CSV chứa cột Time_s và các cột dữ liệu được bôi đen trong Listbox"""
        if self.df_processed is None: return messagebox.showwarning("Lỗi", "Chưa có dữ liệu!")
        
        selected_indices = self.list_custom.curselection()
        if not selected_indices:
            return messagebox.showwarning("Chưa chọn", "Vui lòng bôi đen (chọn) ít nhất 1 dữ liệu trong danh sách!")
            
        selected_cols = [self.list_custom.get(i) for i in selected_indices]
        valid_cols = [c for c in selected_cols if c in self.df_processed.columns]
        
        export_cols = ['Time_s'] + valid_cols
        
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")], initialfile="Custom_Data_Export.csv")
        if path:
            try:
                self.df_processed[export_cols].to_csv(path, index=False)
                messagebox.showinfo("Thành công", f"Đã xuất file CSV với {len(valid_cols)} cột dữ liệu (kèm Time_s).")
            except Exception as e:
                messagebox.showerror("Lỗi Xuất File", str(e))
    def apply_washout(self, benchmark=False):
        if self.df_processed is None: return messagebox.showwarning("Lỗi", "Chưa có dữ liệu!")
        try:
            gains = [float(self.params_ui[ch][0].get()) for ch in self.ch_names]
            p_mat = [[float(e.get()) for e in self.params_ui[ch][1:]] for ch in self.ch_names]
            is_3rd = "Bậc 3" in self.wo_type_var.get()
            thr = float(self.e_thr.get())
            pos, ang, sf, acc = WashoutCore.run(self.fs, gains, p_mat, is_3rd, self.df_processed, thr)
            
            if not benchmark:
                dt, g = 1.0 / self.fs, 9.81
                self.df_processed['Platform_Surge_m'], self.df_processed['Platform_Sway_m'], self.df_processed['Platform_Heave_m'] = pos
                self.df_processed['Platform_Roll_deg'], self.df_processed['Platform_Pitch_deg'], self.df_processed['Platform_Yaw_deg'] = ang
                self.df_processed['SF_Plat_Surge'], self.df_processed['SF_Plat_Sway'], self.df_processed['SF_Plat_Heave'] = sf
                self.df_processed['SF_Raw_Surge'] = self.df_processed['Surge_ms2'].values * gains[0]
                self.df_processed['SF_Raw_Sway'] = self.df_processed['Sway_ms2'].values * gains[1]
                self.df_processed['SF_Raw_Heave'] = (self.df_processed['Heave_ms2'].values * gains[2]) + g
                self.df_processed['Target_Surge_Accel_ms2'] = self.df_processed['Surge_ms2'].values * gains[0]
                self.df_processed['Target_Sway_Accel_ms2'] = self.df_processed['Sway_ms2'].values * gains[1]
                self.df_processed['Target_Heave_Accel_ms2'] = self.df_processed['Heave_ms2'].values * gains[2]
                self.df_processed['Target_RollRate_rads'] = self.df_processed['RollRate_rads'].values * gains[3]
                self.df_processed['Target_PitchRate_rads'] = self.df_processed['PitchRate_rads'].values * gains[4]
                self.df_processed['Target_YawRate_rads'] = self.df_processed['YawRate_rads'].values * gains[5]
                self.df_processed['Platform_Surge_Accel_ms2'], self.df_processed['Platform_Sway_Accel_ms2'], self.df_processed['Platform_Heave_Accel_ms2'] = acc
                self.df_processed['Platform_RollRate_rads'] = np.gradient(np.radians(ang[0]), dt)
                self.df_processed['Platform_PitchRate_rads'] = np.gradient(np.radians(ang[1]), dt)
                self.df_processed['Platform_YawRate_rads'] = np.gradient(np.radians(ang[2]), dt)
                
                self.lbl_status.config(text="Đã chạy Washout Filter thành công.")
                self.update_plots()
            return pos, ang, sf
        except Exception as e: messagebox.showerror("Lỗi Washout", str(e))

    def eval_step_response(self):
        try:
            amp = float(self.e_step.get()); ch_sel = self.step_ch_var.get(); t = np.arange(0, 8.0, 1.0/self.fs)
            df_step = pd.DataFrame({'Time_s': t})
            for c in self.cols_process: df_step[c] = 0.0
            idx = self.ch_names.index(ch_sel); col_in = self.cols_process[idx]
            df_step[col_in] = np.where(t >= 1.0, amp, 0)
            gains = [float(self.params_ui[ch][0].get()) for ch in self.ch_names]
            p_mat = [[float(e.get()) for e in self.params_ui[ch][1:]] for ch in self.ch_names]
            is_3rd = "Bậc 3" in self.wo_type_var.get()
            pos, ang, sf, _ = WashoutCore.run(self.fs, gains, p_mat, is_3rd, df_step, 0.01)
            
            fig, ax = plt.subplots(3, 1, figsize=(8, 9)); fig.canvas.manager.set_window_title(f"Step Response - {ch_sel}"); fig.suptitle(f"Đáp ứng Step kênh {ch_sel}", fontweight='bold')
            ax[0].plot(t, df_step[col_in], 'k--', label="Input Step")
            if idx < 3: ax[0].plot(t, sf[idx], 'r', lw=2, label='Specific Force (Lực cảm nhận)')
            else: ax[0].plot(t, np.gradient(np.radians(ang[idx-3]), 1.0/self.fs), 'r', lw=2, label='Angular Rate (rad/s)')
            ax[0].set_title("Đầu vào (Input) và Đầu ra chính"); ax[0].grid(True); ax[0].legend()
            
            if idx < 3: ax[1].plot(t, pos[idx], 'b', lw=2, label=f'Chuyển vị {ch_sel} (m)')
            else: ax[1].plot(t, ang[idx-3], 'b', lw=2, label=f'Góc xoay {ch_sel} (deg)')
            ax[1].set_title("Kênh High-Pass (Rửa trôi về 0)"); ax[1].grid(True); ax[1].legend()
            
            if idx == 0: ax[2].plot(t, ang[1], 'g', lw=2, label='Góc Pitch (deg)')
            elif idx == 1: ax[2].plot(t, ang[0], 'g', lw=2, label='Góc Roll (deg)')
            else: ax[2].text(0.5, 0.5, "Không sử dụng Tilt", ha='center', va='center'); ax[2].axis('off')
            if idx < 2: ax[2].set_title("Kênh Low-Pass Tilt"); ax[2].grid(True); ax[2].legend()
            fig.tight_layout(); plt.show(block=False)
        except Exception as e: messagebox.showerror("Lỗi Step Response", str(e))

    def update_plots(self):
        if self.df_processed is None or 'Platform_Surge_m' not in self.df_processed.columns: return
        self.notebook.select(self.tab_plots); t = self.df_processed['Time_s'].values; keys_ch = ["Surge", "Sway", "Heave", "Roll", "Pitch", "Yaw"]
        # =================================================================
        # ---> CHÈN THÊM KHỐI NÀY VÀO ĐÂY: 0. TÍN HIỆU IMU ĐẦU VÀO
        if self.canvas_imu: self.canvas_imu.get_tk_widget().destroy()
        fig_imu, axs_imu = plt.subplots(2, 3, figsize=(12, 6))
        fig_imu.suptitle("1. Tín hiệu IMU đầu vào (Đã qua xử lý lọc thô)", fontweight='bold')
        
        imu_data = [
            (self.df_processed['Surge_ms2'], 'Surge Accel (m/s²)', 'r'), 
            (self.df_processed['Sway_ms2'], 'Sway Accel (m/s²)', 'g'), 
            (self.df_processed['Heave_ms2'], 'Heave Accel (m/s²)', 'b'),
            (self.df_processed['RollRate_rads'], 'Roll Rate (rad/s)', 'r'), 
            (self.df_processed['PitchRate_rads'], 'Pitch Rate (rad/s)', 'g'), 
            (self.df_processed['YawRate_rads'], 'Yaw Rate (rad/s)', 'b')
        ]
        
        for i, (sig, ti, c) in enumerate(imu_data):
            row, col = i//3, i%3
            if not self.plot_vars[keys_ch[i]].get():
                axs_imu[row, col].set_title(f"{ti} (Đã Ẩn)", color='gray')
                axs_imu[row, col].grid(True, alpha=0.3); continue
            axs_imu[row, col].plot(t, sig, color=c, lw=1.5)
            axs_imu[row, col].set_title(ti); axs_imu[row, col].grid(True)
            
        fig_imu.tight_layout()
        self.canvas_imu = FigureCanvasTkAgg(fig_imu, self.f_plot_imu)
        self.canvas_imu.draw()
        self.canvas_imu.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        # =================================================================
        if self.canvas_pos: self.canvas_pos.get_tk_widget().destroy()
        fig_pos, axs_pos = plt.subplots(2, 3, figsize=(12, 6)); fig_pos.suptitle("1. Vị trí & Góc bệ (Kinematics)", fontweight='bold')
        pos_data = [(self.df_processed['Platform_Surge_m'], 'Surge (m)', 'r'), (self.df_processed['Platform_Sway_m'], 'Sway (m)', 'g'), (self.df_processed['Platform_Heave_m'], 'Heave (m)', 'b'),
                    (self.df_processed['Platform_Roll_deg'], 'Roll (°)', 'r'), (self.df_processed['Platform_Pitch_deg'], 'Pitch (°)', 'g'), (self.df_processed['Platform_Yaw_deg'], 'Yaw (°)', 'b')]
        for i, (sig, ti, c) in enumerate(pos_data):
            if not self.plot_vars[keys_ch[i]].get(): axs_pos[i//3, i%3].set_title(f"{ti} (Đã Ẩn)", color='gray'); continue
            axs_pos[i//3, i%3].plot(t, sig, color=c, lw=1.5); axs_pos[i//3, i%3].set_title(ti); axs_pos[i//3, i%3].grid(True)
        fig_pos.tight_layout(); self.canvas_pos = FigureCanvasTkAgg(fig_pos, self.f_plot_pos); self.canvas_pos.draw(); self.canvas_pos.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        if self.canvas_sf: self.canvas_sf.get_tk_widget().destroy()
        fig_sf, axs_sf = plt.subplots(2, 3, figsize=(12, 6)); fig_sf.suptitle("2. Specific Force (Lực Cảm Nhận) & Sai số SF", fontweight='bold')
        sf_data = [('SF_Raw_Surge', 'SF_Plat_Surge', 'SF X (m/s²)', 'r'), ('SF_Raw_Sway', 'SF_Plat_Sway', 'SF Y (m/s²)', 'g'), ('SF_Raw_Heave', 'SF_Plat_Heave', 'SF Z (m/s²)', 'b')]
        for i, (t_c, s_c, ti, c) in enumerate(sf_data):
            if not self.plot_vars[keys_ch[i]].get(): continue
            axs_sf[0, i].plot(t, self.df_processed[t_c], color='gray', ls='--', lw=2); axs_sf[0, i].plot(t, self.df_processed[s_c], color=c, lw=1.5); axs_sf[0, i].set_title(ti); axs_sf[0, i].grid(True)
            axs_sf[1, i].plot(t, self.df_processed[s_c] - self.df_processed[t_c], color='purple', lw=1.5); axs_sf[1, i].axhline(0, color='k'); axs_sf[1, i].grid(True)
        fig_sf.tight_layout(); self.canvas_sf = FigureCanvasTkAgg(fig_sf, self.f_plot_sf); self.canvas_sf.draw(); self.canvas_sf.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.update_data_table()
        #================================================================
        # ---> MỚI: VẼ ĐỒ THỊ TÍN HIỆU TÙY CHỌN (TAB "TÍN HIỆU KHÁC")
        if hasattr(self, 'list_custom'):
            selected_indices = self.list_custom.curselection()
            valid_cols = [self.list_custom.get(i) for i in selected_indices if self.list_custom.get(i) in self.df_processed.columns]
            
            if self.canvas_custom: 
                self.canvas_custom.get_tk_widget().destroy()
                self.canvas_custom = None
                
            if valid_cols:
                fig_custom, ax_custom = plt.subplots(figsize=(10, 6))
                fig_custom.suptitle("Đồ thị Tín hiệu Tùy chọn (Custom Signals)", fontweight='bold')
                
                # Vẽ tất cả các kênh được chọn lên cùng một đồ thị
                for col in valid_cols:
                    ax_custom.plot(t, self.df_processed[col], label=col, lw=1.5)
                    
                ax_custom.set_xlabel("Thời gian Time (s)")
                ax_custom.grid(True)
                # Đặt chú thích (Legend) ra ngoài để không che mất đồ thị
                ax_custom.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                fig_custom.tight_layout()
                
                self.canvas_custom = FigureCanvasTkAgg(fig_custom, self.f_plot_custom)
                self.canvas_custom.draw()
                self.canvas_custom.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                
        # Gọi cập nhật bảng
        self.update_data_table()

    # --- AUTO-TUNE PSO ---
    def run_pso_18d(self):
        if self.df_processed is None: return messagebox.showwarning("Lỗi", "Vui lòng nạp dữ liệu!")
        try:
            self.p_num, self.p_iter = int(self.e_p.get()), int(self.e_i.get())
            self.p_w, self.p_c1, self.p_c2 = float(self.e_w.get()), float(self.e_c1.get()), float(self.e_c2.get())
            self.cur_w = {k: float(e.get()) for k, e in self.weight_entries.items()}
            self.code_str = self.text_code.get("1.0", tk.END); self.lims = [float(e.get()) for e in self.ws_entries]
            self.is_3rd, self.thr = "Bậc 3" in self.wo_type_var.get(), float(self.e_thr.get())
            self.base_p_mat = [[float(e.get()) for e in self.params_ui[ch][1:]] for ch in self.ch_names]
        except Exception as e: return messagebox.showerror("Lỗi", str(e))
        self.btn_pso.config(state=tk.DISABLED); threading.Thread(target=self._pso_worker_18d, daemon=True).start()

    def _pso_worker_18d(self):
        dim = 18; b_min, b_max = [0.1]*6 + [0.5]*12, [1.5]*6 + [5.0]*6 + [2.0]*6
        pts = np.random.uniform(b_min, b_max, (self.p_num, dim)); vels = np.zeros((self.p_num, dim))
        pbest, pbest_c = pts.copy(), np.full(self.p_num, np.inf)
        gbest, gbest_c = pts[0].copy(), np.inf
        t_sfx, t_sfy, t_sfz = self.df_processed['Surge_ms2'].values, self.df_processed['Sway_ms2'].values, self.df_processed['Heave_ms2'].values + 9.81
        t_wx, t_wy, t_wz = self.df_processed['RollRate_rads'].values, self.df_processed['PitchRate_rads'].values, self.df_processed['YawRate_rads'].values
        dt = 1.0 / self.fs
        def prsn(t, p): return np.corrcoef(t, p)[0, 1] if (np.std(p)!=0 and np.std(t)!=0) else 0.0

        self.cost_history = [] # MỚI: Khởi tạo mảng lưu lịch sử hội tụ

        for it in range(self.p_iter):
            for i in range(self.p_num):
                gains = pts[i][0:6]; mat = []
                for c in range(6): r = self.base_p_mat[c].copy(); r[0], r[1] = pts[i][6+c], pts[i][12+c]; mat.append(r)
                pos, ang, sf, _ = WashoutCore.run(self.fs, gains, mat, self.is_3rd, self.df_processed, self.thr)
                max_x, max_y, max_z = np.max(np.abs(pos[0])), np.max(np.abs(pos[1])), np.max(np.abs(pos[2]))
                max_r, max_p, max_yaw = np.max(np.abs(ang[0])), np.max(np.abs(ang[1])), np.max(np.abs(ang[2]))
                vars_dict = {
                    'max_x':max_x, 'max_y':max_y, 'max_z':max_z, 'max_r':max_r, 'max_p':max_p, 'max_yaw':max_yaw,
                    'lim_x':self.lims[0], 'lim_y':self.lims[1], 'lim_z':self.lims[2], 'lim_r':self.lims[3], 'lim_p':self.lims[4], 'lim_yaw':self.lims[5],
                    'pearson_x': prsn(t_sfx*gains[0], sf[0]), 'pearson_y': prsn(t_sfy*gains[1], sf[1]), 'pearson_z': prsn(t_sfz*gains[2], sf[2]),
                    'err_wx': np.mean((t_wx*gains[3] - np.gradient(np.radians(ang[0]), dt))**2), 'err_wy': np.mean((t_wy*gains[4] - np.gradient(np.radians(ang[1]), dt))**2), 'err_wz': np.mean((t_wz*gains[5] - np.gradient(np.radians(ang[2]), dt))**2),
                    'err_fx': np.mean((t_sfx*gains[0] - sf[0])**2), 'err_fy': np.mean((t_sfy*gains[1] - sf[1])**2), 'err_fz': np.mean((t_sfz*gains[2] - sf[2])**2),
                    'gain_x': gains[0], 'gain_y': gains[1], 'gain_z': gains[2], 'gain_r': gains[3], 'gain_p': gains[4], 'gain_yaw': gains[5], 'cost': 999999.0
                }
                vars_dict.update(self.cur_w)
                try: exec(self.code_str, {'np': np}, vars_dict); cost = vars_dict.get('cost', 999999.0)
                except Exception as e: print(f"Lỗi Cost: {e}"); self.root.after(0, lambda: self.btn_pso.config(state=tk.NORMAL)); return
                if cost < pbest_c[i]: pbest_c[i] = cost; pbest[i] = pts[i]
                if cost < gbest_c: gbest_c = cost; gbest = pts[i].copy()

            r1, r2 = np.random.rand(self.p_num, dim), np.random.rand(self.p_num, dim)
            vels = self.p_w * vels + self.p_c1 * r1 * (pbest - pts) + self.p_c2 * r2 * (gbest - pts)
            pts = np.clip(pts + vels, b_min, b_max)
            
            self.cost_history.append(gbest_c) # MỚI: Ghi lại Cost tốt nhất của vòng lặp này
            self.root.after(0, self.pso_progress.config, {'value': (it+1)/self.p_iter * 100})
            
        self.root.after(0, self._apply_pso_results, gbest)

    def _apply_pso_results(self, gbest):
        for i, ch in enumerate(self.ch_names):
            self.params_ui[ch][0].delete(0, tk.END); self.params_ui[ch][0].insert(0, f"{gbest[i]:.2f}")
            self.params_ui[ch][1].delete(0, tk.END); self.params_ui[ch][1].insert(0, f"{gbest[6+i]:.2f}")
            self.params_ui[ch][2].delete(0, tk.END); self.params_ui[ch][2].insert(0, f"{gbest[12+i]:.2f}")
        self.btn_pso.config(state=tk.NORMAL); self.apply_washout()
        messagebox.showinfo("Hoàn tất", "Đã tìm thấy thông số tối ưu!")
        # MỚI: Tự động bật đồ thị hội tụ khi PSO chạy xong
        self.plot_pso_convergence()
    # --- MỚI: HÀM VẼ ĐỒ THỊ HỘI TỤ PSO ---
    def plot_pso_convergence(self):
        if not hasattr(self, 'cost_history') or not self.cost_history:
            return
            
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.canvas.manager.set_window_title("PSO Convergence Graph")
        
        # Lọc bỏ các giá trị 999999.0 (nếu có ở những vòng đầu do phạt vượt giới hạn) để đồ thị dễ nhìn hơn
        valid_history = [c for c in self.cost_history if c < 999999.0]
        start_idx = len(self.cost_history) - len(valid_history) + 1
        
        if valid_history:
            ax.plot(range(start_idx, len(self.cost_history) + 1), valid_history, 'b-', lw=2, marker='o', markersize=4)
            ax.set_title("Biểu đồ Hội tụ Thuật toán PSO", fontweight='bold', fontsize=12)
            ax.set_xlabel("Vòng lặp (Iteration)", fontsize=10)
            ax.set_ylabel("Giá trị Hàm Mục Tiêu (Cost Function)", fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Đánh dấu điểm Cost thấp nhất (Tối ưu)
            min_cost = valid_history[-1]
            ax.annotate(f'Best Cost: {min_cost:.4f}', 
                        xy=(len(self.cost_history), min_cost), 
                        xytext=(len(self.cost_history)-len(valid_history)*0.2, min_cost + (max(valid_history)-min_cost)*0.1),
                        arrowprops=dict(facecolor='red', shrink=0.05, width=1.5, headwidth=6),
                        fontsize=10, color='red', fontweight='bold')
                        
            fig.tight_layout()
            plt.show(block=False)
        else:
            messagebox.showwarning("Cảnh báo", "Thuật toán không tìm được điểm nào thỏa mãn giới hạn không gian (Cost luôn = 999999). Hãy nới lỏng Limits!")

    # --- LOGIC BENCHMARK VÀ SO SÁNH 3 TÍN HIỆU ---
    # --- BENCHMARK ---
    def load_ref_config(self):
        choice = messagebox.askyesno("Tải Cấu hình", "Bạn có muốn tải thông số tham chiếu từ file .txt không?\n\n(Nhấn 'No' để sử dụng tham số mặc định của CKAS trích xuất từ màn hình hệ thống)")
        if choice:
            path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
            if path:
                try:
                    with open(path, 'r', encoding='utf-8') as f: text = f.read()
                    gains = []; p_mat = []
                    for ch in self.ch_names:
                        m = re.search(rf"\[{re.escape(ch)}\]\s*([\d\.-]+),([\d\.-]+),([\d\.-]+),([\d\.-]+),([\d\.-]+),([\d\.-]+)", text)
                        if m:
                            gains.append(float(m.group(1)))
                            p_mat.append([float(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5)), float(m.group(6))])
                        else:
                            gains.append(1.0); p_mat.append([1.5, 1.0, 0.0, 1.5, 1.0])
                    self.ref_params = {'gains': gains, 'p_mat': p_mat}
                    self.lbl_ref_stt.config(text=f"Đã nạp file: {os.path.basename(path)}", foreground="green")
                except Exception as e: return messagebox.showerror("Lỗi", str(e))
        else:
            # THIẾT LẬP THAM SỐ CHUẨN CỦA CKAS (TRÍCH XUẤT TỪ ẢNH CHỤP)
            self.ref_params = {
                # Gains theo thứ tự: Surge(X), Sway(Y), Heave(Z), Roll, Pitch, Yaw
                # Trích xuất từ tham số 'k' của từng dòng LHP và AHP
                'gains': [0.100, 0.100, 0.200, 0.100, 0.100, 0.200], 
                
                # Ma trận bộ lọc (p_mat): [HP wn, HP zeta, HP wb (Bậc 3), LP wn, LP zeta]
                # 'f' tương đương 'wn', 'z' tương đương 'zeta'
                'p_mat': [
                    [0.640, 1.500, 0.0, 1.000, 1.500], # Surge (X): LHP x & CT x
                    [0.640, 1.500, 0.0, 1.000, 1.500], # Sway (Y): LHP y & CT y
                    [0.640, 1.500, 0.0, 0.000, 0.000], # Heave (Z): LHP z
                    [0.300, 1.500, 0.0, 0.000, 0.000], # Roll: AHP roll
                    [0.300, 1.500, 0.0, 0.000, 0.000], # Pitch: AHP pitch
                    [0.300, 1.500, 0.0, 0.000, 0.000]  # Yaw: AHP yaw
                ]
            }
            self.lbl_ref_stt.config(text="Đã nạp bộ tham số CKAS Default (Từ hệ thống thật)", foreground="green")
            
        # Cập nhật hiển thị lên Tab 4
        txt = "THÔNG SỐ MẪU ĐANG DÙNG (CKAS REFERENCE):\n\n"
        for i, ch in enumerate(self.ch_names):
            txt += f"• {ch}: \tGain={self.ref_params['gains'][i]:.3f}   |   Params=[{', '.join(f'{x:.3f}' for x in self.ref_params['p_mat'][i])}]\n"
        if hasattr(self, 'lbl_ref_params'):
            self.lbl_ref_params.config(text=txt)

    def run_comparison(self):
        if self.ref_params is None or self.df_processed is None: 
            return messagebox.showwarning("Lỗi", "Cần nạp dữ liệu và cấu hình mẫu!")
        
        ch_sel = self.comp_ch_var.get()
        idx = self.ch_names.index(ch_sel)
        t = self.df_processed['Time_s'].values
        dt = 1.0 / self.fs
        
        # 1. Chạy mô phỏng cho cả 2 bộ tham số
        # Bộ Tuned (Hiện tại trên UI)
        pos_t, ang_t, sf_t = self.apply_washout(benchmark=True)
        # Bộ Reference (CKAS Mẫu)
        pos_r, ang_r, sf_r, _ = WashoutCore.run(self.fs, self.ref_params['gains'], self.ref_params['p_mat'], False, self.df_processed, 0.01)
        
        if self.canvas_comp: self.canvas_comp.get_tk_widget().destroy()
        fig, ax = plt.subplots(2, 1, figsize=(10, 8))
        fig.suptitle(f"So sánh Benchmark 3 tín hiệu: Kênh {ch_sel}", fontweight='bold')

        # --- LOGIC VẼ THEO CẶP TÍN HIỆU ĐẶC TRƯNG ---
        
        if idx == 0: # SURGE (X) -> Vẽ SF_x và Pitch Angle
            # Đồ thị 1: Specific Force X
            ax[0].plot(t, self.df_processed['Surge_ms2'] * float(self.params_ui[ch_sel][0].get()), 'gray', alpha=0.5, label='Target Accel')
            ax[0].plot(t, sf_r[0], 'k--', label='CKAS SF_x (Ref)')
            ax[0].plot(t, sf_t[0], 'r', lw=2, label='Tuned SF_x (PSO)')
            ax[0].set_title("Surge Specific Force (m/s²)")
            
            # Đồ thị 2: Pitch Angle (Tilt Coordination)
            ax[1].plot(t, ang_r[1], 'k--', label='CKAS Pitch (Ref)')
            ax[1].plot(t, ang_t[1], 'b', lw=2, label='Tuned Pitch (PSO)')
            ax[1].set_title("Pitch Tilt Angle (deg)")

        elif idx == 1: # SWAY (Y) -> Vẽ SF_y và Roll Angle
            ax[0].plot(t, self.df_processed['Sway_ms2'] * float(self.params_ui[ch_sel][0].get()), 'gray', alpha=0.5, label='Target Accel')
            ax[0].plot(t, sf_r[1], 'k--', label='CKAS SF_y (Ref)')
            ax[0].plot(t, sf_t[1], 'r', lw=2, label='Tuned SF_y (PSO)')
            ax[0].set_title("Sway Specific Force (m/s²)")
            
            ax[1].plot(t, ang_r[0], 'k--', label='CKAS Roll (Ref)')
            ax[1].plot(t, ang_t[0], 'b', lw=2, label='Tuned Roll (PSO)')
            ax[1].set_title("Roll Tilt Angle (deg)")

        elif idx == 2: # HEAVE (Z) -> Vẽ SF_z và Position Z
            ax[0].plot(t, self.df_processed['Heave_ms2'] * float(self.params_ui[ch_sel][0].get()) + 9.81, 'gray', alpha=0.5, label='Target Accel + G')
            ax[0].plot(t, sf_r[2], 'k--', label='CKAS SF_z (Ref)')
            ax[0].plot(t, sf_t[2], 'r', lw=2, label='Tuned SF_z (PSO)')
            ax[0].set_title("Heave Specific Force (m/s²)")
            
            ax[1].plot(t, pos_r[2], 'k--', label='CKAS Heave Pos (Ref)')
            ax[1].plot(t, pos_t[2], 'b', lw=2, label='Tuned Heave Pos (PSO)')
            ax[1].set_title("Heave Platform Position (m)")

        else: # CÁC KÊNH XOAY (Roll, Pitch, Yaw)
            # idx: 3->Roll, 4->Pitch, 5->Yaw. Mảng ang/rot index là idx-3
            rot_idx = idx - 3
            label_name = ["Roll", "Pitch", "Yaw"][rot_idx]
            in_col = ['RollRate_rads', 'PitchRate_rads', 'YawRate_rads'][rot_idx]
            
            # Đồ thị 1: Vận tốc góc (Angular Rate)
            rate_r = np.gradient(np.radians(ang_r[rot_idx]), dt)
            rate_t = np.gradient(np.radians(ang_t[rot_idx]), dt)
            
            ax[0].plot(t, self.df_processed[in_col] * float(self.params_ui[ch_sel][0].get()), 'gray', alpha=0.5, label='Input Rate')
            ax[0].plot(t, rate_r, 'k--', label=f'CKAS {label_name} Rate (Ref)')
            ax[0].plot(t, rate_t, 'r', lw=2, label=f'Tuned {label_name} Rate (PSO)')
            ax[0].set_title(f"{label_name} Angular Rate (rad/s)")
            
            # Đồ thị 2: Góc xoay (Angle)
            ax[1].plot(t, ang_r[rot_idx], 'k--', label=f'CKAS {label_name} Angle (Ref)')
            ax[1].plot(t, ang_t[rot_idx], 'b', lw=2, label=f'Tuned {label_name} Angle (PSO)')
            ax[1].set_title(f"{label_name} Platform Angle (deg)")

        for a in ax: a.legend(); a.grid(True)
        fig.tight_layout()
        self.canvas_comp = FigureCanvasTkAgg(fig, self.f_comp_plot)
        self.canvas_comp.draw()
        self.canvas_comp.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        # --- XUẤT FILE & SCRIPT ---
    # --- LIÊN KẾT 3D STEWART ĐƯỢC TỐI ƯU HÓA ---
    def launch_stewart_dashboard(self):
        if self.df_processed is None: return messagebox.showwarning("Lỗi", "Chưa có dữ liệu!")
        
        # 1. Lưu file quỹ đạo kết nối
        link_df = pd.DataFrame({'t': self.df_processed['Time_s'], 'x': self.df_processed['Platform_Surge_m']*1000.0, 'y': self.df_processed['Platform_Sway_m']*1000.0, 'z': self.df_processed['Platform_Heave_m']*1000.0, 'yaw': self.df_processed['Platform_Yaw_deg'], 'pitch': self.df_processed['Platform_Pitch_deg'], 'roll': self.df_processed['Platform_Roll_deg']})
        link_df.to_csv("stewart_link.csv", index=False, header=False, sep=' ')
        
        # 2. Tìm kiếm và chạy Dashboard 3D
        script_path = "dashboard_stewart.py"
        if not os.path.exists(script_path):
            resp = messagebox.askokcancel(
                "Yêu cầu Module 3D", 
                "Không tìm thấy file 'dashboard_stewart.py' trong thư mục hiện tại.\n\n"
                "Vui lòng tải module này từ link Google Drive của hệ thống:\n"
                "https://drive.google.com/file/d/1uezF5ZdslnLze63n1waYUXqX7G9o-OZR/view\n\n"
                "Nhấn OK nếu bạn đã tải về và muốn trỏ đường dẫn tới file đó."
            )
            if resp:
                script_path = filedialog.askopenfilename(title="Chọn file dashboard_stewart.py", filetypes=[("Python Files", "*.py")])
            else:
                return
                
        if script_path and os.path.exists(script_path):
            subprocess.Popen(["python", script_path])
        else:
            messagebox.showwarning("Cảnh báo", "Đã hủy thao tác hoặc đường dẫn file không hợp lệ.")
    def export_ckas_accel(self):
        if self.df_processed is None: return messagebox.showwarning("Lỗi", "Chưa có dữ liệu!")
        axi, ayi, azi = self.df_processed['Target_Surge_Accel_ms2'].values, self.df_processed['Target_Sway_Accel_ms2'].values, self.df_processed['Target_Heave_Accel_ms2'].values 
        wz, wy, wx = self.df_processed['Target_YawRate_rads'].values, self.df_processed['Target_PitchRate_rads'].values, self.df_processed['Target_RollRate_rads'].values
        dt_exp = float(self.export_dt_var.get())
        cmds = []
        for i in range(len(axi)):
            t = i * dt_exp
            cmds.append(f"{t:.4f} ~M 0.00 0.00 0.00 0.00 0.00 0.00 {axi[i]:.4f} {ayi[i]:.4f} {azi[i]:.4f} 0.0 0.0 9.81 {wz[i]:.4f} {wy[i]:.4f} {wx[i]:.4f}\n")
            
        path = filedialog.asksaveasfilename(defaultextension=".txt", initialfile="ckas_dynamic_only.txt")
        if path:
            with open(path, 'w') as f: f.writelines(cmds)
            messagebox.showinfo("Thành công", "Đã xuất file kịch bản Dynamic.")
            
            # --- TÍNH NĂNG MỚI: Hiển thị nội dung lên Script Viewer ---
            self.text_script.delete("1.0", tk.END)
            self.text_script.insert(tk.END, "".join(cmds))
            self.notebook.select(self.tab_process) # Tự động nhảy về Tab 1 để xem mã

    def generate_script_to_viewer(self):
        if self.df_processed is None: return
        x, y, z = self.df_processed['Platform_Surge_m'].values * 1000.0, self.df_processed['Platform_Sway_m'].values * 1000.0, self.df_processed['Platform_Heave_m'].values * 1000.0
        yaw, pitch, roll = self.df_processed['Platform_Yaw_deg'].values, self.df_processed['Platform_Pitch_deg'].values, self.df_processed['Platform_Roll_deg'].values
        dt_exp = float(self.export_dt_var.get()); t_arr = np.arange(len(x)) * dt_exp; cmds = []
        if not "15" in self.ckas_format_var.get():
            for i in range(len(x)): cmds.append(f"{t_arr[i]:.4f} ~M {x[i]:.2f} {y[i]:.2f} {z[i]:.2f} {yaw[i]:.2f} {pitch[i]:.2f} {roll[i]:.2f}\n")
        else:
            axi, ayi, azi = self.df_processed['Target_Surge_Accel_ms2'].values, self.df_processed['Target_Sway_Accel_ms2'].values, self.df_processed['Target_Heave_Accel_ms2'].values 
            wz, wy, wx = self.df_processed['Target_YawRate_rads'].values, self.df_processed['Target_PitchRate_rads'].values, self.df_processed['Target_RollRate_rads'].values
            for i in range(len(x)): cmds.append(f"{t_arr[i]:.4f} ~M {x[i]:.2f} {y[i]:.2f} {z[i]:.2f} {yaw[i]:.2f} {pitch[i]:.2f} {roll[i]:.2f} {axi[i]:.4f} {ayi[i]:.4f} {azi[i]:.4f} 0.0 0.0 9.81 {wz[i]:.4f} {wy[i]:.4f} {wx[i]:.4f}\n")
        self.text_script.delete("1.0", tk.END); self.text_script.insert(tk.END, "".join(cmds))

    def export_ckas_script(self):
        txt = self.text_script.get("1.0", tk.END).strip()
        path = filedialog.asksaveasfilename(defaultextension=".txt", initialfile="ckas_script.txt")
        if path and txt: open(path, 'w').write(txt)

    def save_config_txt(self):
        path = filedialog.asksaveasfilename(defaultextension=".txt", initialfile="washout_cfg.txt")
        if path:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(f"Type: {self.wo_type_var.get()}\nLP_Pre: {self.e_lp.get()}\nHP_Pre: {self.e_hp.get()}\n")
                for ch in self.ch_names: f.write(f"[{ch}] " + ",".join([e.get() for e in self.params_ui[ch]]) + "\n")
                f.write(f"Threshold: {self.e_thr.get()}\n")

    def load_config_txt(self):
        path = filedialog.askopenfilename()
        if path:
            try:
                with open(path, 'r', encoding='utf-8') as f: text = f.read()
                def fetch(pat): m = re.search(pat, text); return m.group(1) if m else None
                t = fetch(r"Type:\s*(.+)"); self.wo_type_var.set(t) if t else None
                lp, hp = fetch(r"LP_Pre:\s*([\d\.]+)"), fetch(r"HP_Pre:\s*([\d\.]+)")
                if lp: self.e_lp.delete(0, tk.END); self.e_lp.insert(0, lp)
                if hp: self.e_hp.delete(0, tk.END); self.e_hp.insert(0, hp)
                for ch in self.ch_names:
                    m = re.search(rf"\[{re.escape(ch)}\]\s*([\d\.-]+),([\d\.-]+),([\d\.-]+),([\d\.-]+),([\d\.-]+),([\d\.-]+)", text)
                    if m:
                        for i in range(6):
                            self.params_ui[ch][i].config(state=tk.NORMAL)
                            self.params_ui[ch][i].delete(0, tk.END); self.params_ui[ch][i].insert(0, m.group(i+1))
                            if ch in ["Heave (Z)", "Roll", "Pitch", "Yaw"] and i >= 4: self.params_ui[ch][i].config(state=tk.DISABLED)
                thr = fetch(r"Threshold:\s*([\d\.]+)"); 
                if thr: self.e_thr.delete(0, tk.END); self.e_thr.insert(0, thr)
            except Exception as e: messagebox.showerror("Lỗi", str(e))
# --- LOGIC LƯU/TẢI CẤU HÌNH PSO (DÙNG JSON) ---
    def save_pso_config(self):
        """Lưu toàn bộ thông số và mã Python của PSO ra file JSON"""
        path = filedialog.asksaveasfilename(defaultextension=".json", initialfile="pso_config.json", filetypes=[("JSON Files", "*.json")])
        if path:
            try:
                pso_data = {
                    "swarm": {
                        "particles": self.e_p.get(), "iterations": self.e_i.get(),
                        "w": self.e_w.get(), "c1": self.e_c1.get(), "c2": self.e_c2.get()
                    },
                    "limits": [e.get() for e in self.ws_entries],
                    "weights": {k: v.get() for k, v in self.weight_entries.items()},
                    "code": self.text_code.get("1.0", tk.END).strip()
                }
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(pso_data, f, indent=4, ensure_ascii=False)
                messagebox.showinfo("Thành công", f"Đã lưu toàn bộ cấu hình PSO vào:\n{os.path.basename(path)}")
            except Exception as e:
                messagebox.showerror("Lỗi khi lưu", str(e))

    def load_pso_config(self):
        """Tải thông số và mã Python của PSO từ file JSON lên giao diện"""
        path = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
        if path:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    pso_data = json.load(f)
                
                # Phục hồi bầy đàn
                if "swarm" in pso_data:
                    self.e_p.delete(0, tk.END); self.e_p.insert(0, pso_data["swarm"].get("particles", 20))
                    self.e_i.delete(0, tk.END); self.e_i.insert(0, pso_data["swarm"].get("iterations", 30))
                    self.e_w.delete(0, tk.END); self.e_w.insert(0, pso_data["swarm"].get("w", 0.5))
                    self.e_c1.delete(0, tk.END); self.e_c1.insert(0, pso_data["swarm"].get("c1", 1.5))
                    self.e_c2.delete(0, tk.END); self.e_c2.insert(0, pso_data["swarm"].get("c2", 1.5))
                
                # Phục hồi giới hạn
                if "limits" in pso_data:
                    for i, val in enumerate(pso_data["limits"]):
                        if i < len(self.ws_entries):
                            self.ws_entries[i].delete(0, tk.END); self.ws_entries[i].insert(0, val)
                            
                # Phục hồi trọng số
                if "weights" in pso_data:
                    for k, v in pso_data["weights"].items():
                        if k in self.weight_entries:
                            self.weight_entries[k].delete(0, tk.END); self.weight_entries[k].insert(0, v)
                            
                # Phục hồi mã lập trình hàm Cost
                if "code" in pso_data:
                    self.text_code.delete("1.0", tk.END)
                    self.text_code.insert(tk.END, pso_data["code"])
                    
                messagebox.showinfo("Thành công", "Đã tải cấu hình PSO thành công.")
            except Exception as e:
                messagebox.showerror("Lỗi nạp file", f"File không đúng định dạng hoặc bị lỗi:\n{e}")
    # --- KẾT NỐI & VIDEO ---
    def connect_hardware(self):
        try:
            if self.conn_mode.get() == "Serial":
                if not SERIAL_AVAILABLE: return
                self.serial_port = serial.Serial(self.e_com.get().strip(), 250000, timeout=1)
            else:
                self.udp_ip, self.udp_port = self.e_ip.get().strip(), 4000
                self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        except: pass

    def disconnect_hardware(self):
        try:
            if self.serial_port: self.serial_port.close()
            if self.udp_socket: self.udp_socket.close()
        except: pass

    def _send_raw(self, cmd_str):
        data = cmd_str.encode('ascii')
        if self.serial_port and self.serial_port.is_open: self.serial_port.write(data)
        elif self.udp_socket: self.udp_socket.sendto(data, (self.udp_ip, self.udp_port))

    def browse_video(self):
        path = filedialog.askopenfilename()
        if path:
            self.video_path_var.set(path)
            if self.cap: self.cap.release()
            self.cap = cv2.VideoCapture(path)
            self.show_video_frame()

    def close_fullscreen(self, event=None):
        if self.fullscreen_win: self.fullscreen_win.destroy(); self.fullscreen_win = None

    def show_video_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                mode = self.video_mode_var.get()
                if "Fullscreen" in mode and self.fullscreen_win and self.fullscreen_win.winfo_exists():
                    self.fullscreen_win.update_idletasks()
                    tw, th = self.fullscreen_win.winfo_width(), self.fullscreen_win.winfo_height()
                    if tw < 100: tw, th = 1920, 1080 
                    tgt = self.fullscreen_lbl
                else:
                    tw, th = self.lbl_video.winfo_width(), self.lbl_video.winfo_height()
                    if tw < 100: tw, th = 640, 480
                    tgt = self.lbl_video
                frame = cv2.cvtColor(cv2.resize(frame, (tw, th)), cv2.COLOR_BGR2RGB)
                imgtk = ImageTk.PhotoImage(image=Image.fromarray(frame))
                tgt.imgtk = imgtk; tgt.config(image=imgtk, text="")
            return ret
        return False

    def update_video_loop(self):
        if self.is_transmitting and self.video_playing:
            if self.show_video_frame():
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0 or np.isnan(fps): fps = 30
                self.root.after(int(1000/fps), self.update_video_loop)
            else: self.video_playing = False; self.close_fullscreen()

    def start_transmission(self):
        if self.df_processed is None: return
        self.cmd_list = []
        x, y, z = self.df_processed['Platform_Surge_m'].values * 1000.0, self.df_processed['Platform_Sway_m'].values * 1000.0, self.df_processed['Platform_Heave_m'].values * 1000.0
        yaw, pitch, roll = self.df_processed['Platform_Yaw_deg'].values, self.df_processed['Platform_Pitch_deg'].values, self.df_processed['Platform_Roll_deg'].values
        is_15 = "15" in self.ckas_format_var.get()
        for i in range(len(x)):
            if not is_15: self.cmd_list.append(f"~M {x[i]:.2f} {y[i]:.2f} {z[i]:.2f} {yaw[i]:.2f} {pitch[i]:.2f} {roll[i]:.2f}\r\n")
            else:
                axi, ayi, azi = self.df_processed['Target_Surge_Accel_ms2'].values, self.df_processed['Target_Sway_Accel_ms2'].values, self.df_processed['Target_Heave_Accel_ms2'].values 
                wz, wy, wx = self.df_processed['Target_YawRate_rads'].values, self.df_processed['Target_PitchRate_rads'].values, self.df_processed['Target_RollRate_rads'].values
                self.cmd_list.append(f"~M {x[i]:.2f} {y[i]:.2f} {z[i]:.2f} {yaw[i]:.2f} {pitch[i]:.2f} {roll[i]:.2f} {axi[i]:.4f} {ayi[i]:.4f} {azi[i]:.4f} 0.0 0.0 9.81 {wz[i]:.4f} {wy[i]:.4f} {wx[i]:.4f}\r\n")
        
        self.is_transmitting = True
        self.btn_tx.config(state=tk.DISABLED); self.btn_stop.config(state=tk.NORMAL)
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0); self.video_playing = True
            mode = self.video_mode_var.get()
            if "Fullscreen" in mode:
                self.fullscreen_win = tk.Toplevel(self.root)
                self.fullscreen_win.title("Video Sync")
                self.fullscreen_win.configure(bg="black")
                if "Màn 2" in mode: self.fullscreen_win.geometry(f"400x300+{self.root.winfo_screenwidth() + 50}+50")
                self.fullscreen_win.update_idletasks()
                self.fullscreen_win.attributes("-fullscreen", True)
                self.fullscreen_lbl = tk.Label(self.fullscreen_win, bg="black")
                self.fullscreen_lbl.pack(fill=tk.BOTH, expand=True)
                self.fullscreen_win.bind("<Escape>", self.close_fullscreen)
            self.update_video_loop()
        threading.Thread(target=self.tx_worker, daemon=True).start()

    def stop_transmission(self):
        self.is_transmitting = False; self.video_playing = False
        self.close_fullscreen()
        self.btn_tx.config(state=tk.NORMAL); self.btn_stop.config(state=tk.DISABLED)
        try: self._send_raw("~M 0 0 0 0 0 0\r\n")
        except: pass

    def tx_worker(self):
        total = len(self.cmd_list); sleep_time = 1.0 / self.fs
        for i, cmd in enumerate(self.cmd_list):
            if not self.is_transmitting: break
            try: self._send_raw(cmd)
            except: self.is_transmitting = False; break
            if i % 10 == 0: self.root.after(0, self.progress_var.set, (i/total)*100); self.root.after(0, self.lbl_tx_prog.config, {'text': f"{i}/{total} lệnh"})
            time.sleep(sleep_time)
        self.stop_transmission()
        # =====================================================================
    # --- CÔNG CỤ CONVERT & NỘI SUY TẦN SỐ (DIALOG ĐỘC LẬP) ---
    # =====================================================================
    def open_converter_tool(self):
        win = tk.Toplevel(self.root)
        win.title("Công cụ Convert & Nội suy Tần số xuất kịch bản CKAS")
        win.geometry("950x700")
        win.transient(self.root) # Nổi trên cửa sổ chính
        
        # Biến trạng thái nội bộ
        self.tool_df = None
        if self.df_processed is not None and 'Platform_Surge_m' in self.df_processed.columns:
            self.tool_df = self.df_processed.copy()
            stt = "Trạng thái: Đã tự động nạp dữ liệu từ chương trình chính"
        else:
            stt = "Trạng thái: Chưa có dữ liệu. Vui lòng nạp File CSV hoặc chạy Washout ở app chính."

        # Giao diện Top
        f_top = ttk.Frame(win); f_top.pack(fill=tk.X, padx=10, pady=10)
        lbl_stt = ttk.Label(f_top, text=stt, foreground="blue", font=("Arial", 10, "bold"))
        lbl_stt.pack(side=tk.TOP, anchor=tk.W, pady=5)
        
        f_ctrl = tk.Frame(f_top); f_ctrl.pack(fill=tk.X, pady=5)
        tk.Button(f_ctrl, text="🔄 Nạp lại Data từ Main App", bg="#E0E0E0", command=lambda: load_main()).pack(side=tk.LEFT, padx=5)
        tk.Button(f_ctrl, text="📂 Nạp file CSV cần Convert", command=lambda: load_file()).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(f_ctrl, text=" |  Thời gian update dt(s):", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=(15, 5))
        e_dt = ttk.Entry(f_ctrl, width=8); e_dt.insert(0, "0.0100"); e_dt.pack(side=tk.LEFT)
        ttk.Label(f_ctrl, text="(Bệ CKAS thường dùng 100Hz = 0.01s)", foreground="gray").pack(side=tk.LEFT, padx=2)
        
        ttk.Label(f_ctrl, text="Format:").pack(side=tk.LEFT, padx=(15, 5))
        cb_fmt = ttk.Combobox(f_ctrl, values=["6 Tham số", "15 Tham số (Dynamic)"], state="readonly", width=20)
        cb_fmt.set("15 Tham số (Dynamic)"); cb_fmt.pack(side=tk.LEFT)
        
        f_act = tk.Frame(win); f_act.pack(fill=tk.X, padx=10)
        tk.Button(f_act, text="⚡ CHẠY NỘI SUY & XUẤT M-CODE", bg="#FF9800", fg="white", font=("Arial", 10, "bold"), command=lambda: do_convert()).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(f_act, text="💾 LƯU FILE KỊCH BẢN (.txt)", bg="#4CAF50", fg="white", font=("Arial", 10, "bold"), command=lambda: save_file()).pack(side=tk.RIGHT, padx=5, pady=5)

        # Giao diện Bottom (Trình xem Text)
        f_bot = ttk.Frame(win); f_bot.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        txt_out = tk.Text(f_bot, wrap=tk.NONE, font=("Consolas", 10), bg="#1E1E1E", fg="#00FF00")
        sy = ttk.Scrollbar(f_bot, orient=tk.VERTICAL, command=txt_out.yview)
        sx = ttk.Scrollbar(f_bot, orient=tk.HORIZONTAL, command=txt_out.xview)
        txt_out.config(yscrollcommand=sy.set, xscrollcommand=sx.set)
        sx.pack(side=tk.BOTTOM, fill=tk.X); sy.pack(side=tk.RIGHT, fill=tk.Y); txt_out.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # --- CÁC HÀM XỬ LÝ LOGIC NỘI BỘ TRONG HỘP THOẠI ---
        def load_main():
            if self.df_processed is not None and 'Platform_Surge_m' in self.df_processed.columns:
                self.tool_df = self.df_processed.copy()
                lbl_stt.config(text="Trạng thái: Đã nạp lại dữ liệu Washout từ chương trình chính", foreground="blue")
            else: messagebox.showwarning("Lỗi", "Chương trình chính chưa có kết quả Washout hoàn chỉnh!")

        def load_file():
            path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
            if path:
                try:
                    tmp_df = pd.read_csv(path)
                    if 'Platform_Surge_m' not in tmp_df.columns:
                        return messagebox.showerror("Lỗi", "File CSV tải lên không chứa dữ liệu kết quả Washout!")
                    self.tool_df = tmp_df
                    lbl_stt.config(text=f"Trạng thái: Đang dùng Data từ File tải lên ({os.path.basename(path)})", foreground="green")
                except Exception as e: messagebox.showerror("Lỗi đọc file", str(e))

        def do_convert():
            if self.tool_df is None: return messagebox.showwarning("Lỗi", "Chưa có dữ liệu để nội suy!")
            try:
                dt_new = float(e_dt.get())
                if dt_new <= 0: raise ValueError
            except: return messagebox.showerror("Lỗi", "Thời gian dt(s) không hợp lệ!")
                
            df = self.tool_df
            t_old = df['Time_s'].values
            t_new = np.arange(0, t_old[-1], dt_new) # Tạo lưới thời gian nội suy mới
            
            # Nội suy vị trí và góc
            x = np.interp(t_new, t_old, df['Platform_Surge_m'].values * 1000.0)
            y = np.interp(t_new, t_old, df['Platform_Sway_m'].values * 1000.0)
            z = np.interp(t_new, t_old, df['Platform_Heave_m'].values * 1000.0)
            yaw = np.interp(t_new, t_old, df['Platform_Yaw_deg'].values)
            pitch = np.interp(t_new, t_old, df['Platform_Pitch_deg'].values)
            roll = np.interp(t_new, t_old, df['Platform_Roll_deg'].values)
            
            is_15 = "15" in cb_fmt.get()
            if is_15:
                req = ['Target_Surge_Accel_ms2', 'Target_Sway_Accel_ms2', 'Target_Heave_Accel_ms2', 'Target_YawRate_rads', 'Target_PitchRate_rads', 'Target_RollRate_rads']
                if any(c not in df.columns for c in req):
                    return messagebox.showerror("Lỗi tương thích", "Dữ liệu hiện tại thiếu các tham số động lực học (Target Accel/Rate) để xuất kịch bản 15 tham số. Hãy kiểm tra lại file CSV.")
                
                axi = np.interp(t_new, t_old, df['Target_Surge_Accel_ms2'].values)
                ayi = np.interp(t_new, t_old, df['Target_Sway_Accel_ms2'].values)
                azi = np.interp(t_new, t_old, df['Target_Heave_Accel_ms2'].values)
                wz = np.interp(t_new, t_old, df['Target_YawRate_rads'].values)
                wy = np.interp(t_new, t_old, df['Target_PitchRate_rads'].values)
                wx = np.interp(t_new, t_old, df['Target_RollRate_rads'].values)

            # Khởi tạo mã kịch bản
            cmds = []
            for i in range(len(t_new)):
                if not is_15: 
                    cmds.append(f"{t_new[i]:.4f} ~M {x[i]:.2f} {y[i]:.2f} {z[i]:.2f} {yaw[i]:.2f} {pitch[i]:.2f} {roll[i]:.2f}\n")
                else: 
                    cmds.append(f"{t_new[i]:.4f} ~M {x[i]:.2f} {y[i]:.2f} {z[i]:.2f} {yaw[i]:.2f} {pitch[i]:.2f} {roll[i]:.2f} {axi[i]:.4f} {ayi[i]:.4f} {azi[i]:.4f} 0.0 0.0 9.81 {wz[i]:.4f} {wy[i]:.4f} {wx[i]:.4f}\n")
            
            txt_out.delete("1.0", tk.END)
            txt_out.insert(tk.END, "".join(cmds))
            messagebox.showinfo("Thành công", f"Nội suy hoàn tất.\nĐã tạo ra {len(t_new)} dòng lệnh M-Code chạy ở tần số {1/dt_new:.0f}Hz.")

        def save_file():
            content = txt_out.get("1.0", tk.END).strip()
            if not content: return messagebox.showwarning("Trống", "Chưa có kịch bản nào được tạo để lưu!")
            p = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")], initialfile=f"ckas_script_converted.txt")
            if p:
                try: 
                    open(p, 'w').write(content)
                    messagebox.showinfo("Thành công", f"Đã lưu kịch bản nội suy ra file:\n{p}")
                except Exception as e: messagebox.showerror("Lỗi lưu file", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = IMUAnalyzerApp(root)
    root.mainloop()
