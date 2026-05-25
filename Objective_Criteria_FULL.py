import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import scipy.io as sio
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import curve_fit
import pingouin as pg
from statsmodels.formula.api import ols
import warnings
import os
import itertools # Thêm thư viện này ở đầu file python của bạn
from sklearn.metrics import mean_squared_error # Để tính RMSE

warnings.filterwarnings('ignore')

# =========================================================================
# LÕI TOÁN HỌC: TÍNH TOÁN TIÊU CHÍ (TỔNG HỢP 3D & CHI TIẾT TỪNG TRỤC)
# =========================================================================
def calculate_custom_dof_metrics(df, f_mask, w_mask, centrifuge_mode=False):
    try:
        t = df['t'].values; dt = np.mean(np.diff(t)) if len(t) > 1 else 0.01
        
        ax_sim_vals = df['ax_sim'].values.copy()
        ay_sim_vals = df['ay_sim'].values.copy()
        az_sim_vals = df['az_sim'].values.copy()
        wx_sim_vals = df['wx_sim'].values.copy()
        wy_sim_vals = df['wy_sim'].values.copy()
        wz_sim_vals = df['wz_sim'].values.copy()
        px_sim_vals = df['px_sim'].values.copy()
        py_sim_vals = df['py_sim'].values.copy()
        pz_sim_vals = df['pz_sim'].values.copy()
        
        # --- TÍCH HỢP GIA TỐC HƯỚNG TÂM (CENTRIFUGE MODE) ---
        if centrifuge_mode:
            r_xE0 = 0.465; r_yE0 = 2.637; bE0 = 0.35
            R_cir = np.sqrt(r_yE0**2 + r_xE0**2) + bE0
            
            if 'pa_y_sim' in df.columns: ay_accel = df['pa_y_sim'].values
            else:
                vy = np.gradient(py_sim_vals, dt)
                ay_accel = np.gradient(vy, dt)
                
            alpha_dotdot = ay_accel / R_cir
            alpha_dot = cumulative_trapezoid(alpha_dotdot, t, initial=0) 
            centri_accel = R_cir * (alpha_dot**2)
            
            ax_sim_vals = centri_accel
            wz_sim_vals = alpha_dot
        
        # 1. TÍNH TOÁN TỔNG HỢP 3D VECTOR
        a_ref = np.vstack([df['ax_ref']*f_mask[0], df['ay_ref']*f_mask[1], df['az_ref']*f_mask[2]])
        f_sim = np.vstack([ax_sim_vals*f_mask[0], ay_sim_vals*f_mask[1], az_sim_vals*f_mask[2]])
        da_ref = np.gradient(a_ref, dt, axis=1); df_sim_grad = np.gradient(f_sim, dt, axis=1)
        
        err_f = a_ref - f_sim; err_df = da_ref - df_sim_grad
        l1_f = np.mean(np.linalg.norm(err_f, axis=0)); l2_f = np.mean(np.linalg.norm(err_df, axis=0))
        
        w_ref = np.vstack([df['wx_ref']*w_mask[0], df['wy_ref']*w_mask[1], df['wz_ref']*w_mask[2]])
        w_sim = np.vstack([wx_sim_vals*w_mask[0], wy_sim_vals*w_mask[1], wz_sim_vals*w_mask[2]])
        dw_ref = np.gradient(w_ref, dt, axis=1); dw_sim = np.gradient(w_sim, dt, axis=1)
        
        err_w = w_ref - w_sim; err_dw = dw_ref - dw_sim
        l1_w = np.mean(np.linalg.norm(err_w, axis=0)); l2_w = np.mean(np.linalg.norm(err_dw, axis=0))
        
        a_max = 5.0; w_max = 1.0; delta_O = 0.17; delta_S = 3.0 * np.pi / 180
        max_a_ref = max(np.max(np.linalg.norm(a_ref, axis=0)), 1.0)
        max_w_ref = max(np.max(np.linalg.norm(w_ref, axis=0)), 1.0)
        
        pouliot_L1 = (l1_f / a_max)**2 + (l1_w / w_max)**2
        pouliot_L2 = (l2_f / a_max)**2 + (l2_w / w_max)**2
        
        rms_a_ref = np.sqrt(np.mean(np.linalg.norm(a_ref, axis=0)**2))
        rms_f_sim = np.sqrt(np.mean(np.linalg.norm(f_sim, axis=0)**2))
        k_f = rms_f_sim / rms_a_ref if rms_a_ref > 0 else 0
        
        rms_w_ref = np.sqrt(np.mean(np.linalg.norm(w_ref, axis=0)**2))
        rms_w_sim = np.sqrt(np.mean(np.linalg.norm(w_sim, axis=0)**2))
        k_w = rms_w_sim / rms_w_ref if rms_w_ref > 0 else 0
        
        f_scale_err = np.mean(np.linalg.norm(a_ref * (1 - k_f), axis=0))
        f_shape_err = np.mean(np.linalg.norm(k_f * a_ref - f_sim, axis=0))
        w_scale_err = np.mean(np.linalg.norm(w_ref * (1 - k_w), axis=0))
        w_shape_err = np.mean(np.linalg.norm(k_w * w_ref - w_sim, axis=0))
        
        fischer_L1 = ((f_scale_err + f_shape_err)/max_a_ref)**2 + ((w_scale_err + w_shape_err)/max_w_ref)**2
        fischer_L2 = (l2_f / max_a_ref)**2 + (l2_w / max_w_ref)**2 
        norm_L1 = ((f_scale_err + f_shape_err)/delta_O)**2 + ((w_scale_err + w_shape_err)/delta_S)**2
        norm_L2 = (l2_f / delta_O)**2 + (l2_w / delta_S)**2
        
        score_F15 = 4.10 - 0.05*pouliot_L1 - 0.08*fischer_L1 - 0.15*norm_L1 
        
        # 2. TÍNH TOÁN SAI SỐ CHI TIẾT THEO TỪNG TRỤC
        def calc_1d_err(ref, sim, active):
            if not active: return 0.0, 0.0
            r_rms = np.sqrt(np.mean(ref**2)); s_rms = np.sqrt(np.mean(sim**2))
            k = s_rms / r_rms if r_rms > 0 else 0.0
            sc = np.mean(np.abs(ref * (1 - k)))
            sh = np.mean(np.abs(k * ref - sim))
            return sc, sh

        sc_fx, sh_fx = calc_1d_err(df['ax_ref'], ax_sim_vals, f_mask[0])
        sc_fy, sh_fy = calc_1d_err(df['ay_ref'], ay_sim_vals, f_mask[1])
        sc_fz, sh_fz = calc_1d_err(df['az_ref'], az_sim_vals, f_mask[2])
        
        sc_wx, sh_wx = calc_1d_err(df['wx_ref'], wx_sim_vals, w_mask[0])
        sc_wy, sh_wy = calc_1d_err(df['wy_ref'], wy_sim_vals, w_mask[1])
        sc_wz, sh_wz = calc_1d_err(df['wz_ref'], wz_sim_vals, w_mask[2])
        
        p_sim_mat = np.vstack([px_sim_vals*f_mask[0], py_sim_vals*f_mask[1], pz_sim_vals*f_mask[2]])
        max_pos = np.max(np.linalg.norm(p_sim_mat, axis=0))
        
        return {
            'Score F15': score_F15, 'Max Position (m)': max_pos,
            'Pouliot L1': pouliot_L1, 'Pouliot L2': pouliot_L2,
            'Fischer L1': fischer_L1, 'Fischer L2': fischer_L2, 
            'Norm L1': norm_L1, 'Norm L2': norm_L2,
            'Scale Err fx (Surge)': sc_fx, 'Shape Err fx (Surge)': sh_fx,
            'Scale Err fy (Sway)': sc_fy, 'Shape Err fy (Sway)': sh_fy,
            'Scale Err fz (Heave)': sc_fz, 'Shape Err fz (Heave)': sh_fz,
            'Scale Err wx (Roll)': sc_wx, 'Shape Err wx (Roll)': sh_wx,
            'Scale Err wy (Pitch)': sc_wy, 'Shape Err wy (Pitch)': sh_wy,
            'Scale Err wz (Yaw)': sc_wz, 'Shape Err wz (Yaw)': sh_wz
        }
    except Exception as e:
        raise ValueError(f"Lỗi tính toán: {e}")

# =========================================================================
# GIAO DIỆN CHÍNH
# =========================================================================
class MCA6DOFEvaluatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Universal MCA Evaluator - Subjective & Objective Benchmark")
        self.root.geometry("1500x850")
        
        style = ttk.Style(); style.theme_use('clam')
        style.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"), background="#d0e1f9")
        
        self.datasets = {} 
        self.df_subj = None
        self.df_merged = None
        
        self.metrics_list = [
            'Score F15', 'Pouliot L1', 'Pouliot L2', 'Fischer L1', 'Fischer L2', 'Norm L1', 'Norm L2',
            'Scale Err fx (Surge)', 'Shape Err fx (Surge)', 
            'Scale Err fy (Sway)', 'Shape Err fy (Sway)', 
            'Scale Err fz (Heave)', 'Shape Err fz (Heave)',
            'Scale Err wx (Roll)', 'Shape Err wx (Roll)', 
            'Scale Err wy (Pitch)', 'Shape Err wy (Pitch)', 
            'Scale Err wz (Yaw)', 'Shape Err wz (Yaw)'
        ]
        
        self.f_axis_vars = [tk.BooleanVar(value=True), tk.BooleanVar(value=True), tk.BooleanVar(value=True)] 
        self.w_axis_vars = [tk.BooleanVar(value=True), tk.BooleanVar(value=True), tk.BooleanVar(value=True)] 
        self.centrifuge_var = tk.BooleanVar(value=False)
        
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        self.tab_data = ttk.Frame(self.notebook)
        self.tab_compare = ttk.Frame(self.notebook)
        self.tab_adv_compare = ttk.Frame(self.notebook)
        self.tab_subj = ttk.Frame(self.notebook) # TAB MỚI
        
        self.notebook.add(self.tab_data, text=' 📁 Nạp Dữ Liệu Khách quan (.MAT) ')
        self.notebook.add(self.tab_compare, text=' 📊 Biểu đồ Cột So sánh ')
        self.notebook.add(self.tab_adv_compare, text=' 📉 Đánh giá Nâng cao (2D/3D) ')
        self.notebook.add(self.tab_subj, text=' 🧑‍🔬 Đánh giá Chủ quan & Tương quan ')
        
        self.setup_data_tab()
        self.setup_compare_tab()
        self.setup_adv_compare_tab()
        self.setup_subj_tab()

    # -----------------------------------------------------------------------
    # TAB 1: DATA KHÁCH QUAN
    # -----------------------------------------------------------------------
    def setup_data_tab(self):
        f_left = ttk.Frame(self.tab_data, width=350); f_left.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        f_left.pack_propagate(False)
        
        f_ctrl = ttk.LabelFrame(f_left, text=" 1. QUẢN LÝ DỮ LIỆU ĐẦU VÀO ")
        f_ctrl.pack(fill=tk.X, pady=5)
        ttk.Button(f_ctrl, text="📥 Tải nhiều file CSV / MAT", command=self.load_file).pack(fill=tk.X, padx=10, pady=(10,5))
        ttk.Checkbutton(f_ctrl, text="🧮 Tích hợp Gia tốc Hướng tâm (Centrifuge)", variable=self.centrifuge_var, command=self.recalculate_all).pack(anchor=tk.W, padx=10, pady=5)
        ttk.Button(f_ctrl, text="🗑 Xóa tất cả Dữ liệu", command=self.clear_data).pack(fill=tk.X, padx=10, pady=5)
        ttk.Button(f_ctrl, text="💾 Xuất Toàn bộ Bảng (CSV)", command=self.export_results).pack(fill=tk.X, padx=10, pady=(5,10))
        
        f_axis = ttk.LabelFrame(f_left, text=" 2. LỰA CHỌN TỔ HỢP TRỤC ")
        f_axis.pack(fill=tk.X, pady=10)
        f_t_axis = ttk.Frame(f_axis); f_t_axis.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(f_t_axis, text="Tịnh tiến (Lực):", font=("Segoe UI", 9, "bold"), foreground="#005a9e").pack(anchor=tk.W)
        ttk.Checkbutton(f_t_axis, text="Surge", variable=self.f_axis_vars[0], command=self.recalculate_all).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(f_t_axis, text="Sway", variable=self.f_axis_vars[1], command=self.recalculate_all).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(f_t_axis, text="Heave", variable=self.f_axis_vars[2], command=self.recalculate_all).pack(side=tk.LEFT, padx=5)
        
        f_r_axis = ttk.Frame(f_axis); f_r_axis.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(f_r_axis, text="Xoay (Vận tốc góc):", font=("Segoe UI", 9, "bold"), foreground="#d83b01").pack(anchor=tk.W)
        ttk.Checkbutton(f_r_axis, text="Roll", variable=self.w_axis_vars[0], command=self.recalculate_all).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(f_r_axis, text="Pitch", variable=self.w_axis_vars[1], command=self.recalculate_all).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(f_r_axis, text="Yaw", variable=self.w_axis_vars[2], command=self.recalculate_all).pack(side=tk.LEFT, padx=5)
        
        f_table = ttk.LabelFrame(self.tab_data, text=" BẢNG TỔNG HỢP VÀ PHÂN TÍCH SAI SỐ TỪNG TRỤC ")
        f_table.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        cols = ["Thuật toán"] + self.metrics_list
        self.tv_results = ttk.Treeview(f_table, columns=cols, show="headings", height=20)
        
        for c in cols:
            self.tv_results.heading(c, text=c)
            w = 200 if c == "Thuật toán" else 120
            self.tv_results.column(c, width=w, anchor=tk.CENTER, stretch=False)
            
        scroll_y = ttk.Scrollbar(f_table, orient=tk.VERTICAL, command=self.tv_results.yview)
        scroll_x = ttk.Scrollbar(f_table, orient=tk.HORIZONTAL, command=self.tv_results.xview)
        self.tv_results.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
        
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y); scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.tv_results.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # -----------------------------------------------------------------------
    # TAB 2: BIỂU ĐỒ SO SÁNH
    # -----------------------------------------------------------------------
    def setup_compare_tab(self):
        f_ctrl = ttk.Frame(self.tab_compare)
        f_ctrl.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(f_ctrl, text="Chọn Tiêu chí:", font=("Segoe UI", 11, "bold")).pack(side=tk.LEFT, padx=5)
        self.cb_metric = ttk.Combobox(f_ctrl, values=self.metrics_list, state='readonly', width=35)
        self.cb_metric.current(0)
        self.cb_metric.pack(side=tk.LEFT, padx=10)
        self.cb_metric.bind('<<ComboboxSelected>>', lambda e: self.plot_comparison())
        
        # NÚT XUẤT CSV CHO 1 TIÊU CHÍ (Yêu cầu 1)
        ttk.Button(f_ctrl, text="📥 Xuất Bảng Riêng (CSV)", command=self.export_single_metric_table).pack(side=tk.LEFT, padx=10)
        
        f_plot = ttk.Frame(self.tab_compare)
        f_plot.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=f_plot)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # -----------------------------------------------------------------------
    # TAB 3: ĐÁNH GIÁ NÂNG CAO 2D/3D
    # -----------------------------------------------------------------------
    def setup_adv_compare_tab(self):
        f_ctrl = ttk.Frame(self.tab_adv_compare)
        f_ctrl.pack(fill=tk.X, padx=10, pady=10)
        
        f_ctrl_3d = ttk.LabelFrame(f_ctrl, text=" Cấu hình Biểu đồ 3D ")
        f_ctrl_3d.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        f_3d_top = ttk.Frame(f_ctrl_3d); f_3d_top.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(f_3d_top, text="Trục X,Y,Z biểu diễn:").pack(side=tk.LEFT)
        self.cb_3d_type = ttk.Combobox(f_3d_top, values=[
            "Shape Error Lực (Surge, Sway, Heave)", "Scale Error Lực (Surge, Sway, Heave)",
            "Shape Error Vận tốc góc (Roll, Pitch, Yaw)", "Scale Error Vận tốc góc (Roll, Pitch, Yaw)"
        ], state='readonly', width=40)
        self.cb_3d_type.current(0)
        self.cb_3d_type.pack(side=tk.LEFT, padx=5)
        self.cb_3d_type.bind('<<ComboboxSelected>>', lambda e: self.plot_adv_comparison())

        # GÓC NHÌN CHUẨN (Yêu cầu 2: Bỏ slider, thay bằng button)
        f_3d_bot = ttk.Frame(f_ctrl_3d); f_3d_bot.pack(fill=tk.X, padx=5, pady=(2,5))
        ttk.Label(f_3d_bot, text="Góc nhìn 2D/3D:").pack(side=tk.LEFT, padx=5)
        ttk.Button(f_3d_bot, text="Phối cảnh (Iso)", command=lambda: self.update_3d_view(30, 45)).pack(side=tk.LEFT, padx=2)
        ttk.Button(f_3d_bot, text="Từ trên (Top)", command=lambda: self.update_3d_view(90, -90)).pack(side=tk.LEFT, padx=2)
        ttk.Button(f_3d_bot, text="Mặt trước (Front)", command=lambda: self.update_3d_view(0, -90)).pack(side=tk.LEFT, padx=2)
        ttk.Button(f_3d_bot, text="Mặt bên (Side)", command=lambda: self.update_3d_view(0, 0)).pack(side=tk.LEFT, padx=2)

        f_ctrl_2d = ttk.LabelFrame(f_ctrl, text=" Cấu hình Biểu đồ 2D (L1 vs L2) ")
        f_ctrl_2d.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)
        f_2d_top = ttk.Frame(f_ctrl_2d); f_2d_top.pack(fill=tk.X, padx=5, pady=10)
        ttk.Label(f_2d_top, text="So sánh tiêu chuẩn:").pack(side=tk.LEFT)
        self.cb_2d_type = ttk.Combobox(f_2d_top, values=["Pouliot (L1 vs L2)", "Fischer (L1 vs L2)", "Normalized (L1 vs L2)"], state='readonly', width=30)
        self.cb_2d_type.current(1) 
        self.cb_2d_type.pack(side=tk.LEFT, padx=5)
        self.cb_2d_type.bind('<<ComboboxSelected>>', lambda e: self.plot_adv_comparison())

        f_plot = ttk.Frame(self.tab_adv_compare)
        f_plot.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.fig_adv = plt.Figure(figsize=(12, 6))
        self.ax_3d = self.fig_adv.add_subplot(121, projection='3d')
        self.ax_2d = self.fig_adv.add_subplot(122)
        self.canvas_adv = FigureCanvasTkAgg(self.fig_adv, master=f_plot)
        self.canvas_adv.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # -----------------------------------------------------------------------
    # TAB 4: ĐÁNH GIÁ CHỦ QUAN VÀ TƯƠNG QUAN
    # -----------------------------------------------------------------------
    def setup_subj_tab(self):
        # KHUNG TRÁI: Dữ liệu Chủ quan & Lọc nhiễu
        f_left = ttk.Frame(self.tab_subj) # Đổi self.tab3 thành tên tab tương ứng của bạn
        f_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        f_input = ttk.LabelFrame(f_left, text=" 1. Nạp & Lọc Dữ liệu Chủ quan ")
        f_input.pack(fill=tk.X, pady=5)
        ttk.Button(f_input, text="Tải File CSV", command=self.process_subjective_v2).pack(side=tk.LEFT, padx=5, pady=5)
        
        # Bảng hiển thị Raw vs Filtered
        cols_data = ("Subject", "Algorithm", "Raw", "Filtered", "Ghi chú")
        self.tv_data = ttk.Treeview(f_left, columns=cols_data, show="headings", height=8)
        for c in cols_data: self.tv_data.heading(c, text=c)
        self.tv_data.column("Subject", width=60); self.tv_data.column("Algorithm", width=100)
        self.tv_data.column("Raw", width=60); self.tv_data.column("Filtered", width=60)
        self.tv_data.column("Ghi chú", width=180)
        self.tv_data.pack(fill=tk.BOTH, expand=True, pady=5)
        self.tv_data.tag_configure('changed', background='#ffcccc') # Highlight màu đỏ nhạt nếu có thay đổi
        
        # Biểu đồ Mean & Variance
        self.fig_mean, self.ax_mean = plt.subplots(figsize=(5, 3))
        self.canvas_mean = FigureCanvasTkAgg(self.fig_mean, master=f_left)
        self.canvas_mean.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # KHUNG PHẢI: Phân tích Tương quan Tổ hợp
        f_right = ttk.Frame(self.tab_subj)
        f_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        f_config = ttk.LabelFrame(f_right, text=" 2. Phân tích Tương quan Tổ hợp (Objective vs Subjective) ")
        f_config.pack(fill=tk.X, pady=5)
        
        ttk.Label(f_config, text="Chọn các tiêu chí Khách quan (Giữ Ctrl để chọn nhiều):").pack(anchor=tk.W, padx=5)
        self.lb_metrics = tk.Listbox(f_config, selectmode=tk.MULTIPLE, height=5)
        self.lb_metrics.pack(fill=tk.X, padx=5, pady=2)
        
        f_btn = ttk.Frame(f_config)
        f_btn.pack(fill=tk.X, pady=5)
        ttk.Button(f_btn, text="Phân tích Hồi quy Vét cạn", command=self.run_combinatorial_regression).pack(side=tk.LEFT, padx=5)
        ttk.Button(f_btn, text="Lưu Bảng Công Thức (CSV)", command=self.export_formulas).pack(side=tk.LEFT, padx=5)
        
        # Bảng hiển thị kết quả hàm tương quan
        cols_corr = ("Loại Hàm", "Tổ hợp Tiêu chí", "Phương trình", "R²", "RMSE")
        self.tv_corr = ttk.Treeview(f_right, columns=cols_corr, show="headings", height=12)
        for c in cols_corr: self.tv_corr.heading(c, text=c)
        self.tv_corr.column("Loại Hàm", width=80); self.tv_corr.column("Tổ hợp Tiêu chí", width=150)
        self.tv_corr.column("Phương trình", width=300); self.tv_corr.column("R²", width=60); self.tv_corr.column("RMSE", width=60)
        self.tv_corr.pack(fill=tk.BOTH, expand=True, pady=5)
        self.tv_corr.tag_configure('best', background='#d4edda', font=('Segoe UI', 9, 'bold')) # Highlight màu xanh lá

    # -----------------------------------------------------------------------
    # LOGIC: XỬ LÝ KHÁCH QUAN & CẬP NHẬT GIAO DIỆN
    # -----------------------------------------------------------------------
    def load_file(self):
        filepaths = filedialog.askopenfilenames(filetypes=[("Dữ liệu Matlab/CSV", "*.csv *.mat")])
        if not filepaths: return
        
        for filepath in filepaths:
            filename = os.path.basename(filepath)
            algo_name = filename.replace('.mat','').replace('.csv','').strip()
            
            base_name = algo_name; idx = 1
            while algo_name in self.datasets:
                algo_name = f"{base_name} ({idx})"; idx += 1
                
            try:
                if filepath.endswith('.csv'): df = pd.read_csv(filepath)
                else:
                    mat = sio.loadmat(filepath); df_dict = {}
                    df_dict['t'] = mat.get('t_sim', mat.get('t', np.arange(len(mat.get('Pf', [0]))))).flatten()
                    if 'Pf' in mat: df_dict['ax_ref'] = mat['Pf'][:, 0]; df_dict['ay_ref'] = mat['Pf'][:, 1]; df_dict['az_ref'] = mat['Pf'][:, 2]
                    if 'Pfsim' in mat: df_dict['ax_sim'] = mat['Pfsim'][:, 0]; df_dict['ay_sim'] = mat['Pfsim'][:, 1]; df_dict['az_sim'] = mat['Pfsim'][:, 2]
                    if 'Pw' in mat: df_dict['wx_ref'] = mat['Pw'][:, 0]; df_dict['wy_ref'] = mat['Pw'][:, 1]; df_dict['wz_ref'] = mat['Pw'][:, 2]
                    if 'Pwsim' in mat: df_dict['wx_sim'] = mat['Pwsim'][:, 0]; df_dict['wy_sim'] = mat['Pwsim'][:, 1]; df_dict['wz_sim'] = mat['Pwsim'][:, 2]
                    if 'Sd' in mat: df_dict['px_sim'] = mat['Sd'][:, 0]; df_dict['py_sim'] = mat['Sd'][:, 1]; df_dict['pz_sim'] = mat['Sd'][:, 2]
                    if 'Betasim' in mat: df_dict['roll_sim'] = mat['Betasim'][:, 0]; df_dict['pitch_sim'] = mat['Betasim'][:, 1]; df_dict['yaw_sim'] = mat['Betasim'][:, 2]
                    if 'Pasim' in mat: df_dict['pa_x_sim'] = mat['Pasim'][:, 0]; df_dict['pa_y_sim'] = mat['Pasim'][:, 1]; df_dict['pa_z_sim'] = mat['Pasim'][:, 2]
                    df = pd.DataFrame(df_dict)
                
                self.datasets[algo_name] = {'df': df, 'metrics': {}}
            except Exception as e:
                messagebox.showerror("Lỗi đọc file", f"Lỗi ở file {filename}: {str(e)}")
                continue
                
        self.recalculate_all()
        messagebox.showinfo("Thành công", f"Đã nạp xong {len(filepaths)} file.")

    def recalculate_all(self):
        f_mask = [val.get() for val in self.f_axis_vars]
        w_mask = [val.get() for val in self.w_axis_vars]
        cmode = self.centrifuge_var.get()
        
        for name, data in self.datasets.items():
            data['metrics'] = calculate_custom_dof_metrics(data['df'], f_mask, w_mask, cmode)
            
        self.update_results_table()
        self.plot_comparison()
        self.plot_adv_comparison()

    def update_results_table(self):
        for row in self.tv_results.get_children(): self.tv_results.delete(row)
        for name, data in self.datasets.items():
            m = data['metrics']
            vals = [name] + [f"{m[key]:.4f}" for key in self.metrics_list]
            self.tv_results.insert("", tk.END, values=vals)

    def clear_data(self):
        self.datasets.clear()
        self.update_results_table()
        self.ax.clear(); self.canvas.draw()
        self.ax_3d.clear(); self.ax_2d.clear(); self.canvas_adv.draw()
        self.ax_subj.clear(); self.canvas_subj.draw()
        self.lbl_result_subj.config(text="Mô hình Tối ưu: Trống")

    def export_results(self):
        if not self.datasets: return
        filepath = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if filepath:
            data_list = []
            for name, d in self.datasets.items():
                row = {'Thuật toán': name}; row.update(d['metrics'])
                data_list.append(row)
            pd.DataFrame(data_list).to_csv(filepath, index=False, encoding='utf-8-sig')
            messagebox.showinfo("Thành công", "Đã xuất bảng kết quả ra file CSV.")

    def export_single_metric_table(self):
        if not self.datasets: return
        metric_name = self.cb_metric.get()
        safe_name = metric_name.replace("/", "_").replace("\\", "_").replace(" ", "_")
        filepath = filedialog.asksaveasfilename(
            initialfile=f"{safe_name}_Results.csv", defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if filepath:
            data_list = []
            for name, d in self.datasets.items():
                data_list.append({'Thuật toán': name, metric_name: d['metrics'][metric_name]})
            pd.DataFrame(data_list).to_csv(filepath, index=False, encoding='utf-8-sig')
            messagebox.showinfo("Thành công", f"Đã xuất bảng {metric_name} ra file CSV.")

    def update_3d_view(self, elev, azim):
        if hasattr(self, 'ax_3d'):
            self.ax_3d.view_init(elev=elev, azim=azim)
            self.canvas_adv.draw_idle()

    # -----------------------------------------------------------------------
    # LOGIC: CHỦ QUAN, HỢP NHẤT VÀ HỒI QUY
    # -----------------------------------------------------------------------
    def get_objective_dataframe(self):
        if not self.datasets: return None
        obj_list = []
        for name, data in self.datasets.items():
            row = {'Algorithm': name}
            row.update(data['metrics'])
            obj_list.append(row)
        return pd.DataFrame(obj_list)

    def process_subjective_v2(self):
        filepath = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not filepath: return
        try:
            # === ĐOẠN CODE CẬP NHẬT CHỐNG LỖI JAGGED CSV ===
            df = None
            encodings = ['utf-8-sig', 'utf-8', 'latin1']
            delimiters = [',', ';', '\t']
            
            for enc in encodings:
                for delim in delimiters:
                    try:
                        # Bỏ sep=None, ép dùng engine='c' chuẩn để xử lý tốt các dấu phẩy thừa
                        temp_df = pd.read_csv(filepath, encoding=enc, sep=delim, on_bad_lines='skip')
                        if len(temp_df.columns) > 1:
                            df = temp_df
                            break
                    except Exception:
                        continue
                if df is not None:
                    break
                    
            if df is None:
                raise ValueError("Không thể đọc file CSV. Dữ liệu có thể bị hỏng cấu trúc cột.")
            # ===============================================

            # Xóa cột rỗng và đổi tên cột đầu tiên thành Algorithm
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            df.columns = ['Algorithm'] + list(df.columns[1:])
            df['Algorithm'] = df['Algorithm'].astype(str).str.strip()
            
            # Chuyển đổi dữ liệu sang dạng cột dài (Melt)
            df_long = pd.melt(df, id_vars=['Algorithm'], var_name='Subject', value_name='Score')
            df_long.dropna(subset=['Score'], inplace=True)
            
            # Xử lý dấu phẩy thập phân (nếu có)
            if df_long['Score'].dtype == object:
                df_long['Score'] = df_long['Score'].astype(str).str.replace(',', '.').astype(float)
            df_long['Raw_Score'] = df_long['Score'].astype(float)
            
            # Khử nhiễu OLS
            model = ols('Raw_Score ~ C(Algorithm) + C(Subject)', data=df_long).fit()
            df_long['Residuals'] = model.resid
            df_long['Std_Residuals'] = (df_long['Residuals'] - df_long['Residuals'].mean()) / df_long['Residuals'].std()
            
            outlier_mask = df_long['Std_Residuals'].abs() > 2.0
            df_long['Filtered_Score'] = df_long['Raw_Score']
            df_long.loc[outlier_mask, 'Filtered_Score'] = model.fittedvalues[outlier_mask]
            df_long['Note'] = np.where(outlier_mask, "Lọc nhiễu (Residual > 2σ)", "Giữ nguyên")
            
            # Xóa data cũ trên UI và điền data mới (Hiển thị Raw vs Filtered)
            for row in self.tv_data.get_children(): self.tv_data.delete(row)
            for _, r in df_long.iterrows():
                tag = 'changed' if r['Note'] != "Giữ nguyên" else ''
                self.tv_data.insert("", tk.END, values=(r['Subject'], r['Algorithm'], f"{r['Raw_Score']:.2f}", f"{r['Filtered_Score']:.2f}", r['Note']), tags=(tag,))
                
            # Tính Mean & Variance
            stats = df_long.groupby('Algorithm')['Filtered_Score'].agg(['mean', 'std']).reset_index()
            stats.rename(columns={'mean': 'Subjective_Mean_Score'}, inplace=True)
            self.df_subj = stats
            
            # Vẽ biểu đồ Mean & Variance (Error bars)
            self.ax_mean.clear()
            self.ax_mean.bar(stats['Algorithm'], stats['Subjective_Mean_Score'], yerr=stats['std'], capsize=5, color='#4c72b0', alpha=0.8, edgecolor='black')
            self.ax_mean.set_title("Điểm Trung bình & Phương sai (Sau khi lọc)")
            self.ax_mean.set_ylabel("Điểm Chủ quan")
            plt.setp(self.ax_mean.get_xticklabels(), rotation=15, ha="right")
            self.fig_mean.tight_layout()
            self.canvas_mean.draw()
            
            # Cập nhật danh sách tiêu chí Khách quan để User chọn
            # GHI CHÚ: Tương tự lỗi ghép tên thuật toán (MergeKey) ở bước trước
            if hasattr(self, 'df_obj') and self.df_obj is not None:
                self.df_subj['MergeKey'] = self.df_subj['Algorithm'].astype(str).str.extract(r'(\d+)')[0]
                self.df_obj['MergeKey'] = self.df_obj['Algorithm'].astype(str).str.extract(r'(\d+)')[0]
                self.df_subj['MergeKey'] = self.df_subj['MergeKey'].fillna(self.df_subj['Algorithm'].astype(str))
                self.df_obj['MergeKey'] = self.df_obj['MergeKey'].fillna(self.df_obj['Algorithm'].astype(str))

                self.df_merged = pd.merge(self.df_subj, self.df_obj, on='MergeKey', how='inner', suffixes=('_subj', '_obj'))
                if 'Algorithm_obj' in self.df_merged.columns:
                    self.df_merged['Algorithm'] = self.df_merged['Algorithm_obj']

                obj_cols = [c for c in self.df_merged.columns if c not in ['Algorithm', 'Algorithm_subj', 'Algorithm_obj', 'MergeKey', 'Subjective_Mean_Score', 'std']]
                
                self.lb_metrics.delete(0, tk.END)
                for c in obj_cols: self.lb_metrics.insert(tk.END, c)
                
            messagebox.showinfo("Hoàn tất", "Đã nạp, lọc nhiễu OLS và vẽ phổ thành công!")
        except Exception as e:
            messagebox.showerror("Lỗi", str(e))

    def run_regression(self):
        df_obj = self.get_objective_dataframe()
        if self.df_subj is None or df_obj is None:
            messagebox.showwarning("Cảnh báo", "Hãy nạp dữ liệu Khách quan (Tab 1) và Chủ quan (Tab 4) trước.")
            return
            
        # ==============================================================
        # BẢN VÁ LỖI TRÙNG TÊN THUẬT TOÁN (TÁCH SỐ ĐỂ MATCHING)
        # ==============================================================
        # Trích xuất chữ số từ tên thuật toán để làm khóa nối (MergeKey)
        # VD: "Data_1" -> "1", "Algo 2" -> "2", "1" -> "1"
        self.df_subj['MergeKey'] = self.df_subj['Algorithm'].astype(str).str.extract(r'(\d+)')[0]
        df_obj['MergeKey'] = df_obj['Algorithm'].astype(str).str.extract(r'(\d+)')[0]
        
        # Nếu có thuật toán không có số (VD: "CWA_Goc"), dùng tên gốc làm dự phòng
        self.df_subj['MergeKey'] = self.df_subj['MergeKey'].fillna(self.df_subj['Algorithm'].astype(str))
        df_obj['MergeKey'] = df_obj['MergeKey'].fillna(df_obj['Algorithm'].astype(str))

        # Hợp nhất dựa trên MergeKey thay vì Algorithm
        self.df_merged = pd.merge(self.df_subj, df_obj, on='MergeKey', how='inner', suffixes=('_subj', '_obj'))
        
        # Giữ lại tên file .MAT gốc để hiển thị trên đồ thị cho chuyên nghiệp
        if 'Algorithm_obj' in self.df_merged.columns:
            self.df_merged['Algorithm'] = self.df_merged['Algorithm_obj']
        # ==============================================================

        if self.df_merged.empty:
            messagebox.showerror("Lỗi ghép dữ liệu", "Không thể map dữ liệu CSV và MAT. Vui lòng kiểm tra lại số thứ tự thuật toán.")
            return

        x_col = self.cb_metric_subj.get()
        if not x_col or x_col not in self.df_merged.columns: return
            
        x = self.df_merged[x_col].values
        y = self.df_merged['Subjective_Mean_Score'].values
        
        models = {
            'Tuyến tính (Linear)': lambda x, a, b: a*x + b,
            'Bậc 2 (Quadratic)': lambda x, a, b, c: a*x**2 + b*x + c,
            'Bậc 3 (Cubic)': lambda x, a, b, c, d: a*x**3 + b*x**2 + c*x + d,
            'Hàm mũ (Exponential)': lambda x, a, b, c: a * np.exp(b * x) + c,
            'Logarit (Logarithmic)': lambda x, a, b: a * np.log(np.abs(x) + 1e-5) + b
        }
        
        best_name, best_func, best_popt, best_r2 = None, None, None, -np.inf
        
        for name, func in models.items():
            try:
                p0 = [1] * (func.__code__.co_argcount - 1)
                if name == 'Hàm mũ (Exponential)': p0 = [1, 0.1, 1]
                
                popt, _ = curve_fit(func, x, y, p0=p0, maxfev=10000)
                y_pred = func(x, *popt)
                
                ss_res = np.sum((y - y_pred)**2)
                ss_tot = np.sum((y - np.mean(y))**2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                if r2 > best_r2:
                    best_r2 = r2; best_name = name; best_func = func; best_popt = popt
            except: pass
            
        if best_name:
            self.lbl_result_subj.config(text=f"Mô hình Tối ưu: {best_name}  |  Độ khớp (R²): {best_r2:.4f}")
            self.ax_subj.clear()
            self.ax_subj.scatter(x, y, color='royalblue', s=100, label='Dữ liệu mô phỏng', zorder=5, edgecolors='black')
            
            # Tính toán khoảng margin mượt mà hơn cho đường dự đoán
            x_min, x_max = min(x), max(x)
            margin = 0.05 * abs(x_max - x_min) if x_max != x_min else 0.1
            x_smooth = np.linspace(x_min - margin, x_max + margin, 200)
            
            y_smooth = best_func(x_smooth, *best_popt)
            self.ax_subj.plot(x_smooth, y_smooth, color='crimson', linestyle='--', linewidth=2, label=f'Dự đoán ({best_name})')
            
            for i, txt in enumerate(self.df_merged['Algorithm']):
                self.ax_subj.annotate(txt, (x[i], y[i]), xytext=(5,5), textcoords='offset points', fontweight='bold')
                
            self.ax_subj.set_title(f'Tương quan: {x_col} vs Đánh giá Chủ quan (Subjective Score)', fontweight='bold')
            self.ax_subj.set_xlabel(x_col)
            self.ax_subj.set_ylabel('Điểm Chủ quan (1: Tốt -> 5: Tệ)')
            self.ax_subj.grid(True, linestyle='--', alpha=0.7)
            self.ax_subj.legend()
            self.fig_subj.tight_layout()
            self.canvas_subj.draw()
        else:
            messagebox.showerror("Lỗi", "Không thể nội suy hàm toán học nào cho bộ dữ liệu này.")

    def run_combinatorial_regression(self):
        if self.df_merged is None: return
        selected_indices = self.lb_metrics.curselection()
        if not selected_indices:
            messagebox.showwarning("Nhắc nhở", "Hãy chọn ít nhất 1 tiêu chí khách quan.")
            return
            
        selected_metrics = [self.lb_metrics.get(i) for i in selected_indices]
        y_true = self.df_merged['Subjective_Mean_Score'].values
        
        self.correlation_results = [] # Lưu trữ để xuất CSV sau này
        
        # Hàm hỗ trợ tính R2 và RMSE
        def calc_r2_rmse(y_t, y_p):
            ss_res = np.sum((y_t - y_p)**2)
            ss_tot = np.sum((y_t - np.mean(y_t))**2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            rmse = np.sqrt(mean_squared_error(y_t, y_p))
            return r2, rmse

        # 3a & 3c: Phân tích TƯƠNG QUAN ĐƠN (1 biến) với nhiều dạng hàm
        for col in selected_metrics:
            x = self.df_merged[col].values
            
            # 1. Tuyến tính (Linear): y = w1*x + w0
            popt, _ = curve_fit(lambda x, w1, w0: w1*x + w0, x, y_true)
            r2, rmse = calc_r2_rmse(y_true, popt[0]*x + popt[1])
            eq = f"{popt[0]:.4f}*{col} + {popt[1]:.4f}"
            self.correlation_results.append(("Tuyến tính", col, eq, r2, rmse))
            
            # 2. Bậc 2 (Quadratic): y = w2*x^2 + w1*x + w0
            popt, _ = curve_fit(lambda x, w2, w1, w0: w2*(x**2) + w1*x + w0, x, y_true)
            r2, rmse = calc_r2_rmse(y_true, popt[0]*(x**2) + popt[1]*x + popt[2])
            eq = f"{popt[0]:.4f}*{col}² + {popt[1]:.4f}*{col} + {popt[2]:.4f}"
            self.correlation_results.append(("Bậc 2", col, eq, r2, rmse))
            
            # 3. Hàm mũ (Exponential): y = a * exp(b*x) + c
            try:
                popt, _ = curve_fit(lambda x, a, b, c: a*np.exp(b*x) + c, x, y_true, p0=[1, 0.1, 1], maxfev=5000)
                r2, rmse = calc_r2_rmse(y_true, popt[0]*np.exp(popt[1]*x) + popt[2])
                eq = f"{popt[0]:.4f} * e^({popt[1]:.4f}*{col}) + {popt[2]:.4f}"
                self.correlation_results.append(("Hàm Mũ", col, eq, r2, rmse))
            except: pass

        # 3b: Phân tích TƯƠNG QUAN TỔ HỢP ĐA BIẾN (Tuyến tính bội: 2 -> N biến)
        if len(selected_metrics) >= 2:
            for r in range(2, len(selected_metrics) + 1):
                for combo in itertools.combinations(selected_metrics, r):
                    X_multi = self.df_merged[list(combo)].values
                    # Sử dụng numpy lstsq để tìm nghiệm hồi quy đa biến
                    X_design = np.column_stack([X_multi, np.ones(len(X_multi))])
                    weights, residuals, _, _ = np.linalg.lstsq(X_design, y_true, rcond=None)
                    
                    y_pred = X_design @ weights
                    r2, rmse = calc_r2_rmse(y_true, y_pred)
                    
                    # Tạo chuỗi phương trình
                    eq_parts = [f"{w:.4f}*{var}" for w, var in zip(weights[:-1], combo)]
                    eq = " + ".join(eq_parts) + f" + {weights[-1]:.4f}"
                    combo_name = " + ".join(combo)
                    
                    self.correlation_results.append((f"Tổ hợp {r} biến", combo_name, eq, r2, rmse))

        # Hiển thị lên Bảng, Sort theo R2 giảm dần
        self.correlation_results.sort(key=lambda x: x[3], reverse=True)
        for row in self.tv_corr.get_children(): self.tv_corr.delete(row)
        
        for res in self.correlation_results:
            tag = 'best' if res[3] >= 0.8 else '' # Highlight R2 >= 0.8
            self.tv_corr.insert("", tk.END, values=(res[0], res[1], res[2], f"{res[3]:.4f}", f"{res[4]:.4f}"), tags=(tag,))

    def refresh_metrics_list(self):
        """Cập nhật danh sách tiêu chí vào Listbox ở Tab 4"""
        if hasattr(self, 'datasets') and len(self.datasets) > 0:
            # Lấy key từ dict metrics của thuật toán đầu tiên
            first_algo = list(self.datasets.keys())[0]
            metrics = self.datasets[first_algo]['metrics'].keys()
            
            # Lọc chỉ lấy các tiêu chí phù hợp (bỏ qua những thứ không phải là số)
            filtered_metrics = [m for m in metrics if m not in ['Algorithm', 'Subjective_Mean_Score', 'MergeKey', 'std']]
            
            self.lb_metrics.delete(0, tk.END)
            for m in filtered_metrics:
                self.lb_metrics.insert(tk.END, m)
                
            # Cập nhật cả Combobox ở Tab 4 cho tương quan đơn
            self.cb_metric_subj['values'] = filtered_metrics

    def export_formulas(self):
        if not hasattr(self, 'correlation_results') or not self.correlation_results:
            messagebox.showinfo("Thông báo", "Chưa có kết quả tương quan để xuất.")
            return
            
        filepath = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if filepath:
            df_export = pd.DataFrame(self.correlation_results, columns=["Loại Hàm", "Tổ hợp Tiêu chí", "Phương trình", "R²", "RMSE"])
            df_export.to_csv(filepath, index=False, encoding='utf-8-sig')
            messagebox.showinfo("Thành công", f"Đã xuất các công thức toán học ra:\n{filepath}")

    # -----------------------------------------------------------------------
    # PLOTTING CHARTS (TAB 2, TAB 3)
    # -----------------------------------------------------------------------
    def plot_comparison(self):
        self.ax.clear()
        if not self.datasets:
            self.canvas.draw()
            return
            
        metric_name = self.cb_metric.get()
        names = list(self.datasets.keys())
        values = [d['metrics'][metric_name] for d in self.datasets.values()]
        
        unit = " (Không thứ nguyên)"
        if "Scale Err" in metric_name or "Shape Err" in metric_name:
            if "f" in metric_name: unit = " (m/s²)"
            elif "w" in metric_name: unit = " (rad/s)"
        elif "Score" in metric_name: unit = " (Điểm)"

        bars = self.ax.bar(names, values, color='#1f77b4', edgecolor='black', width=0.5)
        self.ax.set_ylabel(f"Giá trị{unit}", fontweight='bold')
        self.ax.set_title(f"Đánh giá & So sánh tiêu chí: {metric_name}", fontweight='bold', fontsize=14, color='#d83b01')
        self.ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        for bar in bars:
            yval = bar.get_height()
            self.ax.text(bar.get_x() + bar.get_width()/2, yval + (0.01 * max(values)), f'{yval:.4f}', ha='center', va='bottom', fontweight='bold')
            
        plt.setp(self.ax.get_xticklabels(), rotation=15, ha="right")
        self.fig.tight_layout()
        self.canvas.draw()

    def plot_adv_comparison(self):
        self.ax_3d.clear()
        self.ax_2d.clear()
        
        if not self.datasets:
            self.canvas_adv.draw()
            return
            
        plot_3d_type = self.cb_3d_type.get()
        x_lbl, y_lbl, z_lbl = "", "", ""
        
        for name, data in self.datasets.items():
            m = data['metrics']
            if plot_3d_type == "Shape Error Lực (Surge, Sway, Heave)":
                x, y, z = m['Shape Err fx (Surge)'], m['Shape Err fy (Sway)'], m['Shape Err fz (Heave)']
                x_lbl, y_lbl, z_lbl = "Surge Err (m/s²)", "Sway Err (m/s²)", "Heave Err (m/s²)"
            elif plot_3d_type == "Scale Error Lực (Surge, Sway, Heave)":
                x, y, z = m['Scale Err fx (Surge)'], m['Scale Err fy (Sway)'], m['Scale Err fz (Heave)']
                x_lbl, y_lbl, z_lbl = "Surge Err (m/s²)", "Sway Err (m/s²)", "Heave Err (m/s²)"
            elif plot_3d_type == "Shape Error Vận tốc góc (Roll, Pitch, Yaw)":
                x, y, z = m['Shape Err wx (Roll)'], m['Shape Err wy (Pitch)'], m['Shape Err wz (Yaw)']
                x_lbl, y_lbl, z_lbl = "Roll Err (rad/s)", "Pitch Err (rad/s)", "Yaw Err (rad/s)"
            elif plot_3d_type == "Scale Error Vận tốc góc (Roll, Pitch, Yaw)":
                x, y, z = m['Scale Err wx (Roll)'], m['Scale Err wy (Pitch)'], m['Scale Err wz (Yaw)']
                x_lbl, y_lbl, z_lbl = "Roll Err (rad/s)", "Pitch Err (rad/s)", "Yaw Err (rad/s)"
            
            self.ax_3d.scatter(x, y, z, s=150, label=name, edgecolors='black', depthshade=False)
            self.ax_3d.text(x, y, z, f" {name}", fontsize=9, fontweight='bold')
            
        self.ax_3d.set_xlabel(f"{x_lbl}\n(Càng về 0 càng tốt)", labelpad=10)
        self.ax_3d.set_ylabel(y_lbl, labelpad=10)
        self.ax_3d.set_zlabel(z_lbl, labelpad=10)
        self.ax_3d.set_title(f"Không gian Tọa độ 3D - {plot_3d_type.split('(')[0]}", fontweight='bold', color='#005a9e')
        
        plot_2d_type = self.cb_2d_type.get()
        for name, data in self.datasets.items():
            m = data['metrics']
            if plot_2d_type == "Pouliot (L1 vs L2)": x, y = m['Pouliot L1'], m['Pouliot L2']
            elif plot_2d_type == "Fischer (L1 vs L2)": x, y = m['Fischer L1'], m['Fischer L2']
            elif plot_2d_type == "Normalized (L1 vs L2)": x, y = m['Norm L1'], m['Norm L2']
                
            self.ax_2d.scatter(x, y, s=150, label=name, edgecolors='black')
            self.ax_2d.annotate(name, (x, y), xytext=(8,5), textcoords='offset points', fontweight='bold')

        self.ax_2d.set_xlabel("L1 Error (Sai số Tín hiệu)\n(Lý tưởng: Tiến về gốc tọa độ 0)", fontweight='bold')
        self.ax_2d.set_ylabel("L2 Error (Sai số Đạo hàm)", fontweight='bold')
        self.ax_2d.set_title(f"Tương quan {plot_2d_type}", fontweight='bold', color='#107C10')
        self.ax_2d.grid(True, linestyle='--', alpha=0.7)
        self.ax_2d.legend(loc="upper right")

        self.fig_adv.tight_layout(pad=3.0)
        self.canvas_adv.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = MCA6DOFEvaluatorApp(root)
    root.mainloop()