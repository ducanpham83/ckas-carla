import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import scipy.io as sio
from scipy.signal import correlate
from scipy.optimize import curve_fit
import pingouin as pg
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import warnings
import os
import re

warnings.filterwarnings('ignore')

# ==========================================
# MODULE TÍNH TOÁN KHÁCH QUAN (MATLAB)
# ==========================================
class MATAnalyzer:
    def __init__(self, filepath, T_f=0.17, T_w=0.052):
        data = sio.loadmat(filepath)
        self.dt = np.squeeze(data['t_sim'])[1] - np.squeeze(data['t_sim'])[0] if 't_sim' in data else 0.0125
        
        self.Pf = self._shape(data.get('Pf', np.zeros((3, 1000))))
        self.Pfsim = self._shape(data.get('Pfsim', np.zeros((3, 1000))))
        self.Pw = self._shape(data.get('Pw', np.zeros((3, 1000))))
        self.Pwsim = self._shape(data.get('Pwsim', np.zeros((3, 1000))))
        self.T_f, self.T_w = T_f, T_w

    def _shape(self, a):
        a = np.squeeze(a)
        if a.ndim == 2 and a.shape[0] == 3 and a.shape[1] > 3: return a.T
        if a.ndim == 1: return np.pad(a[:, np.newaxis], ((0,0), (0,2)), 'constant')
        return a

    def run_metrics(self):
        metrics = {'Scale_Err': 0.0, 'Shape_Err': 0.0, 'Total_Err': 0.0, 'Ang_Shape_Err': 0.0, 'False_Cue': 0.0, 'Phase_Delay': 0.0}
        for i in range(3):
            tgt_f, sim_f = self.Pf[:, i], self.Pfsim[:, i]
            K_f = np.sum(tgt_f * sim_f) / np.sum(tgt_f**2) if np.sum(tgt_f**2) > 1e-6 else 0
            
            metrics['Scale_Err'] += np.sqrt(np.mean(((K_f * tgt_f - tgt_f) / self.T_f)**2)) / 3
            metrics['Shape_Err'] += np.sqrt(np.mean(((sim_f - K_f * tgt_f) / self.T_f)**2)) / 3
            metrics['Total_Err'] += np.sqrt(np.mean(((sim_f - tgt_f) / self.T_f)**2)) / 3
            
            t_norm, s_norm = tgt_f - np.mean(tgt_f), sim_f - np.mean(sim_f)
            if np.std(t_norm) > 1e-4 and np.std(s_norm) > 1e-4:
                corr = correlate(s_norm, t_norm, mode='full')
                metrics['Phase_Delay'] += (np.arange(-len(tgt_f)+1, len(tgt_f))[np.argmax(corr)] * self.dt) / 3
            
            tgt_w, sim_w = self.Pw[:, i], self.Pwsim[:, i]
            K_w = np.sum(tgt_w * sim_w) / np.sum(tgt_w**2) if np.sum(tgt_w**2) > 1e-6 else 0
            metrics['Ang_Shape_Err'] += np.sqrt(np.mean(((sim_w - K_w * tgt_w) / self.T_w)**2)) / 3
            
            fc_mask = (np.abs(tgt_w) < 1e-2) & (np.abs(sim_w) > self.T_w)
            metrics['False_Cue'] += (np.sum(np.abs(sim_w[fc_mask]) - self.T_w) / np.sum(fc_mask)) / 3 if np.sum(fc_mask) > 0 else 0
        return metrics

# ==========================================
# GIAO DIỆN DESKTOP (TKINTER) V6.1
# ==========================================
class MotionFidelityApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Motion Fidelity Analyzer - Trạm phân tích Desktop V6.1")
        self.root.geometry("1400x850")
        
        self.df_subj = None
        self.df_obj = None
        self.df_merged = None
        self.algo_order_list = []
        
        self.current_single_res = None
        self.current_multi_res = None
        
        style = ttk.Style()
        style.configure("Treeview.Heading", font=('Arial', 9, 'bold'))
        
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(expand=True, fill='both', padx=10, pady=10)
        
        self.tab1 = ttk.Frame(self.notebook)
        self.tab2 = ttk.Frame(self.notebook)
        self.tab3 = ttk.Frame(self.notebook)
        
        self.notebook.add(self.tab1, text=" 1: Dữ liệu Chủ quan (ICC) ")
        self.notebook.add(self.tab2, text=" 2: Trích xuất Khách quan ")
        self.notebook.add(self.tab3, text=" 3: Phân tích Tương Quan (Đơn & Đa Biến) ")
        
        self.setup_tab1()
        self.setup_tab2()
        self.setup_tab3()

    def setup_tab1(self):
        frame_top = ttk.Frame(self.tab1)
        frame_top.pack(fill='x', padx=10, pady=10)
        ttk.Button(frame_top, text="Tải File CSV Đánh giá", command=self.process_subjective).pack(side='left', padx=10)
        
        # Mở rộng Text Box để chứa trọn bảng ICC
        self.log_tab1 = tk.Text(self.tab1, height=12, bg='#e6f2ff', font=('Consolas', 10))
        self.log_tab1.pack(fill='x', padx=10, pady=5)
        
        self.tree1 = ttk.Treeview(self.tab1, columns=('Algo', 'Subj', 'Old', 'New', 'Status'), show='headings')
        for col, name in zip(self.tree1['columns'], ['Thuật toán', 'Người dùng', 'Điểm Gốc', 'Điểm Sau Lọc', 'Trạng Thái']):
            self.tree1.heading(col, text=name); self.tree1.column(col, anchor='center')
        self.tree1.tag_configure('changed', background='#ffe6e6', foreground='#cc0000')
        self.tree1.pack(expand=True, fill='both', padx=10, pady=5)

    def process_subjective(self):
        filepath = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv;*.xlsx")])
        if not filepath: return
        try:
            self.log_tab1.delete('1.0', tk.END)
            df = None
            for enc in ['utf-8-sig', 'utf-8', 'latin1']:
                for delim in [',', ';', '\t']:
                    try:
                        temp_df = pd.read_csv(filepath, encoding=enc, sep=delim, on_bad_lines='skip')
                        if len(temp_df.columns) > 5: df = temp_df; break
                    except: continue
                if df is not None: break
            
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            df.columns = ['Algorithm'] + list(df.columns[1:])
            df['Algorithm'] = df['Algorithm'].astype(str).str.strip()
            
            self.algo_order_list = df['Algorithm'].unique().tolist()
            
            df_long = pd.melt(df, id_vars=['Algorithm'], var_name='Subject', value_name='Score').dropna()
            if df_long['Score'].dtype == object: df_long['Score'] = df_long['Score'].astype(str).str.replace(',', '.').astype(float)
                
            model = ols('Score ~ C(Algorithm) + C(Subject)', data=df_long).fit()
            df_long['Std_Residuals'] = (model.resid - model.resid.mean()) / model.resid.std()
            outlier_mask = df_long['Std_Residuals'].abs() > 2.0
            
            df_long['Original'] = df_long['Score']
            df_long['Status'] = 'Giữ nguyên'
            df_long.loc[outlier_mask, 'Score'] = model.fittedvalues[outlier_mask]
            df_long.loc[outlier_mask, 'Status'] = 'Đã nắn (Nhiễu)'
            
            # --- TÍNH VÀ HIỂN THỊ CHI TIẾT ICC ---
            icc_filtered = pg.intraclass_corr(data=df_long, targets='Algorithm', raters='Subject', ratings='Score')
            
            self.log_tab1.insert(tk.END, f"Đã nắn lại {outlier_mask.sum()} điểm đánh giá dị biệt. Dưới đây là Bảng ICC chi tiết sau khi lọc:\n")
            self.log_tab1.insert(tk.END, "-"*80 + "\n")
            # Chỉ hiển thị các cột quan trọng cho gọn
            icc_display = icc_filtered[['Type', 'ICC', 'F', 'df1', 'df2', 'pval', 'CI95%']]
            self.log_tab1.insert(tk.END, icc_display.to_string(index=False) + "\n")
            self.log_tab1.insert(tk.END, "-"*80 + "\n")
            self.log_tab1.insert(tk.END, "(* Ghi chú: ICC3k là chỉ số quan trọng nhất cho bài toán đánh giá đồng thuận Motion Cueing)\n")
            
            self.tree1.delete(*self.tree1.get_children())
            for _, row in df_long.iterrows():
                self.tree1.insert('', 'end', values=(row['Algorithm'], row['Subject'], round(row['Original'],2), round(row['Score'],2), row['Status']), tags=('changed',) if row['Status']!='Giữ nguyên' else ())
            
            self.df_subj = df_long.groupby('Algorithm', sort=False)['Score'].agg(Subj_Mean='mean', Subj_Std='std').reset_index()
            
        except Exception as e: messagebox.showerror("Lỗi", str(e))

    def setup_tab2(self):
        frame_top = ttk.Frame(self.tab2)
        frame_top.pack(fill='x', padx=10, pady=10)
        ttk.Label(frame_top, text="Ngưỡng Threshold:").pack(side='left')
        
        # --- BỔ SUNG LỰA CHỌN 6-DOF ---
        self.ang_vel_var = tk.StringVar(value="3deg")
        ttk.Radiobutton(frame_top, text="3 deg/s", variable=self.ang_vel_var, value="3deg").pack(side='left', padx=5)
        ttk.Radiobutton(frame_top, text="6 deg/s", variable=self.ang_vel_var, value="6deg").pack(side='left', padx=5)
        ttk.Radiobutton(frame_top, text="6-DOF (Tiêu chuẩn)", variable=self.ang_vel_var, value="6dof").pack(side='left', padx=5)
        
        ttk.Button(frame_top, text="Tải Files .MAT (Nhiều file)", command=self.process_objective).pack(side='left', padx=20)
        self.log_tab2 = tk.Text(self.tab2, bg='#f4f4f4')
        self.log_tab2.pack(expand=True, fill='both', padx=10, pady=5)

    def process_objective(self):
        if not self.algo_order_list:
            messagebox.showwarning("Yêu cầu", "Vui lòng thực hiện BƯỚC 1 (Tải CSV) trước!")
            return
            
        filepaths = list(filedialog.askopenfilenames(filetypes=[("MATLAB Files", "*.mat")]))
        if not filepaths: return
        
        def natural_sort_key(s):
            return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', os.path.basename(s))]
        filepaths.sort(key=natural_sort_key)
        
        # Thiết lập Threshold
        if self.ang_vel_var.get() == "3deg":
            T_w = 3.0 * np.pi / 180.0; T_f = 0.17
        elif self.ang_vel_var.get() == "6deg":
            T_w = 6.0 * np.pi / 180.0; T_f = 0.17
        else: # 6-DOF
            T_w = 0.1 # rad/s (~5.7 deg/s - Chuẩn hóa thường thấy trên hệ 6-DOF)
            T_f = 0.17
            
        try:
            self.log_tab2.delete('1.0', tk.END)
            self.log_tab2.insert(tk.END, f"Cấu hình Threshold: T_f = {T_f}, T_w = {T_w:.4f} rad/s\n\n")
            obj_results = []
            
            for i, fp in enumerate(filepaths):
                original_filename = os.path.basename(fp)
                mapped_name = self.algo_order_list[i] if i < len(self.algo_order_list) else original_filename.replace('.mat', '')
                
                metrics = MATAnalyzer(fp, T_f, T_w).run_metrics()
                metrics['Algorithm'] = mapped_name
                obj_results.append(metrics)
                
                self.log_tab2.insert(tk.END, f"Đã trích xuất: [{mapped_name}] từ file {original_filename}\n")
                self.root.update()
                
            self.df_obj = pd.DataFrame(obj_results)
            self.update_tab3()
            self.log_tab2.insert(tk.END, "\n✅ BƯỚC 2 HOÀN TẤT!\n")
        except Exception as e: messagebox.showerror("Lỗi", str(e))

    def setup_tab3(self):
        frame_table = ttk.LabelFrame(self.tab3, text=" 1. Bảng Dữ Liệu Hợp Nhất Tổng Thể ")
        frame_table.pack(fill='x', padx=10, pady=5)
        self.tree_merged = ttk.Treeview(frame_table, show='headings', height=5)
        scroll = ttk.Scrollbar(frame_table, orient=tk.VERTICAL, command=self.tree_merged.yview)
        self.tree_merged.configure(yscroll=scroll.set)
        scroll.pack(side='right', fill='y'); self.tree_merged.pack(side='left', expand=True, fill='both', padx=5, pady=5)
        ttk.Button(self.tab3, text="Xuất Bảng Tổng Hợp", command=self.save_csv).pack(anchor='e', padx=10)

        frame_split = ttk.Frame(self.tab3)
        frame_split.pack(expand=True, fill='both', padx=10, pady=5)
        
        # --- LEFT PANE (SINGLE) ---
        frame_single = ttk.LabelFrame(frame_split, text=" 2a. Tương Quan Đơn (1 Biến) ")
        frame_single.pack(side='left', expand=True, fill='both', padx=(0, 5))
        ctl_single = ttk.Frame(frame_single)
        ctl_single.pack(fill='x', pady=5, padx=5)
        self.combo_obj = ttk.Combobox(ctl_single, state="readonly", width=12)
        self.combo_obj.pack(side='left', padx=2)
        self.combo_model = ttk.Combobox(ctl_single, state="readonly", width=10, values=['Auto', 'Linear', 'Quad', 'Cubic', 'Exp', 'Log'])
        self.combo_model.current(0); self.combo_model.pack(side='left', padx=2)
        ttk.Button(ctl_single, text="Chạy", width=6, command=self.run_single).pack(side='left', padx=2)
        ttk.Button(ctl_single, text="📊 Bảng", width=8, command=self.show_single_table).pack(side='right', padx=2)
        
        self.lbl_single = ttk.Label(frame_single, text="", font=('Arial', 9, 'bold'), foreground='blue')
        self.lbl_single.pack(anchor='w', padx=5)
        self.fig_single, self.ax_single = plt.subplots(figsize=(4, 3))
        self.canvas_single = FigureCanvasTkAgg(self.fig_single, master=frame_single)
        self.canvas_single.get_tk_widget().pack(expand=True, fill='both')

        # --- RIGHT PANE (MULTI) ---
        frame_multi = ttk.LabelFrame(frame_split, text=" 2b. Hồi Quy Tổ Hợp Đa Biến (Chọn >2) ")
        frame_multi.pack(side='right', expand=True, fill='both', padx=(5, 0))
        ctl_multi = ttk.Frame(frame_multi)
        ctl_multi.pack(fill='x', pady=5, padx=5)
        
        # MENU LỰA CHỌN LOẠI HỒI QUY ĐA BIẾN
        ttk.Label(ctl_multi, text="Loại hàm:").pack(side='left')
        self.combo_multi_model = ttk.Combobox(ctl_multi, state="readonly", width=12, values=['Tuyến tính', 'Log-Tuyến tính', 'Số mũ'])
        self.combo_multi_model.current(0)
        self.combo_multi_model.pack(side='left', padx=(0, 5))
        
        self.listbox_multi = tk.Listbox(ctl_multi, selectmode=tk.MULTIPLE, height=3, exportselection=0)
        self.listbox_multi.pack(side='left', padx=5, fill='x', expand=True)
        ttk.Button(ctl_multi, text="Chạy Tổ Hợp", command=self.run_multi).pack(side='left', padx=2)
        ttk.Button(ctl_multi, text="📊 Bảng", command=self.show_multi_table).pack(side='left', padx=2)
        
        # Mở rộng Text hiển thị cả VIF và P-value
        self.text_multi = tk.Text(frame_multi, height=7, bg='#f9f9f9', font=('Consolas', 9))
        self.text_multi.pack(fill='x', padx=5, pady=2)
        self.fig_multi, self.ax_multi = plt.subplots(figsize=(4, 3))
        self.canvas_multi = FigureCanvasTkAgg(self.fig_multi, master=frame_multi)
        self.canvas_multi.get_tk_widget().pack(expand=True, fill='both')

    def update_tab3(self):
        if self.df_subj is not None and self.df_obj is not None:
            self.df_merged = pd.merge(self.df_subj, self.df_obj, on='Algorithm', how='inner')
            self.tree_merged.delete(*self.tree_merged.get_children())
            cols = list(self.df_merged.columns)
            self.tree_merged['columns'] = cols
            for col in cols:
                self.tree_merged.heading(col, text=col)
                self.tree_merged.column(col, width=70, anchor='center')
            for _, row in self.df_merged.iterrows():
                vals = [row['Algorithm']] + [round(x, 4) for x in row[1:]]
                self.tree_merged.insert('', 'end', values=vals)
                
            obj_cols = [c for c in cols if c not in ['Algorithm', 'Subj_Mean', 'Subj_Std']]
            self.combo_obj['values'] = obj_cols
            if obj_cols: self.combo_obj.current(0)
            
            self.listbox_multi.delete(0, tk.END)
            for col in obj_cols: self.listbox_multi.insert(tk.END, col)

    def run_single(self):
        if self.df_merged is None or self.df_merged.empty: return
        x_col = self.combo_obj.get()
        df_c = self.df_merged.dropna(subset=[x_col, 'Subj_Mean'])
        
        if len(df_c) < 3: return
            
        x, y = np.array(df_c[x_col].values, dtype=float), np.array(df_c['Subj_Mean'].values, dtype=float)
        
        models = {
            'Linear': (lambda x, a, b: a*x + b, [1, 1]),
            'Quad': (lambda x, a, b, c: a*x**2 + b*x + c, [1, 1, 1]),
            'Cubic': (lambda x, a, b, c, d: a*x**3 + b*x**2 + c*x + d, [1, 1, 1, 1]),
            'Exp': (lambda x, a, b, c: a * np.exp(b * x) + c, [1, 0.01, 1]),
            'Log': (lambda x, a, b: a * np.log(np.abs(x) + 1e-5) + b, [1, 1])
        }
        
        best_n, best_f, best_p, best_r2 = None, None, None, -np.inf
        req = self.combo_model.get()
        for name, (func, p0) in models.items():
            if req != 'Auto' and name != req: continue
            try:
                popt, _ = curve_fit(func, x, y, p0=p0, maxfev=50000)
                r2 = 1 - (np.sum((y - func(x, *popt))**2) / np.sum((y - np.mean(y))**2))
                if r2 > best_r2: best_r2, best_n, best_f, best_p = r2, name, func, popt
            except: pass
            
        self.ax_single.clear()
        if best_n:
            self.lbl_single.config(text=f"Mô hình: {best_n} | R² = {best_r2:.4f}")
            self.ax_single.scatter(x, y, color='blue')
            xs = np.linspace(x.min()-0.05, x.max()+0.05, 100)
            self.ax_single.plot(xs, best_f(xs, *best_p), 'r--')
            for i, txt in enumerate(df_c['Algorithm']): self.ax_single.annotate(txt, (x[i], y[i]))
            self.ax_single.set_title(f'{x_col} vs Đánh giá')
            self.ax_single.set_xlabel(x_col); self.ax_single.set_ylabel('Subj Mean')
            
            df_res = pd.DataFrame({'Algorithm': df_c['Algorithm'], x_col: x, 'Subj_Thuc_Te': y, 'Subj_Du_Doan': best_f(x, *best_p)})
            df_res['Sai_So_Error'] = df_res['Subj_Du_Doan'] - df_res['Subj_Thuc_Te']
            self.current_single_res = df_res
        self.canvas_single.draw()

    # --- NÂNG CẤP: CHẠY ĐA BIẾN VỚI VIF VÀ P-VALUE ---
    def run_multi(self):
        if self.df_merged is None or self.df_merged.empty: return
        selected_indices = self.listbox_multi.curselection()
        if len(selected_indices) < 2: return
            
        selected_vars = [self.listbox_multi.get(i) for i in selected_indices]
        df_c = self.df_merged.dropna(subset=selected_vars + ['Subj_Mean'])
        
        if len(df_c) <= len(selected_vars):
            messagebox.showerror("Lỗi", "Số biến được chọn vượt quá số lượng thuật toán hợp lệ!")
            return
            
        multi_model_type = self.combo_multi_model.get()
        
        X_raw = df_c[selected_vars].astype(float)
        y_raw = df_c['Subj_Mean'].astype(float)
        
        # CHUẨN BỊ TOÁN HỌC
        if multi_model_type == 'Log-Tuyến tính':
            X_transformed = np.log(np.abs(X_raw) + 1e-5)
            X_fit = sm.add_constant(X_transformed)
            y_fit = y_raw
        elif multi_model_type == 'Số mũ':
            X_fit = sm.add_constant(X_raw)
            y_fit = np.log(y_raw)
        else: # Tuyến tính
            X_fit = sm.add_constant(X_raw)
            y_fit = y_raw

        self.text_multi.delete('1.0', tk.END)

        # TÍNH VIF (Đa cộng tuyến)
        vif_data = pd.DataFrame()
        vif_data["Tiêu chí"] = X_fit.columns
        vif_data["VIF"] = [variance_inflation_factor(X_fit.values, i) for i in range(X_fit.shape[1])]
        vif_data = vif_data[vif_data["Tiêu chí"] != 'const']
        
        high_vif = vif_data[vif_data["VIF"] > 10]
        if not high_vif.empty:
            self.text_multi.insert(tk.END, "⚠️ CẢNH BÁO ĐA CỘNG TUYẾN (VIF > 10): Các biến bị trùng lặp.\n")
            for _, row in high_vif.iterrows():
                self.text_multi.insert(tk.END, f" -> {row['Tiêu chí']}: VIF={row['VIF']:.2f}\n")
            self.text_multi.insert(tk.END, "-"*40 + "\n")

        try:
            model = sm.OLS(y_fit, X_fit).fit()
            
            # Tính Y_dự_đoán thật (FIX TÊN BIẾN y_raw)
            if multi_model_type == 'Số mũ':
                y_pred = np.exp(model.predict(X_fit))
            else:
                y_pred = model.predict(X_fit)
                
            # Tính R2 thực tế
            r2_real = 1 - (np.sum((y_raw - y_pred)**2) / np.sum((y_raw - np.mean(y_raw))**2))
            
            # Xuất chuỗi Phương trình
            if multi_model_type == 'Log-Tuyến tính':
                terms = [f"({model.params[v]:.2f})*ln({v})" for v in selected_vars]
                eq_str = "Y = " + " + ".join(terms) + f" + ({model.params['const']:.2f})"
            elif multi_model_type == 'Số mũ':
                terms = [f"({model.params[v]:.2f})*{v}" for v in selected_vars]
                eq_str = "Y = exp(" + " + ".join(terms) + f" + {model.params['const']:.2f})"
            else:
                terms = [f"({model.params[v]:.2f})*{v}" for v in selected_vars]
                eq_str = "Y = " + " + ".join(terms) + f" + ({model.params['const']:.2f})"
                
            self.text_multi.insert(tk.END, f"{eq_str}\n")
            self.text_multi.insert(tk.END, f"Độ khớp thực tế (R²) = {r2_real:.4f}\n")
            self.text_multi.insert(tk.END, "-"*40 + "\n")
            self.text_multi.insert(tk.END, "P-VALUE & VIF (P-value < 0.05 là đạt):\n")
            
            # TRÍCH XUẤT P-VALUE
            for var in selected_vars:
                pval = model.pvalues[var]
                vif_val = vif_data.loc[vif_data['Tiêu chí'] == var, 'VIF'].values[0]
                sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else "(Bác bỏ)"
                self.text_multi.insert(tk.END, f" • {var:<12} | P={pval:.4f} {sig:<5} | VIF={vif_val:.1f}\n")
            
            pval_const = model.pvalues['const']
            sig_c = "***" if pval_const < 0.01 else "**" if pval_const < 0.05 else "*" if pval_const < 0.1 else ""
            self.text_multi.insert(tk.END, f" • {'Hằng số (C)':<12} | P={pval_const:.4f} {sig_c:<5} | (N/A)\n")
            
            # Vẽ đồ thị Dự đoán vs Thực tế (FIX TÊN BIẾN)
            self.ax_multi.clear()
            self.ax_multi.scatter(y_pred, y_raw, color='green', s=60)
            if len(y_raw) > 0 and len(y_pred) > 0:
                min_v, max_v = min(y_raw.min(), y_pred.min())-0.2, max(y_raw.max(), y_pred.max())+0.2
                self.ax_multi.plot([min_v, max_v], [min_v, max_v], 'k--', alpha=0.5, label='Hoàn hảo')
            for i, txt in enumerate(df_c['Algorithm']):
                self.ax_multi.annotate(txt, (y_pred.iloc[i], y_raw.iloc[i]), xytext=(4,4), textcoords='offset points')
            self.ax_multi.set_title('Sai số Tổ hợp: Dự đoán vs Thực tế')
            self.ax_multi.set_xlabel('Điểm Dự Đoán'); self.ax_multi.set_ylabel('Điểm Thực Tế')
            self.canvas_multi.draw()
            
            df_res = df_c[['Algorithm'] + selected_vars].copy()
            df_res['Subj_Thuc_Te'] = y_raw
            df_res['Subj_Du_Doan'] = y_pred
            df_res['Sai_So_Error'] = y_pred - y_raw
            self.current_multi_res = df_res
        except Exception as e: messagebox.showerror("Lỗi", str(e))

    def show_single_table(self):
        if self.current_single_res is None: return
        self._create_popup_table("Chi Tiết Tương Quan Đơn", self.current_single_res)

    def show_multi_table(self):
        if self.current_multi_res is None: return
        self._create_popup_table("Chi Tiết Tương Quan Đa Biến", self.current_multi_res)

    def _create_popup_table(self, title, df):
        top = tk.Toplevel(self.root)
        top.title(title); top.geometry("850x350")
        ttk.Button(top, text="Xuất Bảng Này (CSV)", command=lambda: self._export_sub_table(df)).pack(anchor='e', padx=10, pady=5)
        tree_frame = ttk.Frame(top)
        tree_frame.pack(expand=True, fill='both', padx=10, pady=5)
        cols = list(df.columns)
        tree = ttk.Treeview(tree_frame, columns=cols, show='headings')
        for col in cols:
            tree.heading(col, text=col)
            tree.column(col, width=100 if col == 'Algorithm' else 120, anchor='center')
        for _, row in df.iterrows():
            vals = [row['Algorithm']] + [round(x, 4) if isinstance(x, (float, np.float64)) else x for x in row[1:]]
            tree.insert('', 'end', values=vals)
        scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscroll=scroll.set)
        scroll.pack(side='right', fill='y'); tree.pack(side='left', expand=True, fill='both')

    def _export_sub_table(self, df):
        filepath = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
        if filepath:
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            messagebox.showinfo("Xong", "Đã lưu bảng thành CSV.")

    def save_csv(self):
        if self.df_merged is not None:
            filepath = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
            if filepath:
                self.df_merged.to_csv(filepath, index=False, encoding='utf-8-sig')
                messagebox.showinfo("Xong", "Đã xuất bảng tổng hợp.")

if __name__ == "__main__":
    root = tk.Tk()
    app = MotionFidelityApp(root)
    root.mainloop()