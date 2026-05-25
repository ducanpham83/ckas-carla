import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import scipy.io as sio
from scipy.signal import correlate
from scipy.optimize import curve_fit
import pingouin as pg
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import warnings
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
from itertools import combinations

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
        metrics = {'Scale_Err': 0, 'Shape_Err': 0, 'Total_Err': 0, 'Ang_Shape_Err': 0, 'False_Cue': 0, 'Phase_Delay': 0}
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
# GIAO DIỆN DESKTOP (TKINTER)
# ==========================================
class MotionFidelityApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Motion Fidelity Analyzer - Trạm phân tích Desktop V4.0")
        self.root.geometry("1200x850")
        self.root.minsize(1000, 700)
        
        self.df_subj = None
        self.df_obj = None
        self.df_merged = None
        
        style = ttk.Style()
        style.configure("Treeview.Heading", font=('Arial', 9, 'bold'))
        
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(expand=True, fill='both', padx=10, pady=10)
        
        self.tab1 = ttk.Frame(self.notebook)
        self.tab2 = ttk.Frame(self.notebook)
        self.tab3 = ttk.Frame(self.notebook)
        
        self.notebook.add(self.tab1, text=" 1: ICC & Dữ liệu Chủ quan ")
        self.notebook.add(self.tab2, text=" 2: Trích xuất Khách quan ")
        self.notebook.add(self.tab3, text=" 3: Bảng Hợp nhất & Hồi quy ")
        
        self.setup_tab1()
        self.setup_tab2()
        self.setup_tab3()

    # ---------------------------------------------------------
    # TAB 1: DỮ LIỆU CHỦ QUAN VÀ CHI TIẾT ICC
    # ---------------------------------------------------------
    def setup_tab1(self):
        frame_top = ttk.Frame(self.tab1)
        frame_top.pack(fill='x', padx=10, pady=10)
        
        ttk.Button(frame_top, text="Tải File CSV Đánh giá", command=self.process_subjective).pack(side='left', padx=10)
        
        self.log_tab1 = tk.Text(self.tab1, height=8, wrap='word', bg='#e6f2ff', font=('Consolas', 10))
        self.log_tab1.pack(fill='x', padx=10, pady=5)
        
        tree_frame = ttk.Frame(self.tab1)
        tree_frame.pack(expand=True, fill='both', padx=10, pady=5)
        
        cols = ('Algo', 'Subject', 'Old_Score', 'New_Score', 'Status')
        self.tree1 = ttk.Treeview(tree_frame, columns=cols, show='headings')
        for col, name in zip(cols, ['Thuật toán', 'Người dùng', 'Điểm Gốc', 'Điểm Sau Lọc', 'Trạng Thái']):
            self.tree1.heading(col, text=name)
            self.tree1.column(col, anchor='center')
            
        self.tree1.tag_configure('changed', background='#ffe6e6', foreground='#cc0000')
        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree1.yview)
        self.tree1.configure(yscroll=scrollbar.set)
        scrollbar.pack(side='right', fill='y')
        self.tree1.pack(side='left', expand=True, fill='both')

    def process_subjective(self):
        filepath = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv;*.xlsx")])
        if not filepath: return
        
        try:
            self.log_tab1.delete('1.0', tk.END)
            self.log_tab1.insert(tk.END, f"Đang xử lý: {os.path.basename(filepath)}...\n")
            self.root.update()
            
            df = None
            for enc in ['utf-8-sig', 'utf-8', 'latin1']:
                for delim in [',', ';', '\t']:
                    try:
                        temp_df = pd.read_csv(filepath, encoding=enc, sep=delim, on_bad_lines='skip')
                        if len(temp_df.columns) > 5:
                            df = temp_df; break
                    except: continue
                if df is not None: break
            
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            col_names = list(df.columns)
            col_names[0] = 'Algorithm'
            df.columns = col_names
            df['Algorithm'] = df['Algorithm'].astype(str).str.strip()
            
            df_long = pd.melt(df, id_vars=['Algorithm'], var_name='Subject', value_name='Score')
            df_long.dropna(subset=['Score'], inplace=True)
            if df_long['Score'].dtype == object:
                df_long['Score'] = df_long['Score'].astype(str).str.replace(',', '.').astype(float)
                
            # Khử nhiễu
            model = ols('Score ~ C(Algorithm) + C(Subject)', data=df_long).fit()
            df_long['Residuals'] = model.resid
            df_long['Fitted_Score'] = model.fittedvalues
            df_long['Std_Residuals'] = (df_long['Residuals'] - df_long['Residuals'].mean()) / df_long['Residuals'].std()
            
            outlier_mask = df_long['Std_Residuals'].abs() > 2.0
            
            df_long['Original_Score'] = df_long['Score']
            df_long['Status'] = 'Giữ nguyên'
            df_long.loc[outlier_mask, 'Score'] = df_long['Fitted_Score'][outlier_mask]
            df_long.loc[outlier_mask, 'Status'] = 'Đã nắn lại (Nhiễu)'
            
            # Tính ICC sau khi lọc
            icc_filtered = pg.intraclass_corr(data=df_long, targets='Algorithm', raters='Subject', ratings='Score')
            icc3k = icc_filtered[icc_filtered['Type'] == 'ICC3k'].iloc[0]
            
            # IN CHI TIẾT ICC VÀO LOG
            self.log_tab1.insert(tk.END, "--- THỐNG KÊ CHI TIẾT ĐỘ TIN CẬY (ICC 3,k) ---\n")
            self.log_tab1.insert(tk.END, f"• Hệ số ICC: {icc3k['ICC']:.4f}\n")
            self.log_tab1.insert(tk.END, f"• Khoảng tin cậy (CI 95%): {icc3k['CI95%']}\n")
            self.log_tab1.insert(tk.END, f"• Giá trị kiểm định F: {icc3k['F']:.3f} (Bậc tự do: df1={icc3k['df1']}, df2={icc3k['df2']})\n")
            self.log_tab1.insert(tk.END, f"• p-value: {icc3k['pval']:.3e} \n")
            
            # Cập nhật Treeview
            self.tree1.delete(*self.tree1.get_children())
            for _, row in df_long.iterrows():
                tag = ('changed',) if row['Status'] != 'Giữ nguyên' else ()
                self.tree1.insert('', 'end', values=(row['Algorithm'], row['Subject'], round(row['Original_Score'], 2), round(row['Score'], 2), row['Status']), tags=tag)
            
            # LƯU TRUNG BÌNH & PHƯƠNG SAI
            agg_scores = df_long.groupby('Algorithm')['Score'].agg(
                Subj_Mean='mean',
                Subj_Std='std'
            ).reset_index()
            
            self.df_subj = agg_scores
        except Exception as e: messagebox.showerror("Lỗi", str(e))

    # ---------------------------------------------------------
    # TAB 2: DỮ LIỆU KHÁCH QUAN (.MAT)
    # ---------------------------------------------------------
    def setup_tab2(self):
        frame_top = ttk.Frame(self.tab2)
        frame_top.pack(fill='x', padx=10, pady=10)
        ttk.Label(frame_top, text="Ngưỡng Vận tốc góc:").pack(side='left')
        self.ang_vel_var = tk.StringVar(value="3deg")
        ttk.Radiobutton(frame_top, text="3 deg/s", variable=self.ang_vel_var, value="3deg").pack(side='left')
        ttk.Radiobutton(frame_top, text="6 deg/s", variable=self.ang_vel_var, value="6deg").pack(side='left', padx=10)
        ttk.Button(frame_top, text="Tải Files .MAT (Nhiều file)", command=self.process_objective).pack(side='left', padx=20)
        self.log_tab2 = tk.Text(self.tab2, height=20, bg='#f4f4f4')
        self.log_tab2.pack(expand=True, fill='both', padx=10, pady=5)

    def process_objective(self):
        filepaths = filedialog.askopenfilenames(filetypes=[("MATLAB Files", "*.mat")])
        if not filepaths: return
        T_w = 3.0 * np.pi / 180.0 if self.ang_vel_var.get() == "3deg" else 6.0 * np.pi / 180.0
        T_f = 0.17
        try:
            self.log_tab2.delete('1.0', tk.END)
            self.root.update()
            obj_results = []
            for filepath in filepaths:
                algo_name = os.path.basename(filepath).replace('.mat', '').strip()
                analyzer = MATAnalyzer(filepath, T_f, T_w)
                metrics = analyzer.run_metrics()
                metrics['Algorithm'] = algo_name
                obj_results.append(metrics)
                self.log_tab2.insert(tk.END, f"Hoàn tất trích xuất: {algo_name}\n")
                self.root.update()
            self.df_obj = pd.DataFrame(obj_results)
            self.update_tab3_dropdown()
            self.log_tab2.insert(tk.END, "✅ BƯỚC 2 HOÀN TẤT!\n")
        except Exception as e: messagebox.showerror("Lỗi", str(e))

    # ---------------------------------------------------------
    # TAB 3: BẢNG HỢP NHẤT CHI TIẾT & HỒI QUY
    # ---------------------------------------------------------
    def setup_tab3(self):
       # ======================================
    # FRAME CHỨA NÚT
    # ======================================
        frame_top = ttk.Frame(self.tab3)
        frame_top.pack(fill='x', padx=10, pady=5)

        ttk.Label(frame_top, text="Mô hình:").pack(side='left')

        self.combo_model = ttk.Combobox(
            frame_top,
            state="readonly",
            width=18
        )

        self.combo_model['values'] = [
            'Auto (Tốt nhất)',
            'Tuyến tính',
            'Bậc 2',
            'Bậc 3',
            'Số mũ',
            'Logarit'
        ]

        self.combo_model.current(0)

        self.combo_model.pack(side='left', padx=5)

        ttk.Button(
            frame_top,
            text="Tương quan đơn",
            command=self.run_single_regression
        ).pack(side='left', padx=5)

        ttk.Button(
            frame_top,
            text="Tương quan tổ hợp",
            command=self.run_multi_regression
        ).pack(side='left', padx=5)

        ttk.Button(
            frame_top,
            text="Phân tích tất cả",
            command=self.run_all_combinations
        ).pack(side='left', padx=10)

        ttk.Button(
            frame_top,
            text="Xuất CSV",
            command=self.save_csv
        ).pack(side='right')

        # ======================================
        # LISTBOX OBJECTIVE
        # ======================================
        frame_select = ttk.LabelFrame(
            self.tab3,
            text="Chọn nhiều tiêu chí Objective"
        )

        frame_select.pack(fill='x', padx=10, pady=5)

        self.listbox_obj = tk.Listbox(
            frame_select,
            selectmode=tk.MULTIPLE,
            height=6,
            exportselection=False
        )

        self.listbox_obj.pack(
            fill='x',
            padx=5,
            pady=5
        )

        # ==============================
        # BẢNG DỮ LIỆU
        # ==============================
        table_frame = ttk.Frame(self.tab3)
        table_frame.pack(fill='x', padx=10, pady=5)

        self.tree3 = ttk.Treeview(table_frame, show='headings', height=8)

        scroll3 = ttk.Scrollbar(
            table_frame,
            orient=tk.VERTICAL,
            command=self.tree3.yview
        )

        self.tree3.configure(yscroll=scroll3.set)

        scroll3.pack(side='right', fill='y')
        self.tree3.pack(side='left', expand=True, fill='both')

        # ==============================
        # KẾT QUẢ
        # ==============================
        self.result_text = tk.Text(
            self.tab3,
            height=8,
            bg='#eef5ff',
            font=('Consolas', 10)
        )

        self.result_text.pack(fill='x', padx=10, pady=5)

        # ==============================
        # ĐỒ THỊ
        # ==============================
        self.fig_frame = ttk.Frame(self.tab3)
        self.fig_frame.pack(expand=True, fill='both', padx=10, pady=5)

        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 5))
        self.fig.tight_layout(pad=3.0)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.fig_frame)
        self.canvas.get_tk_widget().pack(expand=True, fill='both')
       
    def update_tab3_dropdown(self):
        if self.df_subj is not None and self.df_obj is not None:

            self.df_merged = pd.merge(
                self.df_subj,
                self.df_obj,
                on='Algorithm',
                how='inner'
            )

            self.tree3.delete(*self.tree3.get_children())

            cols = list(self.df_merged.columns)
            self.tree3['columns'] = cols

            for col in cols:
                self.tree3.heading(col, text=col)
                self.tree3.column(col, width=100, anchor='center')

            for _, row in self.df_merged.iterrows():
                vals = []

                for v in row:
                    if isinstance(v, (float, np.floating)):
                        vals.append(round(v, 4))
                    else:
                        vals.append(v)

                self.tree3.insert('', 'end', values=vals)

            obj_cols = [
                c for c in cols
                if c not in ['Algorithm', 'Subj_Mean', 'Subj_Std']
            ]

            
            self.listbox_obj.delete(0, tk.END)

            for col in obj_cols:
                self.listbox_obj.insert(tk.END, col)

            self.plot_variance_bar()

    def run_multi_regression(self):

        if self.df_merged is None:
            return

        selected_indices = self.listbox_obj.curselection()

        if len(selected_indices) == 0:
            messagebox.showwarning(
                "Thiếu dữ liệu",
                "Hãy chọn ít nhất 1 tiêu chí objective"
            )
            return

        selected_cols = [
            self.listbox_obj.get(i)
            for i in selected_indices
        ]

        df_clean = self.df_merged.dropna(
            subset=selected_cols + ['Subj_Mean']
        )

        X = df_clean[selected_cols].values
        y = df_clean['Subj_Mean'].values

        model = LinearRegression()
        model.fit(X, y)

        y_pred = model.predict(X)

        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))

        self.result_text.delete('1.0', tk.END)

        self.result_text.insert(
            tk.END,
            '===== HỒI QUY TỔ HỢP =====\n\n'
        )

        equation = 'Subj_Mean = '

        equation += f'{model.intercept_:.4f}'

        for coef, name in zip(model.coef_, selected_cols):
            equation += f' + ({coef:.4f})*{name}'

        self.result_text.insert(tk.END, equation + '\n\n')

        self.result_text.insert(tk.END, f'R²   = {r2:.4f}\n')
        self.result_text.insert(tk.END, f'RMSE = {rmse:.4f}\n\n')

        self.result_text.insert(tk.END, 'Mức ảnh hưởng:\n')

        impacts = sorted(
            zip(selected_cols, np.abs(model.coef_)),
            key=lambda x: x[1],
            reverse=True
        )

        for i, (name, val) in enumerate(impacts):
            self.result_text.insert(
                tk.END,
                f'{i+1}. {name:<20} : {val:.4f}\n'
            )

        # =========================
        # VẼ ĐỒ THỊ
        # =========================
        self.canvas.draw()

    def plot_variance_bar(self):
        self.ax1.clear()
        algos = self.df_merged['Algorithm'].values
        means = self.df_merged['Subj_Mean'].values
        stds = self.df_merged['Subj_Std'].values
        
        self.ax1.bar(algos, means, yerr=stds, capsize=5, color='skyblue', edgecolor='black', alpha=0.7)
        self.ax1.set_title("Điểm Trung Bình & Phương Sai Chủ Quan")
        self.ax1.set_ylabel("Điểm (1=Tốt, 5=Tệ)")
        self.ax1.set_ylim(0, 5)
        self.ax1.tick_params(axis='x', rotation=45)
        self.canvas.draw()

    def get_models_dict(self):
        return {
            'Tuyến tính': (lambda x, a, b: a*x + b, [1.0, 1.0]),
            'Bậc 2': (lambda x, a, b, c: a*x**2 + b*x + c, [1.0, 1.0, 1.0]),
            'Bậc 3': (lambda x, a, b, c, d: a*x**3 + b*x**2 + c*x + d, [1.0, 1.0, 1.0, 1.0]),
            'Số mũ': (lambda x, a, b, c: a * np.exp(b * x) + c, [1.0, 0.01, 1.0]),
            'Logarit': (lambda x, a, b: a * np.log(np.abs(x) + 1e-5) + b, [1.0, 1.0])
        }

    def run_single_regression(self):

        if self.df_merged is None:
            return

        selected_indices = self.listbox_obj.curselection()

        if len(selected_indices) == 0:
            messagebox.showwarning(
                "Thiếu dữ liệu",
                "Hãy chọn 1 objective"
            )
            return

        x_col = self.listbox_obj.get(selected_indices[0])
        model_req = self.combo_model.get()

        df_clean = self.df_merged.dropna(
            subset=[x_col, 'Subj_Mean']
        )

        x = np.array(df_clean[x_col].values, dtype=float)
        y = np.array(df_clean['Subj_Mean'].values, dtype=float)

        best_name, best_func, best_popt, best_r2 = self._fit_best(
            x,
            y,
            model_req
        )

        self.result_text.delete('1.0', tk.END)

        self.ax2.clear()

        if best_name:

            y_pred = best_func(x, *best_popt)

            rmse = np.sqrt(mean_squared_error(y, y_pred))

            self.result_text.insert(
                tk.END,
                '===== TƯƠNG QUAN ĐƠN =====\n\n'
            )

            self.result_text.insert(
                tk.END,
                f'Tiêu chí: {x_col}\n'
            )

            self.result_text.insert(
                tk.END,
                f'Mô hình tốt nhất: {best_name}\n'
            )

            self.result_text.insert(
                tk.END,
                f'R² = {best_r2:.4f}\n'
            )

            self.result_text.insert(
                tk.END,
                f'RMSE = {rmse:.4f}\n'
            )

            self.ax2.scatter(x, y, color='blue', s=70)

            xs = np.linspace(x.min(), x.max(), 200)
            ys = best_func(xs, *best_popt)
        self.canvas.draw()

    def _fit_best(self, x, y, model_req='Auto (Tốt nhất)'):
        best_n, best_f, best_p, best_r2 = None, None, None, -np.inf
        for name, (func, p0) in self.get_models_dict().items():
            if model_req != 'Auto (Tốt nhất)' and name != model_req: continue
            try:
                popt, _ = curve_fit(func, x, y, p0=p0, maxfev=50000)
                r2 = 1 - (np.sum((y - func(x, *popt))**2) / np.sum((y - np.mean(y))**2))
                if r2 > best_r2: best_r2, best_n, best_f, best_p = r2, name, func, popt
            except: pass
        return best_n, best_f, best_p, best_r2

    def run_all_combinations(self):

        if self.df_merged is None:
            return

        obj_cols = [
            c for c in self.df_merged.columns
            if c not in ['Algorithm', 'Subj_Mean', 'Subj_Std']
        ]

        y = self.df_merged['Subj_Mean'].values

        results = []

        for r in range(1, min(4, len(obj_cols)+1)):

            for combo in combinations(obj_cols, r):

                try:
                    X = self.df_merged[list(combo)].values

                    model = LinearRegression()
                    model.fit(X, y)

                    y_pred = model.predict(X)

                    r2 = r2_score(y, y_pred)
                    rmse = np.sqrt(mean_squared_error(y, y_pred))

                    results.append([
                        ', '.join(combo),
                        len(combo),
                        r2,
                        rmse
                    ])

                except:
                    pass

        results = sorted(results, key=lambda x: x[2], reverse=True)

        top = tk.Toplevel(self.root)

        top.title('Xếp hạng tổ hợp Objective')
        top.geometry('850x450')

        tree = ttk.Treeview(
            top,
            columns=('Combo', 'N', 'R2', 'RMSE'),
            show='headings'
        )

        tree.heading('Combo', text='Tổ hợp Objective')
        tree.heading('N', text='Số biến')
        tree.pack(expand=True, fill='both', padx=10, pady=10)

    def run_all_combinations(self):
        if self.df_merged is None: return
        obj_cols = [c for c in self.df_merged.columns if c not in ['Algorithm', 'Subj_Mean', 'Subj_Std']]
        y = np.array(self.df_merged['Subj_Mean'].values, dtype=float)
        
        results = []
        for col in obj_cols:
            x = np.array(self.df_merged[col].values, dtype=float)
            best_n, _, _, best_r2 = self._fit_best(x, y, 'Auto (Tốt nhất)')
            if best_n: results.append((col, best_n, best_r2))
            
        results.sort(key=lambda item: item[2], reverse=True) # Xếp theo R2 giảm dần
        
        # Tạo cửa sổ Popup hiển thị Bảng Xếp Hạng
        top = tk.Toplevel(self.root)
        top.title("Bảng Xếp Hạng Tiêu Chí Khách Quan")
        top.geometry("500x300")
        
        ttk.Label(top, text="Các tiêu chí ảnh hưởng mạnh nhất đến cảm nhận người dùng:", font=('Arial', 10, 'bold')).pack(pady=10)
        
        tree_top = ttk.Treeview(top, columns=('Rank', 'Criterion', 'Best Model', 'R2'), show='headings')
        tree_top.heading('Rank', text='Hạng'); tree_top.column('Rank', width=50, anchor='center')
        tree_top.heading('Criterion', text='Tiêu chí (Objective)'); tree_top.column('Criterion', width=150)
        tree_top.heading('Best Model', text='Hàm Tương Quan'); tree_top.column('Best Model', width=120, anchor='center')
        tree_top.heading('R2', text='Độ khớp R²'); tree_top.column('R2', width=100, anchor='center')
        
        for i, (col, model, r2) in enumerate(results):
            tree_top.insert('', 'end', values=(i+1, col, model, f"{r2:.4f}"))
            
        tree_top.pack(expand=True, fill='both', padx=10, pady=10)

    def save_csv(self):
        if self.df_merged is not None:
            filepath = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
            if filepath:
                self.df_merged.to_csv(filepath, index=False, encoding='utf-8-sig')
                messagebox.showinfo("Lưu file", "Đã lưu bảng dữ liệu hợp nhất thành công!")

if __name__ == "__main__":
    root = tk.Tk()
    app = MotionFidelityApp(root)
    root.mainloop()