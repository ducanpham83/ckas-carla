import json
import math
import random
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from queue import PriorityQueue
import pandas as pd
import numpy as np

try:
    import carla
    import pygame
    HAS_CARLA = True
except ImportError:
    HAS_CARLA = False

# ==============================================================================
# -- THUẬT TOÁN TÌM ĐƯỜNG NGOẠI TUYẾN (ĐÃ FIX LỖI ĐỒ THỊ) ----------------------
# ==============================================================================
class OfflinePathPlanner:
    def __init__(self, map_data):
        self.topology = map_data['topology']
        self.nodes = []
        self.graph = {}
        self._build_robust_graph()
        
    def _get_node_id(self, pt):
        for i, n in enumerate(self.nodes):
            if math.hypot(n[0]-pt[0], n[1]-pt[1]) < 0.5:
                return i
        self.nodes.append(pt)
        self.graph[len(self.nodes)-1] = []
        return len(self.nodes)-1

    def _build_robust_graph(self):
        for path in self.topology:
            if len(path) < 2: continue
            id_start = self._get_node_id(path[0])
            id_end = self._get_node_id(path[-1])
            dist = sum([math.hypot(path[i][0]-path[i+1][0], path[i][1]-path[i+1][1]) for i in range(len(path)-1)])
            self.graph[id_start].append((dist, id_end, path))

    def _get_closest_node_id(self, x, y):
        min_d = float('inf'); best_id = -1
        for i, n in enumerate(self.nodes):
            d = math.hypot(n[0]-x, n[1]-y)
            if d < min_d: min_d = d; best_id = i
        return best_id

    def run_routing(self, start_pt, end_pt, algo="A*"):
        if algo == "Đường thẳng (Straight Line)": return [start_pt, end_pt]
        
        start_id = self._get_closest_node_id(start_pt['x'], start_pt['y'])
        end_id = self._get_closest_node_id(end_pt['x'], end_pt['y'])
        
        if start_id == -1 or end_id == -1: return None

        pq = PriorityQueue()
        pq.put((0, start_id, [start_pt])) 
        visited = {}

        while not pq.empty():
            cost, curr_id, path = pq.get()
            if curr_id in visited and visited[curr_id] <= cost: continue
            visited[curr_id] = cost
            
            n_curr = self.nodes[curr_id]
            n_end = self.nodes[end_id]
            
            if math.hypot(n_curr[0]-n_end[0], n_curr[1]-n_end[1]) < 2.0 or curr_id == end_id:
                path.append(end_pt)
                return path
                
            for edge_cost, next_id, edge_path in self.graph.get(curr_id, []):
                formatted_edge = [{'x': p[0], 'y': p[1], 'z': p[2]} for p in edge_path[1:]]
                n_next = self.nodes[next_id]
                heuristic = math.hypot(n_next[0]-n_end[0], n_next[1]-n_end[1]) if algo == "A*" else 0
                pq.put((cost + edge_cost + heuristic, next_id, path + formatted_edge))
                
        return None 

    def analyze_geometry(self, path):
        """Phân tích hình học Vĩ mô (Macro-Geometry) để khử nhiễu tuyệt đối"""
        if len(path) < 5: return "straight"
        
        # 1. SUB-SAMPLING (Lọc nhiễu vi mô): Lấy mẫu các điểm cách nhau tối thiểu 2 mét
        sampled = [path[0]]
        for pt in path[1:]:
            if math.hypot(pt['x']-sampled[-1]['x'], pt['y']-sampled[-1]['y']) >= 2.0:
                sampled.append(pt)
        if math.hypot(path[-1]['x']-sampled[-1]['x'], path[-1]['y']-sampled[-1]['y']) >= 1.0:
            sampled.append(path[-1])
            
        if len(sampled) < 3: return "straight"
        
        # 2. XÁC ĐỊNH TRỤC CHÍNH (Base Chord) từ điểm đầu đến điểm cuối
        base_dx = sampled[-1]['x'] - sampled[0]['x']
        base_dy = sampled[-1]['y'] - sampled[0]['y']
        base_yaw = math.atan2(base_dy, base_dx)
        
        # 3. TÌM ĐỘ LỆCH GÓC TỐI ĐA (Max Heading Deviation)
        max_dev = 0.0
        for i in range(len(sampled)-1):
            dx = sampled[i+1]['x'] - sampled[i]['x']
            dy = sampled[i+1]['y'] - sampled[i]['y']
            yaw = math.atan2(dy, dx)
            
            # Tính độ lệch góc chuẩn hóa trong khoảng [-pi, pi]
            diff = abs((yaw - base_yaw + math.pi) % (2 * math.pi) - math.pi)
            if diff > max_dev:
                max_dev = diff
                
        max_dev_deg = math.degrees(max_dev)
        
        # Nếu có bất kỳ đoạn nào bẻ lái lệch quá 12 độ so với trục chính -> Khúc cua (Curve)
        # Đường thẳng dẫu có nhiễu cũng hiếm khi lệch quá 5 độ
        return "curve" if max_dev_deg > 12.0 else "straight"

# ==============================================================================
# -- GIAO DIỆN CHÍNH 
# ==============================================================================
class OfflineMissionPlanner:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("ADAS & MCA Master Planner (Fixed Graph Version)")
        self.root.geometry("1650x950")
        
        try:
            with open("carla_offline_maps.json", "r", encoding='utf-8') as f:
                self.all_maps_data = json.load(f)
        except Exception:
            messagebox.showerror("Lỗi", "Không tìm thấy file 'carla_offline_maps.json'."); self.root.destroy(); return
            
        self.map_name = "Town04"
        self.route_segments = [] 
        
        self.ctrl_mode = tk.StringVar(value="speed")
        self.target_value = tk.DoubleVar(value=50.0)
        self.algo_var = tk.StringVar(value="A*")
        
        self.is_route_finalized = False
        self.show_map = True # BỔ SUNG DÒNG NÀY VÀO ĐÂY
        self.markers_plot = []
        self.bg_c, self.fg_c, self.grid_c = '#FFFFFF', '#000000', '#DDDDDD'
        
        self.last_x = None; self.last_y = None
        
        self._build_gui()
        self.load_offline_map(self.map_name)

    def load_offline_map(self, town_name):
        if town_name not in self.all_maps_data: return
        self.map_data = self.all_maps_data[town_name]
        self.spawn_pts = self.map_data['spawn_points']
        self.planner = OfflinePathPlanner(self.map_data)
        self.map_name = town_name
        self.route_segments = []
        self.is_route_finalized = False
        self._draw_initial_map()

    def _build_gui(self):
        l_frame = ttk.Frame(self.root, width=520); l_frame.pack(side="left", fill="y", padx=10, pady=5)
        
        ttk.Label(l_frame, text="1. BẢN ĐỒ", font=('Arial', 10, 'bold')).pack(anchor="w")
        map_f = ttk.Frame(l_frame); map_f.pack(fill="x", pady=2)
        self.map_combo = ttk.Combobox(map_f, values=list(self.all_maps_data.keys()), width=12)
        self.map_combo.set(self.map_name); self.map_combo.pack(side="left", padx=2)
        ttk.Button(map_f, text="⚡ ĐỔI", command=lambda: self.load_offline_map(self.map_combo.get())).pack(side="left", padx=5)
         # --- [CHÈN THÊM VÀO ĐÂY] CÔNG CỤ QUAN SÁT VÀ SƠ ĐỒ ---
        tool_f = ttk.Frame(l_frame)
        tool_f.pack(fill="x", pady=2)
        ttk.Button(tool_f, text="🏠 Toàn cảnh", command=self.zoom_home).pack(side="left", expand=True, padx=2)
        
        # NÚT MỚI THÊM VÀO:
        ttk.Button(tool_f, text="👁️ Hiện/Ẩn Bản Đồ", command=self.toggle_map).pack(side="left", expand=True, padx=2)
        
        self.map_style = "vector" # Biến trạng thái mặc định

        def toggle_map_style():
            self.map_style = "realistic" if self.map_style == "vector" else "vector"
            self.update_viz()
            
        ttk.Button(tool_f, text="🖼️ Đổi Kiểu Nền (Vector / Pygame)", command=toggle_map_style).pack(side="left", expand=True, padx=2)
        # Nút Toggle chế độ Zoom quét bằng chuột trái
        self.btn_zoom = ttk.Button(tool_f, text="🔍 Bật Zoom Quét (Kéo chuột)", command=self.toggle_zoom)
        self.btn_zoom.pack(side="left", expand=True, padx=2)
        
        ttk.Button(l_frame, text="🕸️ XEM SƠ ĐỒ LIÊN KẾT ĐỘNG HỌC (L/C)", style="Accent.TButton", command=self.show_route_schematic).pack(fill="x", pady=5)
        # ---------------------------------------------------
        ttk.Separator(l_frame, orient='horizontal').pack(fill='x', pady=5)
        
        ttk.Label(l_frame, text="2. LẬP LỊCH QUỸ ĐẠO", font=('Arial', 10, 'bold')).pack(anchor="w")
        self.notebook = ttk.Notebook(l_frame); self.notebook.pack(fill="x", pady=5)
        
        self.tab_manual = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_manual, text="🖱️ CHỈNH TAY")
        mode_f = ttk.Frame(self.tab_manual); mode_f.pack(fill="x", pady=5)
        ttk.Radiobutton(mode_f, text="Vận tốc (km/h)", variable=self.ctrl_mode, value="speed").pack(side="left", padx=5)
        ttk.Radiobutton(mode_f, text="Gia tốc (m/s²)", variable=self.ctrl_mode, value="accel").pack(side="right", padx=5)
        val_f = ttk.Frame(self.tab_manual); val_f.pack(fill="x", pady=5)
        ttk.Label(val_f, text="Giá trị:").pack(side="left", padx=5)
        ttk.Entry(val_f, textvariable=self.target_value, width=10).pack(side="right", padx=5)
        algo_f = ttk.Frame(self.tab_manual); algo_f.pack(fill="x", pady=5)
        ttk.Label(algo_f, text="Thuật toán:").pack(side="left", padx=5)
        ttk.Combobox(algo_f, textvariable=self.algo_var, values=["Dijkstra", "A*", "Đường thẳng (Straight Line)"], width=15).pack(side="right", padx=5)

        self.tab_auto = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_auto, text="🤖 TỰ ĐỘNG TẠO MẪU")
        self.scenarios = [
            "2a. Thẳng (V_max, A_max, Jerk_max)",
            "2b. Cua góc V đều (< Vmax), A_ngang_max",
            "2c. Cua góc bình thường (V_max, A_max)",
            "2d. Phanh thẳng (Giảm về 0 trong T giây)",
            "2e. Chuyển làn / Zigzag (V_max, A_max)",
            "2f. Phối hợp tự do (Chuỗi)"
        ]
        self.scen_combo = ttk.Combobox(self.tab_auto, values=self.scenarios, width=45)
        self.scen_combo.current(0); self.scen_combo.pack(pady=5, padx=5)
        self.scen_combo.bind("<<ComboboxSelected>>", self.on_scenario_change)
        
        self.dyn_frame = ttk.Frame(self.tab_auto)
        self.dyn_frame.pack(fill="x", padx=5, pady=5)
        self.dyn_inputs = {}
        self.on_scenario_change() 
        
        ttk.Button(self.tab_auto, text="💡 TÌM & CHÈN LỘ TRÌNH VÀO MAP", style="Accent.TButton", command=self.generate_auto_scenario).pack(fill="x", pady=5, padx=5)
       
        ttk.Separator(l_frame, orient='horizontal').pack(fill='x', pady=5)
        
        ttk.Label(l_frame, text="3. TRÌNH TỰ ĐÃ CHỌN", font=('Arial', 10, 'bold')).pack(anchor="w")
        self.tree = ttk.Treeview(l_frame, columns=("seg", "type", "val", "dist"), show="headings", height=5)
        self.tree.heading("seg", text="Đoạn"); self.tree.heading("type", text="Loại"); self.tree.heading("val", text="Thông số"); self.tree.heading("dist", text="S(m)")
        self.tree.column("seg", width=40); self.tree.column("type", width=90); self.tree.column("val", width=120); self.tree.column("dist", width=60)
        self.tree.pack(fill="x", pady=2)
        
        btn_f1 = ttk.Frame(l_frame); btn_f1.pack(fill="x", pady=2)
        ttk.Button(btn_f1, text="↩️ Xóa đoạn cuối", command=self.undo_last).pack(side="left", expand=True, padx=2)
        ttk.Button(btn_f1, text="🧹 Xóa toàn bộ", command=self.clear_all).pack(side="right", expand=True, padx=2)
        ttk.Button(l_frame, text="📍 CHỐT ĐIỂM CUỐI LÀM ĐÍCH (GOAL)", style="Accent.TButton", command=self.finalize_route).pack(fill="x", pady=5)
        
        ttk.Separator(l_frame, orient='horizontal').pack(fill='x', pady=5)
        
        ttk.Label(l_frame, text="4. TRÍCH XUẤT & ĐỒ THỊ", font=('Arial', 10, 'bold'), foreground="blue").pack(anchor="w")
        io_f = ttk.Frame(l_frame); io_f.pack(fill="x", pady=2)
        ttk.Button(io_f, text="💾 Lưu SP List", command=self.save_spawn_list).pack(side="left", expand=True, padx=2)
        ttk.Button(io_f, text="📂 Nạp Lộ trình", command=self.import_csv).pack(side="right", expand=True, padx=2)
        # NÚT MỚI: XUẤT ĐỒNG BỘ 2 PHẦN MỀM
        ttk.Button(l_frame, text="📤 XUẤT DỮ LIỆU ĐỂ CHẠY MÔ PHỎNG (JSON + CSV)", 
                   command=self.export_to_simulation_manager, 
                   style="Accent.TButton").pack(fill="x", pady=5)
        # NÚT MỚI: LƯU ẢNH VECTOR
        ttk.Button(io_f, text="🖼️ Lưu Bản Đồ (SVG)", command=self.export_vector_map).pack(side="right", expand=True, padx=2)
        ttk.Button(l_frame, text="📈 XUẤT TỌA ĐỘ ĐỘNG HỌC (X,Y,v,a,jerk,t)", command=self.export_trajectory_time, style="Accent.TButton").pack(fill="x", pady=5)
        ttk.Button(l_frame, text="📊 VẼ ĐỒ THỊ TỪ FILE CSV", command=self.plot_csv_data).pack(fill="x", pady=5)

        self.map_frame = ttk.Frame(self.root); self.map_frame.pack(side="right", fill="both", expand=True)
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.patch.set_facecolor(self.bg_c); self.ax.set_facecolor(self.bg_c)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.map_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        # --- BỔ SUNG KHỞI TẠO TOOLBAR ĐỂ SỬ DỤNG ZOOM/HOME ---
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.map_frame)
        self.toolbar.update()
        self.toolbar.pack_forget() # Ẩn thanh toolbar mặc định đi cho giao diện gọn gàng
        # --------------------------------------------------------
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_drag)

    def on_scroll(self, event):
        base_scale = 1.2; ax = event.inaxes
        if not ax: return
        cur_x, cur_y = ax.get_xlim(), ax.get_ylim()
        sf = 1/base_scale if event.button == 'up' else base_scale
        nw, nh = (cur_x[1]-cur_x[0])*sf, (cur_y[1]-cur_y[0])*sf
        relx, rely = (cur_x[1]-event.xdata)/(cur_x[1]-cur_x[0]), (cur_y[1]-event.ydata)/(cur_y[1]-cur_y[0])
        ax.set_xlim([event.xdata-nw*(1-relx), event.xdata+nw*relx])
        ax.set_ylim([event.ydata-nh*(1-rely), event.ydata+nh*rely])
        self.canvas.draw_idle()

    def on_press(self, event):
        if event.button == 2: 
            self.last_x, self.last_y = event.xdata, event.ydata
        elif event.button == 1: 
            if self.notebook.index("current") != 0 or self.is_route_finalized or not event.inaxes: return
            
            distances = [math.hypot(x - event.xdata, y - event.ydata) for x, y in zip(self.xs, self.ys)]
            min_idx = int(np.argmin(distances))
            if distances[min_idx] > 30.0: return 
            
            if not self.route_segments:
                self.route_segments.append({'end_idx': min_idx, 'type': 'speed', 'val': 0.0, 'path': [self.spawn_pts[min_idx]], 'dist': 0.0, 'params': {}})
            else:
                p1, p2 = self.spawn_pts[self.route_segments[-1]['end_idx']], self.spawn_pts[min_idx]
                path = self.planner.run_routing(p1, p2, self.algo_var.get())
                
                if path is None:
                    messagebox.showwarning("Cảnh báo", "Không có làn đường hợp lệ nối đến điểm này (Nút ngược chiều). Vui lòng chọn điểm khác!")
                    return
                
                d = sum([math.hypot(path[i]['x']-path[i+1]['x'], path[i]['y']-path[i+1]['y']) for i in range(len(path)-1)])
                self.route_segments.append({'start_idx': self.route_segments[-1]['end_idx'], 'end_idx': min_idx, 'type': self.ctrl_mode.get(), 'val': self.target_value.get(), 'path': path, 'dist': d, 'params': {}})
            self.update_viz()

    def on_drag(self, event):
        if event.button == 2 and event.xdata and self.last_x:
            dx = event.xdata - self.last_x; dy = event.ydata - self.last_y
            xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
            self.ax.set_xlim(xlim[0]-dx, xlim[1]-dx)
            self.ax.set_ylim(ylim[0]-dy, ylim[1]-dy)
            self.canvas.draw_idle()

    def _draw_initial_map(self):
        self.xs = np.array([sp['x'] for sp in self.spawn_pts])
        self.ys = np.array([-sp['y'] for sp in self.spawn_pts])
        self.update_viz()

    def on_scenario_change(self, event=None):
        for widget in self.dyn_frame.winfo_children(): widget.destroy()
        self.dyn_inputs.clear()
        idx = self.scen_combo.current()
        
        def add_entry(row, col, label, default_val, key):
            ttk.Label(self.dyn_frame, text=label).grid(row=row, column=col, padx=2, pady=2, sticky="e")
            e = ttk.Entry(self.dyn_frame, width=6); e.insert(0, default_val); e.grid(row=row, column=col+1, padx=2, pady=2)
            self.dyn_inputs[key] = e

        if idx == 0: 
            add_entry(0, 0, "V_max:", "90", "v"); add_entry(0, 2, "A_max:", "5", "a"); add_entry(0, 4, "Jerk:", "2", "j")
        elif idx in [1, 2]: 
            add_entry(0, 0, "V_cua:", "40", "v"); add_entry(0, 2, "A_ngang:", "3", "a_lat")
        elif idx == 3: 
            add_entry(0, 0, "V_init:", "80", "v"); add_entry(0, 2, "Time(s):", "4", "t")
        elif idx == 4: 
            add_entry(0, 0, "V_max:", "60", "v"); add_entry(0, 2, "A_max:", "4", "a")
        elif idx == 5: 
            ttk.Label(self.dyn_frame, text="Nhập chuỗi (VD: 2a-2b-2d):").grid(row=0, column=0)
            e = ttk.Entry(self.dyn_frame, width=20); e.insert(0, "2a-2b-2d"); e.grid(row=0, column=1)
            self.dyn_inputs["seq"] = e

    def generate_auto_scenario(self):
        idx = self.scen_combo.current()
        self.root.config(cursor="wait"); self.root.update()
        
        params = {k: v.get() for k, v in self.dyn_inputs.items()}
        
        if not self.route_segments:
            base_idx = random.randint(0, len(self.spawn_pts)-1)
            init_seg = [{'end_idx': base_idx, 'type': 'speed', 'val': 0.0, 'path': [self.spawn_pts[base_idx]], 'dist': 0.0, 'params': {}}]
        else:
            base_idx = self.route_segments[-1]['end_idx']
            init_seg = [] 
        
        proposals = []
        attempts = 0
        
        while len(proposals) < 5 and attempts < 300:
            attempts += 1
            target_shape = "straight" if idx in [0, 3] else "curve"
            
            cands = [i for i in range(len(self.spawn_pts)) if 60 < math.hypot(self.xs[base_idx]-self.xs[i], self.ys[base_idx]-self.ys[i]) < 250]
            if not cands: continue
            
            next_idx = random.choice(cands)
            if any(p['next_idx'] == next_idx for p in proposals): continue
            
            path = self.planner.run_routing(self.spawn_pts[base_idx], self.spawn_pts[next_idx], "A*")
            if path is None: continue
            
            # ---------------------------------------------------------
            # BỘ LỌC ĐƯỜNG VÒNG (DETOUR FILTER) - CHỐNG LẶP NÚT GIAO
            # ---------------------------------------------------------
            d = sum([math.hypot(path[k]['x']-path[k+1]['x'], path[k]['y']-path[k+1]['y']) for k in range(len(path)-1)])
            crow_dist = math.hypot(self.xs[next_idx] - self.xs[base_idx], self.ys[next_idx] - self.ys[base_idx])
            
            # Nếu quãng đường thực tế dài hơn 1.8 lần đường chim bay -> Đang đi lòng vòng bùng binh -> BỎ QUA!
            if d > crow_dist * 1.8:
                continue
            # ---------------------------------------------------------
            
            if self.planner.analyze_geometry(path) == target_shape or idx == 5:
                scen_name = self.scenarios[idx].split(".")[0]
                new_seg = {
                    'start_idx': base_idx, 'end_idx': next_idx, 
                    'type': scen_name, 'val': params.get('v', 0), 
                    'path': path, 'dist': d, 'params': params
                }
                proposals.append({
                    'next_idx': next_idx,
                    'segments_to_add': init_seg + [new_seg],
                    'total_dist': d,
                    'desc': f"SP {base_idx} ➔ SP {next_idx} ({'Thẳng' if target_shape=='straight' else 'Cua'})"
                })

        self.root.config(cursor="")
        
        if proposals:
            proposals.sort(key=lambda x: x['total_dist'], reverse=True)
            self.show_proposal_dialog(proposals)
        else: 
            messagebox.showwarning("Lỗi", "Không tìm thấy đoạn phù hợp (Các đoạn có thể bị quá ngoằn ngoèo). Thầy thử chọn bằng tay nhé.")

    def show_proposal_dialog(self, proposals):
        top = tk.Toplevel(self.root)
        top.title("Đề xuất Tuyến đường (Sắp xếp theo chiều dài)")
        top.geometry("600x300")
        ttk.Label(top, text="Chọn 1 Option phù hợp:", font=('Arial', 10, 'bold')).pack(pady=10)
        
        tree = ttk.Treeview(top, columns=("opt", "desc", "dist"), show="headings", height=6)
        tree.heading("opt", text="Xếp hạng"); tree.heading("desc", text="Tuyến đường"); tree.heading("dist", text="Chiều dài (m)")
        tree.column("opt", width=80, anchor="center"); tree.column("desc", width=250, anchor="center"); tree.column("dist", width=120, anchor="center")
        tree.pack(fill="both", expand=True, padx=15)
        
        for i, p in enumerate(proposals): tree.insert("", "end", values=(f"Lựa chọn {i+1}", p['desc'], f"{p['total_dist']:.1f} m"))
            
        def on_select():
            selected = tree.selection()
            if not selected: return messagebox.showwarning("Cảnh báo", "Vui lòng chọn 1 tuyến đường!", parent=top)
            chosen = proposals[tree.index(selected[0])]
            self.route_segments.extend(chosen['segments_to_add'])
            self.update_viz(); top.destroy() 
            
        btn_f = ttk.Frame(top); btn_f.pack(pady=10)
        ttk.Button(btn_f, text="✔️ XÁC NHẬN CHỌN", command=on_select).pack(side="left", padx=10)
        ttk.Button(btn_f, text="❌ Hủy bỏ", command=top.destroy).pack(side="right", padx=10)

    def update_viz(self):
        # 1. CLEAR SẠCH MỌI THỨ TRÊN BẢN ĐỒ
        self.ax.clear()
        
        # 2. VẼ LẠI BẢN ĐỒ (Nếu đang bật chế độ Hiện Map)
        if self.show_map:
            # KIỂM TRA CHẾ ĐỘ HIỂN THỊ ĐỒ HỌA
            if getattr(self, 'map_style', 'vector') == "realistic":
                try:
                    import json
                    # Đọc thông số tọa độ thực của bản đồ từ file JSON
                    bbox_file = f"cache/no_rendering_mode/{self.map_name}_bbox.json"
                    with open(bbox_file, "r") as f:
                        bbox = json.load(f)
                    
                    m_x = bbox["min_x"]
                    m_y = bbox["min_y"]
                    m_w = bbox["width"]
                    
                    # Tính toán Extent. Matplotlib dùng thứ tự: [left, right, bottom, top]
                    # Do Thầy đang lật trục Y (py = -y), nên Top = -min_y, Bottom = -(min_y + width)
                    extent_box = [m_x, m_x + m_w, -m_y - m_w, -m_y]
                    
                    # Đọc và lót ảnh nền
                    img = plt.imread(f"cache/no_rendering_mode/{self.map_name}.png")
                    self.ax.imshow(img, extent=extent_box, alpha=0.85, zorder=1)
                    
                except Exception as e:
                    messagebox.showwarning("Thiếu Bản Đồ Gốc", f"Không tìm thấy ảnh hoặc dữ liệu tọa độ của {self.map_name}.\nThầy hãy chạy 'no_rendering_mode.py --map {self.map_name}' một lần để tạo ảnh trước nhé!")
                    self.map_style = "vector" # Tự động lùi về bản đồ Vector nếu lỗi
            
            # NẾU LÀ CHẾ ĐỘ VECTOR (Như cũ)
            if getattr(self, 'map_style', 'vector') == "vector":
                line_c = '#B0B5BA' if self.bg_c in ['#FFFFFF', '#D0D4D8'] else '#555555'
                qx, qy, qdx, qdy, q_c = [], [], [], [], []
                
                for path in self.map_data['topology']:
                    px, py = [pt[0] for pt in path], [-pt[1] for pt in path]
                    self.ax.plot(px, py, color=line_c, linewidth=2.0, alpha=0.8, clip_on=True, zorder=1) 
                    
                    if len(path) > 1:
                        p1, p2 = path[0], path[-1]
                        dx, dy = p2[0]-p1[0], -(p2[1]-p1[1])
                        d = math.hypot(dx,dy)
                        if d > 1.0: 
                            qx.append(p1[0]); qy.append(-p1[1])
                            qdx.append(dx/d); qdy.append(dy/d)
                            q_c.append(math.degrees(math.atan2(dy,dx)))
                        
                if qx: self.ax.quiver(qx, qy, qdx, qdy, q_c, cmap='hsv', scale=50, width=0.004, headwidth=4, alpha=0.5, clip_on=True, zorder=2)

            self.ax.scatter(self.xs, self.ys, c='#3498DB', s=60, edgecolors='black', zorder=5, clip_on=True)
            for i, (x, y) in enumerate(zip(self.xs, self.ys)): 
                self.ax.text(x, y+3, f"SP:{i}", fontsize=9, color='white', fontweight='bold', ha='center', bbox=dict(facecolor='#E74C3C', alpha=0.7, pad=1), clip_on=True)
        
        self.ax.axis('equal')
        self.ax.grid(True, linestyle='-', alpha=0.4, color=self.grid_c)
        
        # 3. LÀM SẠCH BẢNG TREEVIEW
        for r in self.tree.get_children(): self.tree.delete(r)

        if not self.route_segments: 
            self.canvas.draw_idle(); return
        
        # 4. VẼ LẠI QUỸ ĐẠO VÀ MŨI TÊN
        cmap_s = cm.get_cmap('YlOrRd'); norm_s = mcolors.Normalize(vmin=0, vmax=100)

        for i, seg in enumerate(self.route_segments):
            idx = seg['end_idx']
            if i == 0:
                self.ax.scatter(self.xs[idx], self.ys[idx], c='#2ECC71', s=200, marker='s', edgecolors='black', zorder=12, clip_on=True) 
                self.tree.insert("", "end", values=("START", "-", "-", "-"))
            else:
                if seg['type'] in ['speed', 'accel']:
                    c = cmap_s(norm_s(float(seg['val'])))
                    lbl = f"{seg['type']}:{seg['val']}"
                else: 
                    c = '#9B59B6'
                    lbl = f"[{seg['type']}] V={seg['val']}"
                
                px, py = [l['x'] for l in seg['path']], [-l['y'] for l in seg['path']]
                self.ax.plot(px, py, color=c, linewidth=5, zorder=8, clip_on=True)
                
                mid = len(px)//2
                self.ax.text(px[mid], py[mid], lbl, color='black', fontsize=9, bbox=dict(facecolor='white', alpha=0.8, pad=1), clip_on=True)
                
                # Vẽ Vector Hướng (Heading) cho Quỹ Đạo
                if len(px) > 2:
                    step = max(1, len(px) // 10) 
                    qx, qy, qdx, qdy = [], [], [], []
                    for k in range(0, len(px)-1, step):
                        qx.append(px[k]); qy.append(py[k])
                        dx, dy = px[k+1] - px[k], py[k+1] - py[k]
                        length = math.hypot(dx, dy)
                        if length > 0: 
                            qdx.append(dx/length); qdy.append(dy/length)
                    self.ax.scatter(qx, qy, color='black', s=10, zorder=9, clip_on=True)
                    self.ax.quiver(qx, qy, qdx, qdy, color='black', scale=25, width=0.003, headwidth=4, zorder=10, clip_on=True)

                marker = '*' if (i==len(self.route_segments)-1 and self.is_route_finalized) else 'o'
                m_c = '#E74C3C' if marker=='*' else '#F39C12'
                self.ax.scatter(self.xs[idx], self.ys[idx], c=m_c, s=150 if marker=='o' else 350, marker=marker, edgecolors='black', zorder=10, clip_on=True)
                
                self.tree.insert("", "end", values=(f"Đoạn {i}", seg['type'], lbl, f"{seg['dist']:.1f}"))
                
        self.canvas.draw_idle()

    def clear_all(self): 
        self.route_segments = []
        self.is_route_finalized = False
        self.update_viz()
        
    def undo_last(self):
        if len(self.route_segments) > 1: 
            self.route_segments.pop()
            self.is_route_finalized = False
        elif len(self.route_segments) == 1: 
            self.route_segments = []
            self.is_route_finalized = False
        self.update_viz()
        
    def finalize_route(self): 
        if len(self.route_segments) > 1: 
            self.is_route_finalized = True
            self.update_viz()

    def export_trajectory_time(self):
        try:
            if not self.is_route_finalized: return messagebox.showwarning("Lỗi", "Vui lòng '📍 CHỐT ĐIỂM CUỐI'.")
            full_path, d_arr, cur_dist = [], [0.0], 0.0
            for seg in self.route_segments[1:]:
                for pt in seg['path']:
                    if not full_path: full_path.append(pt)
                    else:
                        d = math.hypot(pt['x']-full_path[-1]['x'], pt['y']-full_path[-1]['y'])
                        if d > 0.05: cur_dist += d; full_path.append(pt); d_arr.append(cur_dist)
            if len(full_path) < 3: return messagebox.showwarning("Lỗi", "Quãng đường quá ngắn.")
                
            px, py = [pt['x'] for pt in full_path], [pt['y'] for pt in full_path]
            total_s = d_arr[-1]; seg_bounds, acc_dist = [0.0], 0.0
            for seg in self.route_segments[1:]: acc_dist += seg['dist']; seg_bounds.append(acc_dist)

            dt, t, s, v, a = 0.05, 0.0, 0.0, 0.0, 0.0
            raw_data = []; max_iter = 50000; curr_iter = 0
            
            def safe_float(val, default=0.0):
                try: return float(val)
                except: return default
            
            while s < total_s and curr_iter < max_iter:
                curr_iter += 1
                curr_seg_idx = 1
                for i in range(1, len(self.route_segments)):
                    if s <= seg_bounds[i]: curr_seg_idx = i; break
                seg = self.route_segments[curr_seg_idx]

                v_old = v; params = seg.get('params', {}); seg_type = str(seg.get('type', 'speed'))
                tgt_v = 0.0 

                # --- FIX LOGIC ĐỘNG LỰC HỌC DỌC (SURGE) ---
                if seg_type == '2a': 
                    tgt_v = safe_float(params.get('v', 90)) / 3.6; tgt_a = safe_float(params.get('a', 5)); j_max = safe_float(params.get('j', 2))
                    if v < tgt_v - 0.1: 
                        if a < tgt_a: a += j_max * dt 
                        v += a * dt
                    elif v > tgt_v + 0.1:
                        a = max(-5.0, a - j_max * dt); v += a * dt
                    else: a = 0.0; v = tgt_v # Giữ đều
                        
                elif seg_type == '2d':
                    v_init = safe_float(params.get('v', 80)) / 3.6; t_brake = safe_float(params.get('t', 4))
                    if t_brake <= 0: t_brake = 1.0 
                    a = -(v_init / t_brake); v = max(0, v + a * dt)
                else: 
                    val_num = safe_float(seg.get('val', 0.0))
                    if seg_type == 'speed' or '2' in seg_type:
                        tgt_v = val_num / 3.6
                        if v < tgt_v - 0.1: a = 3.0; v += a * dt 
                        elif v > tgt_v + 0.1: a = -3.0; v += a * dt
                        else: a = 0.0; v = tgt_v # ĐÃ FIX: a = 0 khi chạy đều
                    else: a = val_num; v += a * dt; tgt_v = v 
                        
                v = min(max(v, 0.0), 40.0); jerk = (v - v_old) / dt - a 
                dist_to_end = total_s - s; safe_brake = math.sqrt(2.0 * 5.0 * max(0.0, dist_to_end - 1.0)) 
                if v > safe_brake: v = safe_brake; a = -5.0

                curr_x, curr_y = np.interp(s, d_arr, px), np.interp(s, d_arr, py)
                raw_data.append([t, curr_x, curr_y, v*3.6, a, jerk])
                s += v * dt; t += dt
                
                if v < 0.1 and dist_to_end < 1.5: break
                if v <= 0.01 and tgt_v <= 0.01: break

            if len(raw_data) < 2: return messagebox.showwarning("Lỗi", "Dữ liệu quá ngắn.")
            
            df = pd.DataFrame(raw_data, columns=['Time_s', 'X', 'Y', 'Speed_kmh', 'Accel_X_ms2', 'Jerk'])
            
            # --- FIX LOGIC ĐỘNG LỰC HỌC NGANG (SWAY & YAW) ---
            # 1. Bo tròn quỹ đạo X, Y để khử góc vuông của CARLA (Lọc nhiễu bán kính cong)
            df['X_smooth'] = df['X'].rolling(window=15, min_periods=1, center=True).mean()
            df['Y_smooth'] = df['Y'].rolling(window=15, min_periods=1, center=True).mean()
            
            dX = np.gradient(df['X_smooth']); dY = np.gradient(df['Y_smooth'])
            yaw_unwrapped = np.unwrap(np.arctan2(dY, dX))
            
            # 2. Lọc tiếp Vận tốc góc để mô phỏng quán tính xoay của xe thực tế
            yaw_rate = np.gradient(yaw_unwrapped, dt)
            df['Yaw_rad'] = yaw_unwrapped
            df['YawRate_rad_s'] = pd.Series(yaw_rate).rolling(window=20, min_periods=1, center=True).mean()
            
            # 3. Tính Lực ly tâm và Cắt đỉnh ở mức thực tế (Max 1G ~ 10 m/s2)
            a_lat = (df['Speed_kmh'] / 3.6) * df['YawRate_rad_s']
            df['Accel_Y_ms2'] = np.clip(pd.Series(a_lat).rolling(window=10, min_periods=1).mean(), -10.0, 10.0)

            cols = ['Time_s', 'X', 'Y', 'Yaw_rad', 'Speed_kmh', 'Accel_X_ms2', 'Accel_Y_ms2', 'YawRate_rad_s', 'Jerk']
            self.show_preview_dialog(df[cols])
        except Exception as e: 
            import traceback; print(traceback.format_exc())
            messagebox.showerror("Lỗi", f"Sự cố xuất dữ liệu:\n{str(e)}")
    
    def export_vector_map(self):
        """Lưu khung nhìn bản đồ hiện tại thành file Vector (SVG/PDF) chất lượng cao"""
        path = filedialog.asksaveasfilename(
            defaultextension=".svg", 
            filetypes=[("Vector SVG (Web/Word)", "*.svg"), ("Vector PDF (LaTeX)", "*.pdf"), ("EPS (In ấn)", "*.eps")], 
            initialfile=f"Map_Vector_{self.map_name}.svg"
        )
        if path:
            try:
                # Xóa tạm các viền (spines) của đồ thị nếu muốn ảnh sạch hơn (Tùy chọn)
                # dpi=300 và bbox_inches='tight' đảm bảo cắt bỏ khoảng trắng thừa
                self.fig.savefig(path, format=path.split('.')[-1], dpi=300, bbox_inches='tight', transparent=True)
                messagebox.showinfo("Thành công", f"Đã lưu ảnh Bản đồ Vector siêu nét tại:\n{path}")
            except Exception as e:
                messagebox.showerror("Lỗi Lưu Ảnh", str(e))

    def export_to_simulation_manager(self):
        """Xuất dữ liệu chuẩn định dạng để nạp thẳng vào scenario_manager_carla.py"""
        if len(self.route_segments) < 2:
            return messagebox.showwarning("Cảnh báo", "Thầy cần lập lịch ít nhất 1 đoạn đường (Có START và GOAL).")

        # 1. TRÍCH XUẤT DỮ LIỆU ĐỂ TẠO FILE JSON
        start_idx = self.route_segments[0]['end_idx']
        waypoint_indices = [str(seg['end_idx']) for seg in self.route_segments[1:]]
        max_speed = 30.0
        
        # Tìm vận tốc lớn nhất trong lộ trình làm Speed tham chiếu
        for seg in self.route_segments:
            v = float(seg.get('params', {}).get('v', seg.get('val', 30.0)))
            if v > max_speed: max_speed = v

        json_data = {
            "map": self.map_name,
            "start_index": str(start_idx),
            "waypoints": ", ".join(waypoint_indices),
            "speed": str(max_speed),
            "algorithm": "A* (Chính xác tuyệt đối - 1 Tuyến)",
            "mode": "Chế độ 3D (Đầy đủ Camera thực tế)",
            "profile": "Ramp (Tăng/giảm tốc mượt mà)",
            "video": "Không ghi hình Video",
            "camera": "Góc nhìn thứ Ba (Toàn cảnh sau xe)",
            "traffic": "Bỏ qua Đèn Giao Thông"
        }

        # 2. TRÍCH XUẤT DỮ LIỆU ĐỂ TẠO FILE CSV (Chỉ tọa độ x, y, z)
        csv_data = []
        for seg in self.route_segments[1:]:
            for pt in seg['path']:
                # Lưu ý: Y trong Offline Planner bị đảo dấu để hiển thị 2D, 
                # Cần lật lại dấu Y khi nạp vào môi trường CARLA 3D
                csv_data.append({'x': pt['x'], 'y': -pt['y'], 'z': pt.get('z', 1.0)})

        # BẬT HỘP THOẠI LƯU THƯ MỤC
        folder_path = filedialog.askdirectory(title="Chọn Thư mục để lưu bộ dữ liệu Đồng bộ")
        if not folder_path: return

        try:
            # Lưu File JSON
            json_file = f"{folder_path}/mca_config_{self.map_name}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=4)
                
            # Lưu File CSV (Khử trùng lặp điểm)
            csv_file = f"{folder_path}/carla_route_{self.map_name}.csv"
            df_csv = pd.DataFrame(csv_data).drop_duplicates()
            df_csv.to_csv(csv_file, index=False)

            messagebox.showinfo("Xuất Đồng bộ Thành công", 
                                f"Đã tạo 2 file tương thích với Simulation Manager:\n\n"
                                f"1. {json_file.split('/')[-1]} (Dùng nút 'Tải Cấu hình JSON')\n"
                                f"2. {csv_file.split('/')[-1]} (Dùng nút 'NẠP QUỸ ĐẠO CSV')\n\n"
                                f"Thầy có thể bật scenario_manager_carla.py lên và Nạp 1 trong 2 file này.")
        except Exception as e:
            messagebox.showerror("Lỗi Lưu File", str(e))

    def plot_csv_data(self):
        path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not path: return
        try:
            df = pd.read_csv(path)
            if 'Accel_Y_ms2' not in df.columns: raise ValueError("File thiếu dữ liệu Gia tốc ngang.")
            
            top = tk.Toplevel(self.root); top.title(f"Động lực học MCA: {path.split('/')[-1]}")
            top.geometry("1000x800")
            
            fig, axs = plt.subplots(4, 1, figsize=(9, 10))
            fig.patch.set_facecolor('#F0F0F0')
            
            axs[0].plot(df['X'], df['Y'], color='blue', lw=2); axs[0].set_title("Quỹ đạo không gian (X-Y)", fontweight="bold"); axs[0].axis('equal'); axs[0].grid(True)
            axs[1].plot(df['Time_s'], df['Speed_kmh'], color='orange', lw=2); axs[1].set_title("Vận tốc (V-t)", fontweight="bold"); axs[1].set_ylabel("km/h"); axs[1].grid(True)
            axs[2].plot(df['Time_s'], df['Accel_X_ms2'], color='green', lw=2); axs[2].set_title("Gia tốc DỌC (Surge)", fontweight="bold"); axs[2].set_ylabel("m/s²"); axs[2].grid(True)
            axs[3].plot(df['Time_s'], df['Accel_Y_ms2'], color='red', lw=2); axs[3].set_title("Gia tốc NGANG (Sway)", fontweight="bold"); axs[3].set_ylabel("m/s²"); axs[3].set_xlabel("Thời gian (s)"); axs[3].grid(True)
            
            plt.tight_layout(); canvas = FigureCanvasTkAgg(fig, master=top); canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            NavigationToolbar2Tk(canvas, top)
        except Exception as e: messagebox.showerror("Lỗi Vẽ Đồ thị", str(e))

    def save_spawn_list(self):
        """Lưu danh sách thứ tự các điểm Spawn Point đã chọn"""
        if not self.route_segments: 
            return messagebox.showwarning("Lỗi", "Chưa có lộ trình để lưu.")
        
        data_spawns = []
        for i, seg in enumerate(self.route_segments):
            idx = seg['end_idx']
            sp = self.spawn_pts[idx]
            data_spawns.append({
                'Order': i, 
                'Spawn_Index': idx, 
                'X': sp['x'], 'Y': sp['y'], 'Z': sp['z'],
                'Pitch': sp['pitch'], 'Yaw': sp['yaw'], 'Roll': sp['roll']
            })
            
        path = filedialog.asksaveasfilename(defaultextension=".csv", 
                                             initialfile=f"spawn_list_{self.map_name}.csv",
                                             filetypes=[("CSV Files", "*.csv")])
        if path:
            pd.DataFrame(data_spawns).to_csv(path, index=False)
            messagebox.showinfo("Thành công", "Đã xuất danh sách Spawn Points.")

    def import_csv(self):
        """Nạp lại lộ trình từ file CSV đã lưu"""
        path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not path: return
        try:
            df = pd.read_csv(path)
            # Kiểm tra map
            if 'Town' in df.columns:
                csv_town = df['Town'].iloc[0]
                if csv_town != self.map_name:
                    if not messagebox.askyesno("Đổi Map", f"File thuộc {csv_town}. Thầy có muốn chuyển Map không?"):
                        return
                    self.load_offline_map(csv_town)
            
            self.route_segments = []
            for i, r in df.iterrows():
                # Hỗ trợ cả file cấu hình lộ trình và file spawn list đơn giản
                idx = int(r['Spawn_Index']) if 'Spawn_Index' in r else int(r['wp'])
                t = r['Type'] if 'Type' in r else 'speed'
                v = r['Value'] if 'Value' in r else (r['val'] if 'val' in r else 50.0)
                
                if i == 0:
                    self.route_segments.append({'end_idx': idx, 'type': t, 'val': v, 
                                                'path': [self.spawn_pts[idx]], 'dist': 0.0, 'params': {}})
                else:
                    p1 = self.spawn_pts[self.route_segments[-1]['end_idx']]
                    p2 = self.spawn_pts[idx]
                    path = self.planner.run_routing(p1, p2, self.algo_var.get())
                    if path:
                        d = sum([math.hypot(path[k]['x']-path[k+1]['x'], path[k]['y']-path[k+1]['y']) for k in range(len(path)-1)])
                        self.route_segments.append({'start_idx': self.route_segments[-1]['end_idx'], 
                                                    'end_idx': idx, 'type': t, 'val': v, 
                                                    'path': path, 'dist': d, 'params': {}})
            
            self.update_viz()
            messagebox.showinfo("Thành công", f"Đã nạp {len(self.route_segments)} điểm lộ trình.")
        except Exception as e:
            messagebox.showerror("Lỗi Nạp File", f"Không thể đọc file: {e}") 
    def export_trajectory_time(self):
        try:
            if not self.is_route_finalized: 
                return messagebox.showwarning("Lỗi", "Vui lòng bấm nút '📍 CHỐT ĐIỂM CUỐI LÀM ĐÍCH' trước khi xuất dữ liệu.")
                
            full_path, d_arr, cur_dist = [], [0.0], 0.0
            for seg in self.route_segments[1:]:
                for pt in seg['path']:
                    if not full_path: full_path.append(pt)
                    else:
                        d = math.hypot(pt['x']-full_path[-1]['x'], pt['y']-full_path[-1]['y'])
                        if d > 0.02: # Lọc điểm trùng để tránh lỗi đạo hàm
                            cur_dist += d
                            full_path.append(pt)
                            d_arr.append(cur_dist)
            
            if len(full_path) < 3: 
                return messagebox.showwarning("Lỗi", "Quãng đường quá ngắn, không đủ số liệu để nội suy động học.")
                
            px, py = [pt['x'] for pt in full_path], [pt['y'] for pt in full_path]
            total_s = d_arr[-1]
            seg_bounds, acc_dist = [0.0], 0.0
            for seg in self.route_segments[1:]: 
                acc_dist += seg['dist']
                seg_bounds.append(acc_dist)

            dt, t, s, v, a = 0.05, 0.0, 0.0, 0.0, 0.0
            raw_data = []
            
            def safe_float(val, default=0.0):
                try: return float(val)
                except: return default
            
            # CHỐT CHẶN 1: Giới hạn tối đa 50,000 khung hình (Tránh đơ App tuyệt đối)
            max_iter = 50000
            curr_iter = 0
            
            while s < total_s and curr_iter < max_iter:
                curr_iter += 1
                
                curr_seg_idx = 1
                for i in range(1, len(self.route_segments)):
                    if s <= seg_bounds[i]: curr_seg_idx = i; break
                seg = self.route_segments[curr_seg_idx]

                v_old = v
                params = seg.get('params', {})
                seg_type = str(seg.get('type', 'speed'))
                tgt_v = 0.0 # Khởi tạo biến vận tốc đích

                # Tính toán Động lực học
                if seg_type == '2a': 
                    tgt_v = safe_float(params.get('v', 90)) / 3.6
                    tgt_a = safe_float(params.get('a', 5))
                    j_max = safe_float(params.get('j', 2))
                    if v < tgt_v: 
                        if a < tgt_a: a += j_max * dt 
                        v += a * dt
                    else:
                        a = max(0, a - j_max * dt); v = tgt_v
                elif seg_type == '2d':
                    v_init = safe_float(params.get('v', 80)) / 3.6
                    t_brake = safe_float(params.get('t', 4))
                    if t_brake <= 0: t_brake = 1.0 
                    a = -(v_init / t_brake) 
                    v = max(0, v + a * dt)
                else: 
                    val_num = safe_float(seg.get('val', 0.0))
                    if seg_type == 'speed' or '2' in seg_type:
                        tgt_v = val_num / 3.6
                        if v < tgt_v: a = 3.0; v = min(tgt_v, v + a * dt) 
                        else: a = -3.0; v = max(tgt_v, v + a * dt)
                    else: 
                        a = val_num
                        v += a * dt
                        tgt_v = v # Chế độ Accel không có tgt_v cố định
                        
                v = min(max(v, 0.0), 40.0) 
                jerk = (v - v_old) / dt - a 

                dist_to_end = total_s - s
                safe_brake = math.sqrt(2.0 * 5.0 * max(0.0, dist_to_end - 1.0)) 
                if v > safe_brake: 
                    v = safe_brake
                    a = -5.0

                curr_x, curr_y = np.interp(s, d_arr, px), np.interp(s, d_arr, py)
                raw_data.append([t, curr_x, curr_y, v*3.6, a, jerk])

                s += v * dt; t += dt
                
                # CHỐT CHẶN 2: Ngắt vòng lặp bình thường khi đến đích
                if v < 0.1 and dist_to_end < 1.5: break
                
                # CHỐT CHẶN 3 (SỬA LỖI ĐƠ): Nếu xe đã dừng hẳn theo lệnh (v = 0), ngắt vòng lặp ngay để tránh đóng băng "s"
                if v <= 0.01 and tgt_v <= 0.01:
                    break

            if curr_iter >= max_iter:
                messagebox.showwarning("Cảnh báo", "Hành trình quá dài hoặc xe bị kẹt ở vận tốc 0. Hệ thống đã tự động ngắt để bảo vệ phần mềm!")

            if len(raw_data) < 2:
                return messagebox.showwarning("Lỗi", "Không đủ dữ liệu mô phỏng.")

            df = pd.DataFrame(raw_data, columns=['Time_s', 'X', 'Y', 'Speed_kmh', 'Accel_X_ms2', 'Jerk'])
            
            dX = np.gradient(df['X'])
            dY = np.gradient(df['Y'])
            yaw = np.arctan2(dY, dX)
            yaw_unwrapped = np.unwrap(yaw)
            yaw_rate = np.gradient(yaw_unwrapped, dt)
            
            a_lat = (df['Speed_kmh'] / 3.6) * yaw_rate
            
            df['Yaw_rad'] = yaw_unwrapped
            df['YawRate_rad_s'] = pd.Series(yaw_rate).rolling(window=5, min_periods=1).mean()
            df['Accel_Y_ms2'] = pd.Series(a_lat).rolling(window=5, min_periods=1).mean()

            cols = ['Time_s', 'X', 'Y', 'Yaw_rad', 'Speed_kmh', 'Accel_X_ms2', 'Accel_Y_ms2', 'YawRate_rad_s', 'Jerk']
            self.show_preview_dialog(df[cols])
            
        except Exception as e:
            import traceback
            messagebox.showerror("Lỗi Thuật Toán", f"Hệ thống gặp sự cố:\n{str(e)}")
            print(traceback.format_exc())

    def show_preview_dialog(self, df):
        """Hộp thoại xem trước dữ liệu và chọn chế độ lưu"""
        top = tk.Toplevel(self.root)
        top.title("Preview Trajectory Data")
        top.geometry("800x600")
        
        ttk.Label(top, text="Dữ liệu Trajectory nội suy (20Hz):", font=('Arial', 10, 'bold')).pack(pady=5)
        txt = tk.Text(top, height=25, width=100, font=('Consolas', 9))
        txt.pack(fill="both", expand=True, padx=10)
        
        # Hiển thị 100 dòng đầu
        preview = df.head(100).to_csv(index=False)
        txt.insert(tk.END, preview + "\n... (Còn tiếp) ...")
        txt.config(state=tk.DISABLED)
        
        btn_f = ttk.Frame(top); btn_f.pack(fill="x", pady=10)
        
        def save_full():
            p = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="trajectory_full.csv")
            if p: df.to_csv(p, index=False); top.destroy()
        
        def save_basic():
            p = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="trajectory_basic.csv")
            if p: df[['Time_s', 'X', 'Y', 'Speed_kmh']].to_csv(p, index=False); top.destroy()
            
        ttk.Button(btn_f, text="💾 LƯU ĐẦY ĐỦ (CÓ GIA TỐC/JERK/SWAY)", command=save_full).pack(side="left", expand=True, padx=5)
        ttk.Button(btn_f, text="💾 LƯU CƠ BẢN (X, Y, V, t)", command=save_basic).pack(side="right", expand=True, padx=5)
        # ==========================================================================
        # CÁC TÍNH NĂNG MỚI: ZOOM, PAN VÀ SƠ ĐỒ LIÊN KẾT ĐỘNG HỌC
        # ==========================================================================
    # ==========================================================================
    # CÁC TÍNH NĂNG MỚI: ZOOM, PAN VÀ SƠ ĐỒ LIÊN KẾT ĐỘNG HỌC
    # ==========================================================================
    def toggle_map(self):
        self.show_map = not self.show_map
        self.update_viz()
    def zoom_home(self):
        """Đưa bản đồ về góc nhìn toàn cảnh mặc định"""
        self.toolbar.home()
        self.canvas.draw_idle()

    def toggle_zoom(self):
        """Bật/Tắt chế độ kéo chuột trái để Zoom vùng (Rectangle Zoom)"""
        self.toolbar.zoom() # Gọi chức năng Zoom Rectangle của Matplotlib
        if self.toolbar.mode == 'zoom rect':
            self.btn_zoom.config(text="❌ Tắt Zoom Quét (Để click chọn điểm)")
            messagebox.showinfo("Chế độ Zoom", "Đã BẬT Zoom Quét.\nThầy hãy NHẤN GIỮ CHUỘT TRÁI và kéo thành hình chữ nhật trên bản đồ để phóng to vùng đó.\n(Nhớ tắt đi để có thể click chọn Spawn Point).")
        else:
            self.btn_zoom.config(text="🔍 Bật Zoom Quét (Kéo chuột)")

    def show_route_schematic(self):
        """Vẽ sơ đồ liên kết các Spawn Point (L/C và Khoảng cách)"""
        if len(self.route_segments) < 2:
            return messagebox.showwarning("Nhắc nhở", "Thầy cần vạch ít nhất 1 đoạn đường để tạo sơ đồ liên kết.")
            
        top = tk.Toplevel(self.root)
        top.title("Sơ đồ Liên kết Động học (Topological Schematic)")
        top.geometry("900x400")
        
        fig, ax = plt.subplots(figsize=(8, 3))
        fig.patch.set_facecolor('#F8F9FA')
        ax.set_facecolor('#F8F9FA')
        
        # Vẽ trục thời gian/chuỗi nằm ngang
        nodes_x = list(range(len(self.route_segments)))
        nodes_y = [0] * len(nodes_x)
        
        # Đánh dấu các Node (Spawn Points)
        ax.plot(nodes_x, nodes_y, color='#BDC3C7', lw=3, zorder=1)
        ax.scatter(nodes_x, nodes_y, s=400, c='#3498DB', edgecolors='black', zorder=2)
        
        for i, seg in enumerate(self.route_segments):
            sp_idx = seg['end_idx']
            # Tên Node
            ax.text(i, 0.05, f"SP:{sp_idx}", ha='center', va='bottom', fontweight='bold', fontsize=11)
            
            # Nếu không phải điểm Start, vẽ thông tin cạnh (Edge) ở giữa 2 Node
            if i > 0:
                path = seg['path']
                geom = self.planner.analyze_geometry(path)
                dist = seg['dist']
                
                # Ký hiệu L (Line) hoặc C (Curve)
                symbol = "L" if geom == "straight" else "C"
                color = "green" if symbol == "L" else "red"
                
                # Hiển thị text trên đoạn nối
                mid_x = i - 0.5
                ax.text(mid_x, -0.05, f"({symbol}) {dist:.1f}m", ha='center', va='top', 
                        color=color, fontweight='bold', fontsize=10, bbox=dict(facecolor='white', edgecolor=color, boxstyle='round,pad=0.2'))

        ax.axis('off') # Tắt trục tọa độ vì đây là sơ đồ mạng lưới
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=top)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(canvas, top)
if __name__ == "__main__":
    app = OfflineMissionPlanner(); app.root.mainloop()