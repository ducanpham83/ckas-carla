import carla
import pygame
import numpy as np
import time
import math
import csv
import os
import sys
import json
import datetime
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import pandas as pd
from scipy.signal import butter, filtfilt, freqz
from scipy.fft import fft, fftfreq

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# ==============================================================================
# -- 1. GIAO DIỆN CẤU HÌNH & QUẢN LÝ KỊCH BẢN (V48 - THE FINAL BUILD) ---------
# ==============================================================================

def run_setup_gui():
    root = tk.Tk()
    root.title("CARLA Master Setup v48 - Ultimate Integration")
    root.geometry("560x950")
    
    config_result = {}

    ttk.Label(root, text="THIẾT LẬP NGHIỆM THỨC MCA", font=('Arial', 14, 'bold')).pack(pady=10)
    
    cfg_frame = ttk.Frame(root); cfg_frame.pack(fill="x", padx=20, pady=2)
    
    map_combo = ttk.Combobox(root, values=["Town01", "Town02", "Town03", "Town04", "Town05", "Town06", "Town07", "Town10HD"])
    start_entry = ttk.Entry(root, justify='center')
    path_entry = ttk.Entry(root, width=40, justify='center')
    speed_entry = ttk.Entry(root, justify='center')
    profile_combo = ttk.Combobox(root, values=["Ramp (Tăng/giảm tốc mượt mà)", "Dynamic (Ramp + Tuân thủ Biển báo)", "Step (Giật cấp)"], width=45)
    algo_combo = ttk.Combobox(root, values=["A* (Chính xác tuyệt đối - 1 Tuyến)", "Greedy (Tham lam - Đa tuyến)"], width=45)
    mode_combo = ttk.Combobox(root, values=["Chế độ 2D (Chỉ Sa bàn Vector - Siêu nhẹ)", "Chế độ 3D (Đầy đủ Camera thực tế)"], width=45)
    video_combo = ttk.Combobox(root, values=["Không ghi hình Video", "Có ghi hình Video (Yêu cầu OpenCV)"], width=45)
    camera_combo = ttk.Combobox(root, values=["Góc nhìn thứ Ba (Toàn cảnh sau xe)", "Góc nhìn thứ Nhất (Vị trí người lái)"], width=45)
    traffic_combo = ttk.Combobox(root, values=["Bỏ qua Đèn Giao Thông", "Tuân thủ Dừng Đèn Đỏ"], width=45)

    def save_config():
        data = {
            "map": map_combo.get(), "start_index": start_entry.get().strip(), "waypoints": path_entry.get().strip(), 
            "speed": speed_entry.get().strip(), "algorithm": algo_combo.get(), "mode": mode_combo.get(),
            "profile": profile_combo.get(), "video": video_combo.get(), "camera": camera_combo.get(), "traffic": traffic_combo.get()
        }
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON Config", "*.json")])
        if path:
            with open(path, 'w', encoding='utf-8') as f: json.dump(data, f, indent=4)
            messagebox.showinfo("Thành công", "Đã lưu cấu hình.")

    def load_config():
        path = filedialog.askopenfilename(filetypes=[("JSON Config", "*.json")])
        if path:
            with open(path, 'r', encoding='utf-8') as f: d = json.load(f)
            map_combo.set(d.get("map", "Town04")); start_entry.delete(0, tk.END); start_entry.insert(0, d.get("start_index", ""))
            path_entry.delete(0, tk.END); path_entry.insert(0, d.get("waypoints", "")); speed_entry.delete(0, tk.END); speed_entry.insert(0, d.get("speed", "30.0"))
            algo_combo.set(d.get("algorithm", "A* (Chính xác tuyệt đối - 1 Tuyến)")); mode_combo.set(d.get("mode", "Chế độ 2D (Chỉ Sa bàn Vector - Siêu nhẹ)"))
            profile_combo.set(d.get("profile", "Ramp (Tăng/giảm tốc mượt mà)")); video_combo.set(d.get("video", "Không ghi hình Video"))
            camera_combo.set(d.get("camera", "Góc nhìn thứ Ba (Toàn cảnh sau xe)")); traffic_combo.set(d.get("traffic", "Bỏ qua Đèn Giao Thông"))
            messagebox.showinfo("Thành công", "Nạp cấu hình hoàn tất.")

    ttk.Button(cfg_frame, text="💾 Lưu Cấu hình JSON", command=save_config).pack(side="left", expand=True, padx=2)
    ttk.Button(cfg_frame, text="📂 Tải Cấu hình JSON", command=load_config).pack(side="right", expand=True, padx=2)

    ttk.Label(root, text="1. Chọn Bản đồ (Map):").pack()
    map_combo.current(3); map_combo.pack(pady=2)
    
    def _get_carla_data():
        client = carla.Client('127.0.0.1', 2000); client.set_timeout(30.0); return client.load_world(map_combo.get()).get_map()

    def show_spawn_points_map():
        try:
            carla_map = _get_carla_data(); spawn_pts = carla_map.get_spawn_points()
            map_win = tk.Toplevel(root); map_win.title("Bản đồ Spawn Points & Hướng đi"); map_win.geometry("1100x800")
            fig, ax = plt.subplots(figsize=(12, 10))
            
            qx, qy, qdx, qdy, q_color = [], [], [], [], []
            for seg in carla_map.get_topology():
                wp1, wp2 = seg[0], seg[1]
                x1, y1 = wp1.transform.location.x, -wp1.transform.location.y
                x2, y2 = wp2.transform.location.x, -wp2.transform.location.y
                ax.plot([x1, x2], [y1, y2], color='#909090', linewidth=wp1.lane_width*1.5, alpha=0.3, solid_capstyle='round')
                
                dx, dy = x2 - x1, y2 - y1; length = math.hypot(dx, dy)
                if length > 0.5:
                    qx.append((x1+x2)/2); qy.append((y1+y2)/2); qdx.append(dx/length); qdy.append(dy/length)
                    q_color.append(math.degrees(math.atan2(dy, dx)))

            if qx: 
                ax.quiver(qx, qy, qdx, qdy, q_color, cmap='hsv', scale=70, width=0.003, headwidth=4, alpha=0.5)
                sm = plt.cm.ScalarMappable(cmap='hsv', norm=plt.Normalize(vmin=-180, vmax=180))
                cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.02, ticks=[-180, -90, 0, 90, 180])
                cbar.ax.set_yticklabels(['Tây (W)', 'Nam (S)', 'Đông (E)', 'Bắc (N)', 'Tây (W)'])
                cbar.set_label('Hướng di chuyển', rotation=270, labelpad=15, fontweight='bold')

            xs = [p.location.x for p in spawn_pts]; ys = [-p.location.y for p in spawn_pts]
            ax.scatter(xs, ys, c='#00FFFF', s=50, alpha=1.0, edgecolors='#000000', zorder=5)
            for i, (x, y) in enumerate(zip(xs, ys)): ax.text(x, y + 2.0, str(i), fontsize=9, fontweight='bold', color='#111111', ha='center', zorder=6)
            
            ax.set_title(f"Sơ đồ Làn đường & Hướng đi - {map_combo.get()}", fontweight='bold', fontsize=14)
            ax.axis('equal'); ax.grid(True, linestyle=':', alpha=0.5); fig.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=map_win); canvas.draw(); NavigationToolbar2Tk(canvas, map_win).update(); canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        except Exception as e: messagebox.showerror("Lỗi", str(e))

    def show_satellite_map():
        try:
            client = carla.Client('127.0.0.1', 2000); client.set_timeout(10.0)
            world = client.load_world(map_combo.get()); carla_map = world.get_map()
            spawn_pts = carla_map.get_spawn_points()
            if not spawn_pts: raise Exception("Bản đồ không có điểm Spawn.")
            
            xs = [p.location.x for p in spawn_pts]; ys = [p.location.y for p in spawn_pts]
            center_x, center_y = sum(xs) / len(xs), sum(ys) / len(ys)
            z_cam = max(200.0, (max(xs) - min(xs)) * 0.8)

            pygame.init(); display = pygame.display.set_mode((1280, 720))
            pygame.display.set_caption(f"Vệ Tinh Toàn Cảnh 3D - {map_combo.get()}")

            bp = world.get_blueprint_library().find('sensor.camera.rgb')
            bp.set_attribute('image_size_x', '1280'); bp.set_attribute('image_size_y', '720'); bp.set_attribute('fov', '90')
            camera = world.spawn_actor(bp, carla.Transform(carla.Location(x=center_x, y=center_y, z=z_cam), carla.Rotation(pitch=-90.0, yaw=90.0)))

            surface_dict = {'surf': None}
            def img_cb(image):
                array = np.reshape(np.frombuffer(image.raw_data, dtype=np.dtype("uint8")), (image.height, image.width, 4))[:, :, :3][:, :, ::-1]
                surface_dict['surf'] = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            camera.listen(img_cb)

            clock = pygame.time.Clock(); running = True
            while running:
                world.tick(); clock.tick(30)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE): running = False
                if surface_dict['surf'] is not None: display.blit(surface_dict['surf'], (0, 0))
                pygame.display.flip()

            camera.destroy(); pygame.quit()
        except Exception as e: messagebox.showerror("Lỗi Camera 3D", str(e))

    def open_interactive_picker():
        try:
            carla_map = _get_carla_data(); spawn_pts = carla_map.get_spawn_points()
            picker_win = tk.Toplevel(root); picker_win.title("Trợ lý Chọn Tuyến tương tác"); picker_win.geometry("1100x800")
            
            ctrl_frame = ttk.Frame(picker_win); ctrl_frame.pack(fill="x", padx=10, pady=5)
            ttk.Label(ctrl_frame, text="Click chuột vào các điểm màu Xanh lơ để chọn lộ trình.", font=('Arial', 10, 'bold')).pack(side="left")
            
            picked_indices = []; fig, ax = plt.subplots(figsize=(12, 10))
            for seg in carla_map.get_topology(): ax.plot([seg[0].transform.location.x, seg[1].transform.location.x], [-seg[0].transform.location.y, -seg[1].transform.location.y], color='#C0C0C0', linewidth=1, alpha=0.5)

            xs = np.array([p.location.x for p in spawn_pts]); ys = np.array([-p.location.y for p in spawn_pts])
            ax.scatter(xs, ys, c='#00FFFF', s=60, alpha=0.8, edgecolors='#000000', picker=True)
            for i, (x, y) in enumerate(zip(xs, ys)): ax.text(x, y + 2.5, str(i), fontsize=8, color='#333333', ha='center')

            lines_plot, = ax.plot([], [], color='magenta', linewidth=3, linestyle='-.'); markers_plot = ax.scatter([], [], c='red', s=150, zorder=10)

            def on_pick(event):
                if event.xdata is None or event.ydata is None: return
                idx = int(np.argmin((xs - event.xdata)**2 + (ys - event.ydata)**2))
                if not picked_indices or picked_indices[-1] != idx:
                    picked_indices.append(idx); update_plot()

            def update_plot():
                if not picked_indices: return
                cx = [xs[i] for i in picked_indices]; cy = [ys[i] for i in picked_indices]
                lines_plot.set_data(cx, cy); markers_plot.set_offsets(np.c_[cx, cy])
                markers_plot.set_color(['lime'] + ['red'] * (len(picked_indices) - 1)); fig.canvas.draw_idle()

            def undo_last():
                if picked_indices:
                    picked_indices.pop()
                    if not picked_indices:
                        lines_plot.set_data([], []); markers_plot.set_offsets(np.empty((0, 2))); fig.canvas.draw_idle()
                    else: update_plot()

            def confirm_selection():
                if len(picked_indices) < 2: return messagebox.showwarning("Cảnh báo", "Cần chọn ít nhất 2 điểm")
                start_entry.delete(0, tk.END); start_entry.insert(0, str(picked_indices[0]))
                path_str = ", ".join(map(str, picked_indices[1:])); path_entry.delete(0, tk.END); path_entry.insert(0, path_str)
                picker_win.destroy(); messagebox.showinfo("Thành công", f"Đã nạp tuyến đường.")

            ttk.Button(ctrl_frame, text="✅ XÁC NHẬN VÀ NẠP", command=confirm_selection).pack(side="right", padx=5)
            ttk.Button(ctrl_frame, text="↩️ Hoàn tác (Undo)", command=undo_last).pack(side="right", padx=5)
            
            fig.canvas.mpl_connect('button_press_event', on_pick)
            ax.axis('equal'); ax.grid(True, linestyle=':', alpha=0.5); fig.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=picker_win); canvas.draw(); NavigationToolbar2Tk(canvas, picker_win).update(); canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        except Exception as e: messagebox.showerror("Lỗi", str(e))

    btn_frame = ttk.Frame(root); btn_frame.pack(pady=2)
    ttk.Button(btn_frame, text="📍 XEM BẢN ĐỒ MẠNG LƯỚI", command=show_spawn_points_map).pack(side="left", padx=2)
    ttk.Button(btn_frame, text="🗺️ VỆ TINH TOÀN CẢNH", command=show_satellite_map).pack(side="left", padx=2)
    ttk.Button(btn_frame, text="🖱️ CHỌN TUYẾN BẰNG CHUỘT", command=open_interactive_picker, style="Accent.TButton").pack(side="left", padx=2)

    ttk.Label(root, text="2. Điểm xuất phát (Start Index):").pack(pady=(5,0))
    start_entry.insert(0, "10"); start_entry.pack()
    ttk.Label(root, text="3. Lộ trình (Waypoints, VD: 25, 45, 80):").pack(pady=(5,0))
    path_entry.insert(0, "25, 45, 80"); path_entry.pack()
    ttk.Label(root, text="4. Tốc độ tối đa (km/h):").pack(pady=(5,0))
    speed_entry.insert(0, "30.0"); speed_entry.pack()

    ttk.Label(root, text="5. Hành vi Lái xe & Tìm đường:").pack(pady=(5,0))
    profile_combo.current(0); profile_combo.pack(pady=2)
    algo_combo.current(0); algo_combo.pack(pady=2)
    traffic_combo.current(0); traffic_combo.pack(pady=2)

    ttk.Label(root, text="6. Đồ họa & Video:").pack(pady=(5,0))
    mode_combo.current(0); mode_combo.pack(pady=2)
    camera_combo.current(0); camera_combo.pack(pady=2)
    video_combo.current(0); video_combo.pack(pady=2)

    route_frame = ttk.LabelFrame(root, text="7. Quỹ đạo Tính toán (Đa tuyến)")
    route_frame.pack(fill="x", padx=20, pady=5)
    radio_frame = ttk.Frame(route_frame); radio_frame.pack(fill="x", padx=5, pady=5)
    selected_route_idx = tk.IntVar(value=-1); computed_routes = []
    ttk.Label(radio_frame, text="Vui lòng nhấn 'Tìm Quỹ Đạo' trước.", font=('Arial', 9, 'italic')).pack(pady=2)

    def show_trajectory_map():
        nonlocal computed_routes
        try:
            s_idx = int(start_entry.get().strip()); p_indices = [int(i.strip()) for i in path_entry.get().split(',') if i.strip().isdigit()]
            carla_map = _get_carla_data(); spawn_pts = carla_map.get_spawn_points()
            start_loc = spawn_pts[s_idx].location; targets = [spawn_pts[i].location for i in p_indices if i < len(spawn_pts)]
            
            if "A*" in algo_combo.get():
                import sys; carla_agent_path = r"D:\Setup_Program\CARLA\CARLA_0.9.13\WindowsNoEditor\PythonAPI\carla"
                if carla_agent_path not in sys.path: sys.path.insert(0, carla_agent_path)
                from agents.navigation.global_route_planner import GlobalRoutePlanner
                grp = GlobalRoutePlanner(carla_map, 2.0); optimal_path = []; curr_loc = start_loc
                for target in targets:
                    for wp, _ in grp.trace_route(curr_loc, target): optimal_path.append(wp.transform.location)
                    curr_loc = target
                computed_routes = [optimal_path]
            else:
                paths = []
                for var_idx in range(3):
                    path = [start_loc]; curr_wp = carla_map.get_waypoint(start_loc); intersection_count = 0
                    for target in targets:
                        step = 0
                        while curr_wp.transform.location.distance(target) > 3.0 and step < 2000:
                            nexts = curr_wp.next(2.0)
                            if not nexts: break
                            if len(nexts) > 1:
                                intersection_count += 1; nexts = sorted(nexts, key=lambda wp: wp.transform.location.distance(target))
                                curr_wp = nexts[1] if (var_idx > 0 and intersection_count == var_idx and len(nexts) > 1) else nexts[0]
                            else: curr_wp = nexts[0]
                            path.append(curr_wp.transform.location); step += 1
                        curr_wp = carla_map.get_waypoint(target); path.append(target)
                    if not any(abs(len(p) - len(path)) < 5 for p in paths): paths.append(path)
                computed_routes = paths

            for widget in radio_frame.winfo_children(): widget.destroy()
            
            map_win = tk.Toplevel(root); map_win.title("Bản đồ Quỹ Đạo Đề Xuất"); map_win.geometry("1000x800")
            fig, ax = plt.subplots(figsize=(12, 10))
            for seg in carla_map.get_topology():
                ax.plot([seg[0].transform.location.x, seg[1].transform.location.x], [-seg[0].transform.location.y, -seg[1].transform.location.y], color='#EEEEEE', linewidth=1)
            
            route_lines = []
            def update_route_visibility():
                sel_idx = selected_route_idx.get()
                for i, line in enumerate(route_lines):
                    if sel_idx == -1 or i == sel_idx: line.set_alpha(1.0); line.set_linewidth(4.0); line.set_zorder(10)
                    else: line.set_alpha(0.15); line.set_linewidth(1.5); line.set_zorder(3)
                fig.canvas.draw_idle()

            if "A*" in algo_combo.get():
                ttk.Radiobutton(radio_frame, text=f"Tuyến Tối ưu - {len(computed_routes[0])} điểm", variable=selected_route_idx, value=0, command=update_route_visibility).pack(anchor='w', padx=10, pady=2); selected_route_idx.set(0)
            else:
                ttk.Radiobutton(radio_frame, text="Hiển thị tất cả (Greedy)", variable=selected_route_idx, value=-1, command=update_route_visibility).pack(anchor='w', padx=10, pady=2)
                for i, route in enumerate(computed_routes): 
                    ttk.Radiobutton(radio_frame, text=f"Tuyến {i+1} - {len(route)} điểm", variable=selected_route_idx, value=i, command=update_route_visibility).pack(anchor='w', padx=10, pady=2)
                selected_route_idx.set(-1)
            
            c_styles = [('#0066FF', '-'), ('#00AA00', '--'), ('#CC00CC', ':')]
            for i, route in enumerate(computed_routes):
                r_x = [loc.x for loc in route]; r_y = [-loc.y for loc in route]
                color = c_styles[i][0] if i < len(c_styles) else 'black'; lstyle = c_styles[i][1] if i < len(c_styles) else '-'
                line, = ax.plot(r_x, r_y, color=color, linestyle=lstyle, linewidth=4.0, label=f"Quỹ đạo {i+1}", zorder=10); route_lines.append(line)
            
            xs = [p.location.x for p in spawn_pts]; ys = [-p.location.y for p in spawn_pts]
            if s_idx >= 0: ax.scatter(xs[s_idx], ys[s_idx], c='green', s=200, marker='s', edgecolors='black', label='START', zorder=11)
            if len(p_indices) > 1:
                p_xs = [xs[i] for i in p_indices[:-1] if i < len(spawn_pts)]; p_ys = [ys[i] for i in p_indices[:-1] if i < len(spawn_pts)]
                ax.scatter(p_xs, p_ys, c='orange', s=120, marker='o', edgecolors='black', label='Waypoints', zorder=11)
            if p_indices:
                last_idx = p_indices[-1]
                if last_idx < len(spawn_pts): ax.scatter(xs[last_idx], ys[last_idx], c='red', s=300, marker='*', edgecolors='black', label='GOAL', zorder=12)

            ax.legend(); ax.axis('equal'); ax.grid(True, linestyle='--', alpha=0.5); fig.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=map_win); canvas.draw(); NavigationToolbar2Tk(canvas, map_win).update(); canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        except Exception as e: messagebox.showerror("Lỗi", str(e))

    ttk.Button(root, text="🔄 TÌM & CHUẨN BỊ QUỸ ĐẠO", command=show_trajectory_map).pack(pady=5)

    # --- KHAI BÁO CÁC HÀM LƯU / NẠP CSV ---
    def save_route():
        if selected_route_idx.get() == -1 or not computed_routes:
            return messagebox.showwarning("Cảnh báo", "Vui lòng chọn tuyến.")
        route = computed_routes[selected_route_idx.get()]
        save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")], initialfile="my_route.csv")
        if save_path:
            with open(save_path, 'w', newline='') as f:
                writer = csv.writer(f); writer.writerow(['x', 'y', 'z'])
                for loc in route: writer.writerow([loc.x, loc.y, loc.z])
            messagebox.showinfo("Thành công", "Đã lưu tuyến CSV.")

    def load_route():
        nonlocal computed_routes
        load_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if load_path:
            try:
                df = pd.read_csv(load_path)
                loaded_route = [carla.Location(x=row['x'], y=row['y'], z=row['z']) for _, row in df.iterrows()]
                computed_routes = [loaded_route]
                
                for widget in radio_frame.winfo_children(): widget.destroy()
                ttk.Radiobutton(radio_frame, text=f"Tuyến từ file CSV ({len(loaded_route)} điểm)", variable=selected_route_idx, value=0).pack(anchor='w', padx=10, pady=2)
                selected_route_idx.set(0)
                
                carla_map = _get_carla_data()
                map_win = tk.Toplevel(root); map_win.title("Bản đồ Quỹ Đạo (Từ File CSV)"); map_win.geometry("1000x800")
                fig, ax = plt.subplots(figsize=(12, 10))
                for seg in carla_map.get_topology():
                    ax.plot([seg[0].transform.location.x, seg[1].transform.location.x], [-seg[0].transform.location.y, -seg[1].transform.location.y], color='#EEEEEE', linewidth=1)
                
                r_x = [loc.x for loc in loaded_route]; r_y = [-loc.y for loc in loaded_route]
                ax.plot(r_x, r_y, color='#0066FF', linestyle='-', linewidth=4.0, label="Quỹ đạo CSV", zorder=10)
                ax.scatter(loaded_route[0].x, -loaded_route[0].y, c='green', s=200, marker='s', edgecolors='black', label='START', zorder=11)
                ax.scatter(loaded_route[-1].x, -loaded_route[-1].y, c='red', s=300, marker='*', edgecolors='black', label='GOAL', zorder=12)
                
                ax.legend(); ax.axis('equal'); ax.grid(True, linestyle='--', alpha=0.5); fig.tight_layout()
                canvas = FigureCanvasTkAgg(fig, master=map_win); canvas.draw(); NavigationToolbar2Tk(canvas, map_win).update(); canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                messagebox.showinfo("Thành công", "Đã nạp tuyến đường. Thầy có thể bấm CHẠY MÔ PHỎNG.")
            except Exception as e: messagebox.showerror("Lỗi nạp file", str(e))

    file_frame = ttk.LabelFrame(root, text="8. Lưu / Nạp Quỹ Đạo Từ File (CSV)")
    file_frame.pack(fill="x", padx=20, pady=5)
    
    sl_frame = ttk.Frame(file_frame); sl_frame.pack(fill="x", padx=10, pady=10)
    ttk.Button(sl_frame, text="💾 LƯU QUỸ ĐẠO (CSV)", command=save_route).pack(side="left", expand=True, padx=5)
    ttk.Button(sl_frame, text="📂 NẠP QUỸ ĐẠO (CSV)", command=load_route).pack(side="right", expand=True, padx=5)

    def on_submit():
        if selected_route_idx.get() == -1 or not computed_routes: return messagebox.showwarning("Cảnh báo", "Vui lòng bấm Tìm Quỹ Đạo trước!")
        try: target_speed = float(speed_entry.get().strip())
        except: return messagebox.showerror("Lỗi", "Vui lòng nhập Tốc độ hợp lệ.")
        
        video_enabled = "Có" in video_combo.get() and HAS_CV2
        video_path = None
        if video_enabled:
            video_path = filedialog.asksaveasfilename(title="Chọn nơi lưu Video MP4", defaultextension=".mp4", filetypes=[("MP4 Video", "*.mp4")], initialfile=f"mca_video_{datetime.datetime.now().strftime('%H%M%S')}.mp4")
            if not video_path:
                if not messagebox.askyesno("Xác nhận", "Hủy lưu Video. Tiếp tục chạy mà KHÔNG ghi hình?"): return
                video_enabled = False
        
        nonlocal config_result
        config_result = {
            'map': map_combo.get(), 'dense_route': computed_routes[selected_route_idx.get()],
            'mode': mode_combo.get(), 'speed': target_speed, 'profile': profile_combo.get(),
            'video': video_enabled, 'video_path': video_path, 'camera': camera_combo.get(), 'traffic': traffic_combo.get()
        }
        root.quit()

    ttk.Button(root, text="🚀 CHẠY MÔ PHỎNG 3D/2D", command=on_submit, style="Accent.TButton").pack(pady=15)
    
    def safe_close():
        config_result.clear(); root.quit()
    root.protocol("WM_DELETE_WINDOW", safe_close)
    
    root.mainloop() 
    try: root.destroy() 
    except: pass
    return config_result

# ==============================================================================
# -- 2. KINEMATIC PURE PURSUIT (CORNER CUTTING FIXED) --------------------------
# ==============================================================================

class PurePursuitAutopilot:
    def __init__(self, route, target_speed_kmh, profile_type):
        self.route = route; self.user_max_speed = target_speed_kmh / 3.6; self.profile_type = profile_type
        self.current_idx = 0; self.ref_speed = 0.0 

    def run_step(self, vehicle, dt, road_limit_kmh):
        loc = vehicle.get_location(); v = vehicle.get_velocity(); speed = math.sqrt(v.x**2 + v.y**2 + v.z**2)

        min_dist = float('inf'); search_end = min(self.current_idx + 50, len(self.route))
        for i in range(self.current_idx, search_end):
            dist = loc.distance(self.route[i])
            if dist < min_dist: min_dist = dist; self.current_idx = i

        lookahead = np.clip(speed * 0.8, 2.5, 10.0) 
        target_idx = self.current_idx
        for i in range(self.current_idx, len(self.route)):
            if loc.distance(self.route[i]) > lookahead: target_idx = i; break
        target_wp = self.route[target_idx]
        
        car_yaw = vehicle.get_transform().rotation.yaw; target_yaw = math.degrees(math.atan2(target_wp.y - loc.y, target_wp.x - loc.x))
        steer = np.clip(((target_yaw - car_yaw + 180) % 360 - 180) / 45.0, -1.0, 1.0) 

        dist_to_goal = loc.distance(self.route[-1])
        safe_brake_speed = math.sqrt(2.0 * 2.5 * max(0.0, dist_to_goal - 1.5)) 

        target = self.user_max_speed
        if "Dynamic" in self.profile_type:
            limit_ms = road_limit_kmh / 3.6
            target = min(self.user_max_speed, limit_ms if limit_ms > 0 else self.user_max_speed)
            
        a_max = 5.0 
        if "Ramp" in self.profile_type or "Dynamic" in self.profile_type:
            if self.ref_speed < target: self.ref_speed += a_max * dt
            elif self.ref_speed > target: self.ref_speed -= a_max * dt
            self.ref_speed = np.clip(self.ref_speed, 0.0, target)
        else: self.ref_speed = target
            
        corner_safe_speed = max(2.0, self.user_max_speed * (1.0 - abs(steer) * 0.5))
        final_ref = min(self.ref_speed, safe_brake_speed, corner_safe_speed); speed_error = final_ref - speed
        
        if dist_to_goal < 1.5: return carla.VehicleControl(steer=steer, throttle=0.0, brake=1.0)
            
        throttle = np.clip(speed_error * 0.5, 0.0, 0.75)
        brake = np.clip(-speed_error * 1.5, 0.0, 1.0) if speed_error < -0.1 else 0.0
        return carla.VehicleControl(steer=steer, throttle=throttle, brake=brake)

class TopDownMap:
    def __init__(self, world_map, dense_route, width, height):
        self.surface = pygame.Surface((width, height)); self.surface.fill((25, 25, 30))
        xs = [loc.x for loc in dense_route]; ys = [loc.y for loc in dense_route]
        if not xs: return
        self.min_x, self.max_x = min(xs) - 80, max(xs) + 80; self.min_y, self.max_y = min(ys) - 80, max(ys) + 80
        self.scale = min(width / (self.max_x - self.min_x), height / (self.max_y - self.min_y))
        self.off_x = (width - (self.max_x - self.min_x) * self.scale) / 2; self.off_y = (height - (self.max_y - self.min_y) * self.scale) / 2
        for seg in world_map.get_topology(): pygame.draw.line(self.surface, (60, 60, 70), self._to_px(seg[0].transform.location), self._to_px(seg[1].transform.location), 2)
        pts = [self._to_px(loc) for loc in dense_route]
        if len(pts) > 1: pygame.draw.lines(self.surface, (0, 150, 255), False, pts, 4); pygame.draw.circle(self.surface, (0, 255, 0), pts[0], 6); pygame.draw.circle(self.surface, (255, 0, 0), pts[-1], 8) 
    def _to_px(self, loc): return (int((loc.x - self.min_x) * self.scale + self.off_x), int((loc.y - self.min_y) * self.scale + self.off_y))
    def render(self, display, vehicle_tf, position=(0,0)):
        temp_surface = self.surface.copy()
        cx, cy = self._to_px(vehicle_tf.location); yaw = math.radians(vehicle_tf.rotation.yaw)
        pygame.draw.polygon(temp_surface, (0, 255, 100), [(cx + 15 * math.cos(yaw), cy + 15 * math.sin(yaw)), (cx + 8 * math.cos(yaw + 2.5), cy + 8 * math.sin(yaw + 2.5)), (cx + 8 * math.cos(yaw - 2.5), cy + 8 * math.sin(yaw - 2.5))])
        display.blit(temp_surface, position)

class RenderCamera:
    def __init__(self, player, view_type):
        bp = player.get_world().get_blueprint_library().find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', '1280')
        bp.set_attribute('image_size_y', '720')
        
        # --- FIX 1: TĂNG TRƯỜNG NHÌN (FOV) ĐỂ LẤY LẠI TẦM NHÌN NGOẠI VI ---
        # Tăng FOV từ 90 lên 110 hoặc 120 độ. Đây là "chìa khóa vàng" để cảm nhận tốc độ.
        if "người lái" in view_type:
            bp.set_attribute('fov', '110') 
        else:
            bp.set_attribute('fov', '100')
            
        # --- FIX 2: BẬT HIỆU ỨNG NHÒE CHUYỂN ĐỘNG (MOTION BLUR) ---
        # CARLA hỗ trợ Post-processing. Thêm motion blur giúp cảnh vật vụt qua chân thực hơn
        if bp.has_attribute('motion_blur_intensity'):
            bp.set_attribute('motion_blur_intensity', '0.6')
        if bp.has_attribute('motion_blur_max_distortion'):
            bp.set_attribute('motion_blur_max_distortion', '3.5')

        # --- FIX 3: HẠ THẤP ĐỘ CAO CAMERA TỚI ĐÚNG TẦM MẮT THỰC TẾ ---
        # Z=1.1m là độ cao mắt trung bình của người lái xe sedan (Tesla Model 3) tính từ mặt đường
        if "người lái" in view_type: 
            cam_tf = carla.Transform(carla.Location(x=0.2, y=-0.4, z=1.1), carla.Rotation(pitch=-2.0))
        else: 
            # Góc nhìn thứ 3: Hạ thấp xuống và hất cam lên một chút để thấy mặt đường trôi nhanh hơn
            cam_tf = carla.Transform(carla.Location(x=-5.5, z=2.2), carla.Rotation(pitch=-10.0))
            
        self.sensor = player.get_world().spawn_actor(bp, cam_tf, attach_to=player)
        self.sensor.listen(lambda img: setattr(self, 'surface', pygame.surfarray.make_surface(np.reshape(np.frombuffer(img.raw_data, dtype=np.dtype("uint8")), (img.height, img.width, 4))[:, :, :3][:, :, ::-1].swapaxes(0, 1))))
        
    def destroy(self):
        if self.sensor: self.sensor.destroy()

class IMUSensor:
    def __init__(self, player):
        self.accel = (0.0, 0.0, 0.0); self.gyro = (0.0, 0.0, 0.0)
        self.sensor = player.get_world().spawn_actor(player.get_world().get_blueprint_library().find('sensor.other.imu'), carla.Transform(), attach_to=player)
        self.sensor.listen(lambda data: setattr(self, 'accel', (data.accelerometer.x, data.accelerometer.y, data.accelerometer.z - 9.81)) or setattr(self, 'gyro', (data.gyroscope.x, data.gyroscope.y, data.gyroscope.z)))
    def destroy(self):
        if self.sensor: self.sensor.destroy()

# ==============================================================================
# -- 3. VÒNG LẶP MÔ PHỎNG VẬT LÝ & KẾT XUẤT ------------------------------------
# ==============================================================================

def run_simulation(config):
    pygame.init()
    display = pygame.display.set_mode((1280, 720), pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption("Mô phỏng " + config['mode'])
    client = carla.Client('127.0.0.1', 2000); client.set_timeout(20.0)
    world = client.load_world(config['map'])
    
    is_2d_mode = "2D" in config['mode']
    settings = world.get_settings(); settings.synchronous_mode = True; settings.fixed_delta_seconds = 1.0/60.0
    if is_2d_mode: settings.no_rendering_mode = True 
    world.apply_settings(settings)
    
    dense_route = config['dense_route']; start_loc = dense_route[0]
    sparse_route = [dense_route[0]]
    for pt in dense_route[1:]:
        if sparse_route[-1].distance(pt) >= 1.5: sparse_route.append(pt)
    if sparse_route[-1].distance(dense_route[-1]) > 1.5: sparse_route.append(dense_route[-1])

    start_wp = world.get_map().get_waypoint(start_loc)
    spawn_tf = carla.Transform(start_loc, start_wp.transform.rotation); spawn_tf.location.z += 1.0 
    for actor in world.get_actors().filter('vehicle.*'):
        if actor.get_location().distance(spawn_tf.location) < 5.0: actor.destroy()
    vehicle = world.spawn_actor(world.get_blueprint_library().filter('vehicle.tesla.model3')[0], spawn_tf)
    for _ in range(10): world.tick()

    autopilot = PurePursuitAutopilot(sparse_route, config['speed'], config['profile'])
    
    if is_2d_mode:
        camera = None
        map_renderer = TopDownMap(world.get_map(), dense_route, 1280, 720) 
    else:
        camera = RenderCamera(vehicle, config['camera']) 
        map_renderer = TopDownMap(world.get_map(), dense_route, 320, 320) 
        
    imu = IMUSensor(vehicle); dest_loc = dense_route[-1]
    
    timestamp = datetime.datetime.now().strftime('%H%M%S')
    if not os.path.exists('_logs'): os.makedirs('_logs')
    csv_path = f"_logs/mca_data_{timestamp}.csv"
    f = open(csv_path, 'w', newline=''); writer = csv.writer(f)
    writer.writerow(['Time_s', 'Pos_X', 'Pos_Y', 'Pos_Z', 'Vel_X', 'Vel_Y', 'Vel_Z', 'Acc_X', 'Acc_Y', 'Acc_Z', 'Roll', 'Pitch', 'Yaw', 'AngVel_X', 'AngVel_Y', 'AngVel_Z', 'AngAcc_X', 'AngAcc_Y', 'AngAcc_Z'])
    
    video_writer = None
    video_path = config.get('video_path', None)
    if config['video'] and video_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, 60.0, (1280, 720))

    start_t = time.time(); clock = pygame.time.Clock(); stopping = False
    last_gyro = (0.0, 0.0, 0.0); last_t = 0.0; distance_traveled = 0.0; last_loc = vehicle.get_location()
    
    # --- BIẾN MỚI: ĐẾM GIỜ KHI DỪNG ---
    stop_timer_start = None 

    obey_traffic = "Tuân thủ" in config.get('traffic', "")
    frame_times = []
    last_render_time = time.perf_counter() # Dùng perf_counter để có độ chính xác micro-giây
    try:
        while True:
            world.tick()
            
            # --- FIX ĐỒNG BỘ: ÉP CHẠY ĐÚNG 60 FPS (TƯƠNG ĐƯƠNG 0.01667s/LẦN) ---
            clock.tick_busy_loop(60) 
            
            if not vehicle.is_alive: break
                
            t = time.time() - start_t
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE): return csv_path, video_path if video_writer else None
            
            loc = vehicle.get_location(); vel = vehicle.get_velocity(); rot = vehicle.get_transform().rotation; acc = imu.accel; gyro = imu.gyro
            dt = t - last_t if t > last_t else 0.05
            ang_acc = ((gyro[0]-last_gyro[0])/dt, (gyro[1]-last_gyro[1])/dt, (gyro[2]-last_gyro[2])/dt)
            
            distance_traveled += loc.distance(last_loc)
            last_loc = loc; last_gyro = gyro; last_t = t
            
            writer.writerow([t, loc.x, loc.y, loc.z, vel.x, vel.y, vel.z, acc[0], acc[1], acc[2], rot.roll, rot.pitch, rot.yaw, gyro[0], gyro[1], gyro[2], ang_acc[0], ang_acc[1], ang_acc[2]])
            speed_kmh = 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
            
            if not stopping: 
                red_light = False
                if obey_traffic and vehicle.is_at_traffic_light():
                    traffic_light = vehicle.get_traffic_light()
                    if traffic_light and traffic_light.get_state() == carla.TrafficLightState.Red: red_light = True
                
                if red_light:
                    vehicle.apply_control(carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0))
                    status_text = "ĐÈN ĐỎ - DỪNG"
                else:
                    limit = vehicle.get_speed_limit()
                    vehicle.apply_control(autopilot.run_step(vehicle, dt, limit))
                    status_text = f"Mục tiêu: {autopilot.ref_speed * 3.6:.1f}"
            else:
                status_text = "ĐÃ ĐẾN ĐÍCH - ĐANG DỪNG..."

            if is_2d_mode: 
                display.fill((25, 25, 30))
                map_renderer.render(display, vehicle.get_transform(), position=(0,0))
            else:
                if getattr(camera, 'surface', None) is not None: display.blit(camera.surface, (0, 0))
                map_renderer.render(display, vehicle.get_transform(), position=(1280-320-20, 20))

            try: prog = (min(50, [vehicle.get_location().distance(wp) for wp in dense_route].index(min([vehicle.get_location().distance(wp) for wp in dense_route]))) / len(dense_route)) * 100
            except: prog = 0
            
            overlay = pygame.Surface((1280, 50), pygame.SRCALPHA); overlay.fill((0, 0, 0, 150)); display.blit(overlay, (0, 670))
            pygame.draw.rect(display, (0, 200, 255), (40, 685, int(12 * prog), 12))
            mode_text = "[2D]" if is_2d_mode else "[3D]"
            display.blit(pygame.font.SysFont("Arial", 16, bold=True).render(f"{mode_text} Quãng đường: {distance_traveled:.1f} m | Time: {t:.1f}s | Speed: {speed_kmh:.1f} km/h ({status_text})", True, (255, 255, 255)), (50, 682))
            pygame.display.flip()
            # ĐO ĐỘ TRỄ KHUNG HÌNH (FRAME TIME)
            current_time = time.perf_counter()
            dt_render = current_time - last_render_time
            frame_times.append(dt_render)
            last_render_time = current_time
            if video_writer:
                frame = pygame.surfarray.array3d(display).swapaxes(0, 1) 
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_writer.write(frame)

            # --- THUẬT TOÁN DỪNG VÀ ĐỢI 1 GIÂY ---
            if vehicle.get_location().distance(dest_loc) < 3.0 and not stopping:
                stopping = True
                stop_timer_start = t # Bắt đầu bấm giờ
                vehicle.apply_control(carla.VehicleControl(hand_brake=True, brake=1.0, steer=0.0))
                
            if stopping and speed_kmh < 0.5: 
                if (t - stop_timer_start) >= 1.0: # Xe đã đứng yên đủ 1.0 giây
                    break 
            
        return csv_path, video_path if video_writer else None
    finally:
        f.close()
        if video_writer: video_writer.release()
        try: 
            if camera: camera.destroy()
        except: pass
        try: imu.destroy(); vehicle.destroy()
        except: pass
        settings.no_rendering_mode = False; settings.synchronous_mode = False; world.apply_settings(settings); pygame.quit()
        # --- HIỂN THỊ BÁO CÁO ỔN ĐỊNH KHUNG HÌNH QUA POPUP GUI ---
        if 'frame_times' in locals() and len(frame_times) > 10:
            valid_frames = np.array(frame_times[10:]) # Bỏ qua 10 frame khởi động
            frame_time_ms = valid_frames * 1000.0
            
            mean_ft = np.mean(frame_time_ms)
            std_ft = np.std(frame_time_ms)
            
            eval_text = "RẤT MƯỢT (Đạt chuẩn mô phỏng)" if std_ft < 5.0 else "GIẬT LAG (Nguy cơ gây say xe VR cao)"
            
            report_msg = (
                f"📊 BÁO CÁO ĐỘ ỔN ĐỊNH KHUNG HÌNH (FRAME PACING)\n"
                f"{'-'*50}\n"
                f"• FPS Trung bình:\t\t{1000.0 / mean_ft:.1f} FPS\n"
                f"• Frame Time (Mean):\t{mean_ft:.2f} ms\n"
                f"• Jitter (Độ lệch chuẩn):\t{std_ft:.2f} ms\n"
                f"{'-'*50}\n"
                f"💡 ĐÁNH GIÁ: {eval_text}\n"
                f"(*Tiêu chuẩn Jitter đối với hệ thống MCA/VR là < 5.0 ms)"
            )
            
            # Khởi tạo một cửa sổ Tkinter ẩn để gọi MessageBox
            import tkinter as tk
            from tkinter import messagebox
            tmp_root = tk.Tk(); tmp_root.withdraw()
            tmp_root.attributes('-topmost', True)
            messagebox.showinfo("Báo cáo Hiệu năng Mô phỏng", report_msg)
            tmp_root.destroy()
# ==============================================================================
# -- 4. MENU CHUYỂN TIẾP (CRASH-PROOF ARCHITECTURE) ----------------------------
# ==============================================================================

def run_menu_gui(csv_path, video_path):
    root = tk.Tk()
    root.title("Bước tiếp theo")
    
    w = 400; h = 320
    x = (root.winfo_screenwidth() // 2) - (w // 2)
    y = (root.winfo_screenheight() // 2) - (h // 2)
    root.geometry(f'{w}x{h}+{x}+{y}')
    
    root.attributes('-topmost', True); root.lift(); root.focus_force()
    
    action_result = ["exit"]
    def set_action(val):
        action_result[0] = val
        root.quit()

    root.protocol("WM_DELETE_WINDOW", lambda: set_action("exit"))
    
    ttk.Label(root, text="Mô phỏng hoàn tất!", font=("Arial", 12, "bold")).pack(pady=10)
    msg = f"Dữ liệu 18-DOF:\n{csv_path}"
    if video_path: msg += f"\n\nVideo MP4:\n{video_path}"
    ttk.Label(root, text=msg, foreground="green", justify="center").pack(pady=(0,10))
    
    ttk.Button(root, text="📊 Chuyển sang Phân tích Dữ liệu (DSP)", command=lambda: set_action("dsp")).pack(pady=5, fill="x", padx=40)
    ttk.Button(root, text="🔄 Quay lại Bảng Cấu Hình Mô phỏng", command=lambda: set_action("restart")).pack(pady=5, fill="x", padx=40)
    ttk.Button(root, text="❌ Kết thúc Chương trình", command=lambda: set_action("exit")).pack(pady=5, fill="x", padx=40)
    
    root.mainloop()
    try: root.destroy()
    except: pass
    return action_result[0]

# ==============================================================================
# -- 5. BỘ XỬ LÝ TÍN HIỆU CHUYÊN NGHIỆP (DSP - CÓ CẮT VIDEO ĐỒNG BỘ) ------------
# ==============================================================================

def run_dsp_gui(csv_path, video_path=None):
    root = tk.Tk() 
    root.title("HUST MCA - Phân tích & Chuẩn hóa Dữ liệu (DSP Pipeline)")
    root.geometry("850x900")
    
    def safe_close(): root.quit()
    root.protocol("WM_DELETE_WINDOW", safe_close)
    
    df_original = pd.read_csv(csv_path); df_current = df_original.copy()
    
    left_frame = ttk.Frame(root); left_frame.pack(side="left", fill="y", padx=10, pady=10)
    right_frame = ttk.Frame(root); right_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)
    
    ttk.Label(left_frame, text="1. Chọn Tín hiệu (Giữ Ctrl):", font=('Arial', 10, 'bold')).pack(anchor="w", pady=(0,5))
    scroll = tk.Scrollbar(left_frame); scroll.pack(side="right", fill="y")
    listbox = tk.Listbox(left_frame, selectmode=tk.MULTIPLE, yscrollcommand=scroll.set, width=25, height=48, font=('Arial', 10))
    
    signal_cols = [c for c in df_original.columns if c != 'Time_s']
    for col in signal_cols: listbox.insert(tk.END, col)
    listbox.pack(side="left", fill="y"); scroll.config(command=listbox.yview)

    # --- TÍNH NĂNG CŨ: CẮT THỦ CÔNG ---
    frame_trim = ttk.LabelFrame(right_frame, text="Tính năng cũ: Cắt thủ công (Trim)")
    frame_trim.pack(fill="x", pady=5)
    t_start = ttk.Entry(frame_trim, width=10, justify='center'); t_start.insert(0, "0.0"); t_start.grid(row=0, column=1, pady=5)
    t_end = ttk.Entry(frame_trim, width=10, justify='center'); t_end.insert(0, f"{df_original['Time_s'].max():.1f}"); t_end.grid(row=0, column=3, pady=5)
    
    def trim_data():
        nonlocal df_current
        try:
            t_s = float(t_start.get()); t_e = float(t_end.get())
            mask = (df_original['Time_s'] >= t_s) & (df_original['Time_s'] <= t_e)
            df_current = df_original[mask].reset_index(drop=True)
            messagebox.showinfo("Thành công", f"Đã cắt. Còn lại: {len(df_current)} mẫu.")
        except Exception as e: messagebox.showerror("Lỗi Cắt dữ liệu", str(e))
    ttk.Button(frame_trim, text="✂️ Cắt Dữ Liệu", command=trim_data).grid(row=0, column=4, padx=15, pady=5)

    # --- TÍNH NĂNG CŨ: VẼ ĐỒ THỊ & LỌC ---
    frame_plot = ttk.LabelFrame(right_frame, text="2. Hiển thị & Phân tích Phổ")
    frame_plot.pack(fill="x", pady=5)
    def get_selected(): return [signal_cols[i] for i in listbox.curselection()]

    def plot_signals():
        cols = get_selected() 
        if cols:
            plt.figure(figsize=(10, 4)); plt.title("Miền Thời Gian")
            for col in cols: plt.plot(df_current['Time_s'].values, df_current[col].values, label=col)
            plt.grid(); plt.legend(); plt.show()
    ttk.Button(frame_plot, text="📈 Vẽ Tín Hiệu", command=plot_signals).pack(side="left", expand=True, fill="x", padx=5, pady=5)
    
    def plot_fft():
        cols = get_selected()
        if cols:
            fs = float(fs_entry.get())
            plt.figure(figsize=(10, 4)); plt.title("Phân tích Phổ (FFT)")
            for col in cols:
                data = df_current[col].values; N = len(data); yf = fft(data); xf = fftfreq(N, 1/fs)[:N//2]
                plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]), label=f"FFT {col}")
            plt.grid(); plt.legend(); plt.show()
    ttk.Button(frame_plot, text="📊 Phân Tích Phổ", command=plot_fft).pack(side="right", expand=True, fill="x", padx=5, pady=5)

    # --- TÍNH NĂNG MỚI: DSP PIPELINE MODULE HÓA ---
    frame_filter = ttk.LabelFrame(right_frame, text="3. DSP Pipeline (Tách rời & Tổ hợp)")
    frame_filter.pack(fill="x", pady=5)
    
    param_f = ttk.Frame(frame_filter); param_f.pack(fill="x", pady=5)
    ttk.Label(param_f, text="Tần số lấy mẫu (Hz):").grid(row=0, column=0, padx=5, pady=2, sticky='e')
    fs_entry = ttk.Entry(param_f, width=10, justify='center'); fs_entry.insert(0, "60.0"); fs_entry.grid(row=0, column=1, padx=5)
    ttk.Label(param_f, text="Bậc bộ lọc:").grid(row=0, column=2, padx=5, pady=2, sticky='e')
    order_entry = ttk.Entry(param_f, width=10, justify='center'); order_entry.insert(0, "4"); order_entry.grid(row=0, column=3, padx=5)
    
    ttk.Label(param_f, text="Cắt Low-pass (Hz):").grid(row=1, column=0, padx=5, pady=2, sticky='e')
    fc1_entry = ttk.Entry(param_f, width=10, justify='center'); fc1_entry.insert(0, "2.0"); fc1_entry.grid(row=1, column=1, padx=5)
    ttk.Label(param_f, text="Cắt nhiễu Biên (s):").grid(row=1, column=2, padx=5, pady=2, sticky='e')
    trim_entry = ttk.Entry(param_f, width=10, justify='center'); trim_entry.insert(0, "0.3"); trim_entry.grid(row=1, column=3, padx=5)

    # --- HÀM THỰC THI TỪNG BƯỚC ---
    def step1_noise_filter(silent=False):
        nonlocal df_current
        cols = get_selected()
        if not cols: 
            if not silent: messagebox.showwarning("Cảnh báo", "Chọn tín hiệu bên trái trước.")
            return False
        try:
            fs = float(fs_entry.get()); order = int(order_entry.get()); fc = float(fc1_entry.get())
            b, a = butter(order, fc / (0.5 * fs), btype='low')
            
            for col in cols:
                raw_data = df_current[col].values
                time_data = df_current['Time_s'].values
                
                # BỔ SUNG: SAN PHẲNG XUNG "RƠI TỰ DO" TRƯỚC KHI ĐƯA VÀO BỘ LỌC
                # Tìm giá trị tĩnh trong khoảng 0.5s đến 2.0s
                idle_mask = (time_data > 0.5) & (time_data <= 2.0)
                if not idle_mask.any(): idle_mask = time_data <= float(trim_entry.get())
                dc_val = np.median(raw_data[idle_mask]) if idle_mask.any() else raw_data[0]
                
                # Ghi đè 0.5s đầu tiên bằng giá trị tĩnh để "bóp chết" xung nhiễu
                cleaned_data = np.copy(raw_data)
                cleaned_data[time_data < 0.5] = dc_val
                
                # Đưa dữ liệu đã dọn sạch vào bộ lọc Zero-phase
                df_current[col] = filtfilt(b, a, cleaned_data)
                
            if not silent: messagebox.showinfo("Hoàn tất Bước 1", f"Đã san phẳng nhiễu khởi tạo & Lọc Low-pass {fc}Hz.\nHãy bấm 'Vẽ Tín Hiệu' để xem đường cong đã mượt chưa.")
            return True
        except Exception as e:
            if not silent: messagebox.showerror("Lỗi Lọc Nhiễu", str(e))
            return False

    def step2_sensor_calibration(silent=False):
        nonlocal df_current
        cols = get_selected()
        if not cols: 
            if not silent: messagebox.showwarning("Cảnh báo", "Chọn tín hiệu bên trái trước.")
            return False
        try:
            trim_time = float(trim_entry.get())
            for col in cols:
                raw_data = df_current[col].values
                time_data = df_current['Time_s'].values
                # Tính trung vị tĩnh, tránh giai đoạn rơi tự do
                idle_mask = (time_data > 0.5) & (time_data <= 2.0)
                if not idle_mask.any(): idle_mask = time_data <= trim_time
                dc_offset = np.median(raw_data[idle_mask]) if idle_mask.any() else raw_data[0]
                df_current[col] = raw_data - dc_offset
            if not silent: messagebox.showinfo("Hoàn tất Bước 2", "Đã hiệu chuẩn Cảm biến (Khử DC Offset).\nTrục Y đã được kéo về mức 0.")
            return True
        except Exception as e:
            if not silent: messagebox.showerror("Lỗi Hiệu chuẩn", str(e))
            return False

    def step3_truncate_sync(silent=False):
        nonlocal df_current
        cols = get_selected()
        if not cols: return False
        try:
            trim_time = float(trim_entry.get())
            valid_mask = df_current['Time_s'] >= trim_time
            if not valid_mask.any(): raise ValueError("Thời gian cắt quá lớn.")
            
            df_current = df_current[valid_mask].reset_index(drop=True)
            df_current['Time_s'] = df_current['Time_s'] - df_current['Time_s'].iloc[0]
            
            msg = f"Đã cắt bỏ {trim_time}s nhiễu quá độ và kéo gốc thời gian về t=0s."

            if video_path and os.path.exists(video_path) and HAS_CV2:
                import cv2
                cap = cv2.VideoCapture(video_path)
                video_fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
                start_frame = int(trim_time * video_fps)
                
                trimmed_video_path = video_path.replace(".mp4", "_MCA_SYNC.mp4")
                out = cv2.VideoWriter(trimmed_video_path, cv2.VideoWriter_fourcc(*'mp4v'), video_fps, 
                                      (int(cap.get(3)), int(cap.get(4))))
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                while True:
                    ret, frame = cap.read()
                    if not ret: break
                    out.write(frame)
                cap.release(); out.release()
                msg += f"\n\nĐã cắt đồng bộ Video:\n{trimmed_video_path}"
                
            if not silent: messagebox.showinfo("Hoàn tất Bước 3", msg)
            return True
        except Exception as e:
            if not silent: messagebox.showerror("Lỗi Đồng bộ", str(e))
            return False

    def run_all_pipeline():
        if step1_noise_filter(silent=True):
            if step2_sensor_calibration(silent=True):
                if step3_truncate_sync(silent=True):
                    messagebox.showinfo("Hoàn tất Tổ hợp", "Đã chạy trơn tru 3 bước:\n\n1. Lọc nhiễu Low-pass\n2. Hiệu chuẩn DC Offset\n3. Truncate & Sync Video")

    # --- BỐ TRÍ NÚT BẤM (GRID LAYOUT) ---
    btn_grid = ttk.Frame(frame_filter)
    btn_grid.pack(fill="x", padx=10, pady=5)
    btn_grid.columnconfigure(0, weight=1); btn_grid.columnconfigure(1, weight=1); btn_grid.columnconfigure(2, weight=1)
    
    ttk.Button(btn_grid, text="1. Lọc Nhiễu (Filtfilt)", command=step1_noise_filter).grid(row=0, column=0, padx=2, pady=2, sticky="ew")
    ttk.Button(btn_grid, text="2. Khử DC (Hiệu chuẩn)", command=step2_sensor_calibration).grid(row=0, column=1, padx=2, pady=2, sticky="ew")
    ttk.Button(btn_grid, text="3. Cắt & Đồng Bộ", command=step3_truncate_sync).grid(row=0, column=2, padx=2, pady=2, sticky="ew")
    
    ttk.Button(frame_filter, text="🚀 CHẠY TỔ HỢP CẢ 3 BƯỚC", command=run_all_pipeline, style="Accent.TButton").pack(fill="x", padx=10, pady=5)

    # --- KHỐI 4: QUẢN LÝ ---
    frame_action = ttk.LabelFrame(right_frame, text="4. Quản lý Dữ liệu")
    frame_action.pack(fill="x", pady=5)
    def reset_data():
        nonlocal df_current; df_current = df_original.copy()
        messagebox.showinfo("Thành công", "Đã khôi phục dữ liệu gốc!")
    ttk.Button(frame_action, text="🔄 Reset (Khôi phục)", command=reset_data).pack(side="left", expand=True, fill="x", padx=5, pady=10)
    
    def save_data():
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")], initialfile="mca_ready_data.csv")
        if path: df_current.to_csv(path, index=False); messagebox.showinfo("Thành công", f"Đã lưu:\n{path}")
    ttk.Button(frame_action, text="💾 Lưu File CSV (Đã Chuẩn Hóa)", command=save_data).pack(side="right", expand=True, fill="x", padx=5, pady=10)

    root.mainloop()
    try: root.destroy()
    except: pass

# ==============================================================================
# -- 6. MAIN WORKFLOW LOOP -----------------------------------------------------
# ==============================================================================

if __name__ == '__main__':
    while True:
        config = run_setup_gui()
        if not config: break 
            
        csv_file, video_file = run_simulation(config)
        if not csv_file or not os.path.exists(csv_file): break

        while True:
            action = run_menu_gui(csv_file, video_file)
            if action == "dsp": 
                run_dsp_gui(csv_file, video_file) 
            elif action == "restart": 
                break 
            else: 
                sys.exit(0)
