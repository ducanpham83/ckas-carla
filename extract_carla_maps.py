import carla
import json
import math

def extract_all_maps():
    print("🚀 BẮT ĐẦU TRÍCH XUẤT BẢN ĐỒ HD TỪ CARLA...")
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(60.0)
    
    towns = ["Town01", "Town02", "Town03", "Town04", "Town05", "Town06", "Town07", "Town10HD"]
    offline_data = {}

    for town in towns:
        print(f"⏳ Đang xử lý {town}...")
        try:
            # Load World sạch sẽ
            world = client.load_world(town, reset_settings=True)
            carla_map = world.get_map()
            
            # 1. Trích xuất Spawn Points
            spawns = []
            for sp in carla_map.get_spawn_points():
                spawns.append({
                    'x': round(sp.location.x, 3), 'y': round(sp.location.y, 3), 'z': round(sp.location.z, 3),
                    'pitch': round(sp.rotation.pitch, 3), 'yaw': round(sp.rotation.yaw, 3), 'roll': round(sp.rotation.roll, 3)
                })
            
            # 2. Trích xuất Topology (Đường cong độ phân giải cao 2.0m)
            topology_high_res = []
            for wp1, wp2 in carla_map.get_topology():
                path = [[round(wp1.transform.location.x, 3), round(wp1.transform.location.y, 3), round(wp1.transform.location.z, 3)]]
                wp_curr = wp1
                
                # Nội suy các điểm cong giữa 2 ngã tư
                while wp_curr.transform.location.distance(wp2.transform.location) > 2.0:
                    next_wps = wp_curr.next(2.0)
                    if not next_wps: break
                    wp_curr = next_wps[0]
                    path.append([round(wp_curr.transform.location.x, 3), round(wp_curr.transform.location.y, 3), round(wp_curr.transform.location.z, 3)])
                
                path.append([round(wp2.transform.location.x, 3), round(wp2.transform.location.y, 3), round(wp2.transform.location.z, 3)])
                topology_high_res.append(path)

            offline_data[town] = {
                'spawn_points': spawns,
                'topology': topology_high_res
            }
            print(f"   ✅ Đã xong {town}: {len(spawns)} Spawns, {len(topology_high_res)} Segments.")
            
        except Exception as e:
            print(f"   ❌ Lỗi tải {town}: {e}")

    # Ghi ra file JSON
    with open("carla_offline_maps.json", "w", encoding='utf-8') as f:
        json.dump(offline_data, f)
    print("\n🎉 TRÍCH XUẤT HOÀN TẤT! Đã lưu thành 'carla_offline_maps.json'.")

if __name__ == '__main__':
    extract_all_maps()
