import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import urllib.request
import os
import matplotlib.font_manager as fm

# =====================================================================
# 🛠️ 1. การตั้งค่าทรัพยากรและฟอนต์ภาษาไทย (Font Configuration)
# =====================================================================
@st.cache_resource
def setup_thai_font():
    font_url = "https://github.com/Phonbopit/sarabun-webfont/raw/master/fonts/thsarabunnew-webfont.ttf"
    font_path = "thsarabunnew-webfont.ttf"
    if not os.path.exists(font_path):
        urllib.request.urlretrieve(font_url, font_path)
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'TH Sarabun New'
    plt.rcParams['axes.unicode_minus'] = False

setup_thai_font()

# =====================================================================
# ⚙️ 2. ฐานข้อมูลตั้งต้นสำหรับทดสอบ (SUT Default Data)
# =====================================================================
DEFAULT_DATA = [
    ("Depot โรงจัดการขยะ", 14.862939, 102.027903, 0.0),
    ("ภูมิทัศน์(ใหม่)", 14.86903, 102.02135, 0.3),
    ("สวนพฤกษศาสตร์", 14.86991, 102.022113, 0.3),
    ("อุทยานผีเสื้อ", 14.871074, 102.022713, 0.3),
    ("ซินโครตรอน", 14.872731, 102.023232, 0.1),
    ("อาคารสุรพัฒน์ 2", 14.8754, 102.02286, 0.2),
    ("เซเว่น-อีเลฟเว่น เทคโนธานี", 14.876072, 102.022745, 0.7),
    ("หอดูดาว", 14.87414, 102.027598, 0.2),
    ("กาญจนาภิเษก", 14.873602, 102.026147, 0.5),
    ("อุทยานวิทยาศาสตร์", 14.87176, 102.01974, 0.3),
    ("เครื่องมือฯ9", 14.87516, 102.01613, 0.1),
    ("เครื่องมือฯ10", 14.876915, 102.015231, 0.5),
    ("เครื่องมือฯ6", 14.875158, 102.017524, 0.5),
    ("อาคารบริหาร", 14.88013, 102.02042, 0.4),
    ("สุรนิเวศ7", 14.89713, 102.011243, 0.2),
    ("โรงอาหารกาสะลองคำ", 14.896759, 102.012427, 0.5)
]

# =====================================================================
# 📡 3. การเชื่อมต่อโครงข่ายทางภูมิศาสตร์ (OSRM API Integration)
# =====================================================================
@st.cache_data(show_spinner=False)
def get_distance_matrix(locations):
    """ฟังก์ชันสร้าง Distance Matrix จากระยะทางขับขี่จริงบนถนน"""
    N = len(locations)
    distance_matrix = np.zeros((N, N))
    CHUNK_SIZE = 50
    coords = [(item[2], item[1]) for item in locations]
    
    for i in range(0, N, CHUNK_SIZE):
        for j in range(0, N, CHUNK_SIZE):
            src_chunk = coords[i:i+CHUNK_SIZE]
            dst_chunk = coords[j:j+CHUNK_SIZE]
            combined_coords = src_chunk + dst_chunk
            coords_string = ";".join([f"{lon},{lat}" for lon, lat in combined_coords])
            
            num_src = len(src_chunk)
            num_dst = len(dst_chunk)
            sources_str = ";".join([str(x) for x in range(num_src)])
            destinations_str = ";".join([str(x) for x in range(num_src, num_src + num_dst)])
            
            url = f"http://router.project-osrm.org/table/v1/driving/{coords_string}?sources={sources_str}&destinations={destinations_str}&annotations=distance"
            try:
                response = requests.get(url)
                data = response.json()
                if data.get("code") == "Ok":
                    distance_matrix[i:i+num_src, j:j+num_dst] = np.array(data["distances"])
            except Exception as e:
                st.error(f"OSRM API Error (Matrix): {e}")
            time.sleep(0.5)
            
    return pd.DataFrame(distance_matrix) / 1000.0

def get_osrm_route_geometry(route_seq, coords, depot):
    """ดึงพิกัดจุดเลี้ยว (Shapepoints) ตามแนวถนนจริงจาก OSRM Route API"""
    full_route = [depot] + route_seq + [depot]
    route_coords = [coords[n] for n in full_route]
    
    coords_string = ";".join([f"{lon},{lat}" for lon, lat in route_coords])
    url = f"http://router.project-osrm.org/route/v1/driving/{coords_string}?overview=full&geometries=geojson"
    
    try:
        response = requests.get(url)
        data = response.json()
        if data.get("code") == "Ok":
            return data["routes"][0]["geometry"]["coordinates"]
    except Exception as e:
        pass
    return route_coords

# =====================================================================
# 🧠 4. อัลกอริทึมการจัดเส้นทาง (Optimization Engines)
# =====================================================================
def run_savings_algorithm(df_dist, demands, nodes, max_capacity):
    depot = nodes[0]
    customers = nodes[1:]
    
    savings = []
    for i in customers:
        for j in customers:
            if i != j:
                s_ij = df_dist.loc[i, depot] + df_dist.loc[depot, j] - df_dist.loc[i, j]
                if s_ij > 0:
                    savings.append((s_ij, i, j))
    savings.sort(key=lambda x: x[0], reverse=True)

    routes = [[c] for c in customers]
    route_vols = [demands[c] for c in customers]

    def get_route_idx(node):
        for idx, r in enumerate(routes):
            if node in r: return idx
        return -1

    for s_ij, i, j in savings:
        idx_i = get_route_idx(i)
        idx_j = get_route_idx(j)
        if idx_i != idx_j and idx_i != -1 and idx_j != -1:
            if routes[idx_i][-1] == i and routes[idx_j][0] == j:
                if route_vols[idx_i] + route_vols[idx_j] <= max_capacity:
                    routes[idx_i].extend(routes[idx_j])
                    route_vols[idx_i] += route_vols[idx_j]
                    routes.pop(idx_j)
                    route_vols.pop(idx_j)
                    
    return routes, route_vols

def run_sweep_algorithm(locations, demands, nodes, max_capacity):
    depot = nodes[0]
    depot_lat, depot_lon = locations[0][1], locations[0][2]
    
    customer_angles = []
    for i, item in enumerate(locations[1:]):
        node_name = item[0]
        lat, lon = item[1], item[2]
        angle = math.degrees(math.atan2(lat - depot_lat, lon - depot_lon))
        if angle < 0: angle += 360
        customer_angles.append({"node": node_name, "angle": angle, "vol": demands[node_name]})
        
    customer_angles.sort(key=lambda x: x['angle'])
    
    routes = []
    route_vols = []
    current_route = []
    current_vol = 0.0
    
    for c in customer_angles:
        if current_vol + c['vol'] <= max_capacity:
            current_route.append(c['node'])
            current_vol += c['vol']
        else:
            routes.append(current_route)
            route_vols.append(current_vol)
            current_route = [c['node']]
            current_vol = c['vol']
            
    if current_route:
        routes.append(current_route)
        route_vols.append(current_vol)
        
    return routes, route_vols

# =====================================================================
# 🎨 5. โมดูลแสดงผลแผนที่ (Data Visualization Module)
# =====================================================================
def plot_routes(routes, locations, nodes, title, grand_total_distance):
    depot = nodes[0]
    coords = {item[0]: (item[2], item[1]) for item in locations} 
    
    fig, ax = plt.subplots(figsize=(12, 8))
    cmap = cm.get_cmap('tab20', max(20, len(routes)))

    for trip_idx, route_seq in enumerate(routes):
        route_color = cmap(trip_idx % 20)
        road_coords = get_osrm_route_geometry(route_seq, coords, depot)
        x_vals = [pt[0] for pt in road_coords]
        y_vals = [pt[1] for pt in road_coords]
        
        ax.plot(x_vals, y_vals, color=route_color, linewidth=2.5, alpha=0.8, label=f'Trip {trip_idx+1}')

        for k in range(0, len(x_vals) - 1, 15):
            ax.annotate('', xy=(x_vals[k+1], y_vals[k+1]), xytext=(x_vals[k], y_vals[k]),
                         arrowprops=dict(arrowstyle="->", color=route_color, lw=1.5, alpha=0.7))
        
        time.sleep(0.2)

    all_x = [coords[n][0] for n in nodes[1:]]
    all_y = [coords[n][1] for n in nodes[1:]]
    ax.scatter(all_x, all_y, color='dimgray', zorder=5, s=25)
    ax.scatter(coords[depot][0], coords[depot][1], color='red', marker='*', s=350, zorder=10, label='Depot')

    for node, (x, y) in coords.items():
        if node == depot:
            ax.text(x, y + 0.0004, 'DEPOT (บ่อขยะ)', fontsize=10, fontweight='bold', color='red', ha='center')
        else:
            short_name = str(node).replace(' จุดที่ ', '-')
            ax.text(x + 0.0001, y + 0.0001, short_name, fontsize=8, color='black', alpha=0.8)

    ax.set_title(f'{title}\n[ระบบแสดงผลอิงตามโครงข่ายถนนจริงบนระบบ GIS]', fontsize=16, fontweight='bold')
    ax.set_xlabel('Longitude (พิกัด X)', fontsize=12)
    ax.set_ylabel('Latitude (พิกัด Y)', fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10, title="ลำดับรอบวิ่ง (Trip No.)")
    
    return fig

# =====================================================================
# 🖥️ 6. หน้าจอผู้ใช้งาน (Streamlit UI)
# =====================================================================
st.set_page_config(page_title="Smart Waste Collection CVRP", layout="wide")
st.title("🚛 Smart Waste Collection Routing System")
st.markdown("ระบบวิเคราะห์และแสดงผลลัพธ์การจัดเส้นทางแบบปรับเปลี่ยนตัวแปรได้ (Multi-Vehicle DSS) พร้อมโครงข่ายถนนจริง")

# --- แถบเครื่องมือด้านข้าง ---
with st.sidebar:
    st.header("⚙️ 1. ปรับแต่งยานพาหนะ (Fleet)")
    max_vehicles = st.number_input("จำนวนรถขยะที่มีในระบบ (คัน)", min_value=1, value=2, step=1)
    max_capacity = st.number_input("ความจุสูงสุดของรถ (ลบ.ม. / คัน)", min_value=1.0, value=4.5, step=0.5)
    
    st.header("⚙️ 2. เลือกอัลกอริทึม")
    algorithm_choice = st.selectbox("เทคนิคการจัดเส้นทาง", ("Clarke-Wright Savings", "Sweep Algorithm"))
    
    st.header("📂 3. วิธีการนำเข้าข้อมูลพิกัด")
    data_mode = st.radio(
        "เลือกรูปแบบข้อมูลที่ต้องการใช้งาน:",
        ("ใช้ข้อมูลทดสอบ มทส. (SUT Sample Data)", "กรอกข้อมูลใหม่เองทั้งหมด (Manual Entry)", "อัปโหลดไฟล์ Excel/CSV")
    )
    
    uploaded_file = None
    if data_mode == "อัปโหลดไฟล์ Excel/CSV":
        uploaded_file = st.file_uploader("อัปโหลดไฟล์ Excel/CSV", type=["xlsx", "csv"])

# --- ส่วนจัดการตารางข้อมูล (Dynamic Data Editor) ---
st.subheader("📝 ตารางจัดการข้อมูลพิกัดและปริมาณขยะ (Data Editor)")

if data_mode == "ใช้ข้อมูลทดสอบ มทส. (SUT Sample Data)":
    df_input = pd.DataFrame(DEFAULT_DATA, columns=["Node_Name", "Latitude", "Longitude", "Demand"])
    st.info("💡 กำลังใช้งานข้อมูลจำลองในพื้นที่ มทส. คุณสามารถดับเบิ้ลคลิกแก้ไขตัวเลข Demand หรือพิกัดในตารางได้ทันที")

elif data_mode == "กรอกข้อมูลใหม่เองทั้งหมด (Manual Entry)":
    # สร้างโครงสร้างข้อมูลเริ่มต้นแถวเดียวเพื่อให้ระบบล็อก Data Type ไว้ (ป้องกัน Error)
    df_input = pd.DataFrame([
        ("Depot โรงจัดการขยะหลัก", 14.862939, 102.027903, 0.0),
        ("จุดเก็บขยะตัวอย่างที่ 1", 14.869030, 102.021350, 0.5)
    ], columns=["Node_Name", "Latitude", "Longitude", "Demand"])
    st.success("💡 โหมดกรอกข้อมูลเอง: แถวแรกสุดจะถูกกำหนดให้เป็นศูนย์กลาง (Depot) เสมอ คุณสามารถกดปุ่ม ➕ ที่ท้ายตารางเพื่อเพิ่มจุดเก็บขยะใหม่ได้ตามต้องการ")

else: # โหมดอัปโหลดไฟล์
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df_input = pd.read_csv(uploaded_file)
        else:
            df_input = pd.read_excel(uploaded_file)
    else:
        st.warning("⚠️ กรุณาทำการอัปโหลดไฟล์ Excel หรือ CSV ที่แถบเมนูด้านข้างก่อนดำเนินการครับ")
        st.stop()

# เปิดฟังก์ชันตารางอัจฉริยะ ให้เพิ่ม/ลบ/แก้ไข แถวข้อมูลได้แบบเรียลไทม์
edited_df = st.data_editor(df_input, num_rows="dynamic", use_container_width=True)
start_btn = st.button("🚀 ยืนยันข้อมูลและเริ่มการประมวลผล (Start Optimization)", type="primary")

# =====================================================================
# 🚀 7. ส่วนประมวลผลหลัก (Execution Block)
# =====================================================================
if start_btn:
    data_to_use = edited_df.values.tolist()
    nodes = edited_df["Node_Name"].tolist()
    demands = dict(zip(edited_df["Node_Name"], edited_df["Demand"]))
    
    # ดักข้อผิดพลาดเชิงโลจิสติกส์ขั้นพื้นฐาน
    if len(nodes) < 2:
        st.error("❌ เกิดข้อผิดพลาด: ต้องมีจุดข้อมูลอย่างน้อย 2 จุดขึ้นไป (Depot 1 จุด และจุดเก็บขยะอย่างน้อย 1 จุด)")
        st.stop()
        
    max_single_demand = max(demands.values())
    if max_single_demand > max_capacity:
        st.error(f"❌ พบข้อผิดพลาด: มีจุดเก็บขยะบางจุดที่มีปริมาณขยะ ({max_single_demand} ลบ.ม.) เกินกว่าความจุของตัวรถที่กำหนดไว้ ({max_capacity} ลบ.ม.)")
        st.stop()

    osrm_input_format = [(row[0], row[1], row[2], row[3]) for row in data_to_use]

    with st.spinner("📡 กำลังคำนวณระยะทางขับขี่จริงระหว่างคู่จุดจอดจาก OSRM API..."):
        df_dist = get_distance_matrix(osrm_input_format)
        df_dist.columns = nodes
        df_dist.index = nodes

    with st.spinner(f"⚙️ กำลังประมวลผลการจัดเส้นทางด้วย {algorithm_choice}..."):
        if algorithm_choice == "Clarke-Wright Savings":
            routes, route_vols = run_savings_algorithm(df_dist, demands, nodes, max_capacity)
        elif algorithm_choice == "Sweep Algorithm":
            routes, route_vols = run_sweep_algorithm(osrm_input_format, demands, nodes, max_capacity)
            
        grand_total_distance = 0.0
        grand_total_volume = sum(route_vols)
        
        for r in routes:
            full_route = [nodes[0]] + r + [nodes[0]]
            dist = 0
            for k in range(len(full_route)-1):
                dist += df_dist.loc[full_route[k], full_route[k+1]]
            grand_total_distance += dist
            
        emission_factor = 0.3 
        carbon_emitted = grand_total_distance * emission_factor
        
        st.success("✅ ออปติไมซ์คำตอบและออกแบบเส้นทางเสร็จสมบูรณ์!")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("รอบวิ่งที่ต้องใช้ (Trips)", f"{len(routes)} เที่ยว")
        col2.metric("ระยะทางขับขี่จริงรวม", f"{grand_total_distance:.2f} กม.")
        col3.metric("ปริมาตรขยะที่เก็บขน", f"{grand_total_volume:.2f} ลบ.ม.")
        col4.metric("คาร์บอนฟุตพริ้นท์ (CO₂)", f"{carbon_emitted:.2f} กก.")
        
        with st.spinner("🗺️ กำลังเรนเดอร์กราฟิกแผนที่โครงข่ายถนน..."):
            fig = plot_routes(routes, osrm_input_format, nodes, f"แผนภาพจำลองเส้นทางจริง ({algorithm_choice})", grand_total_distance)
            st.pyplot(fig)
        
        # -----------------------------------------------------------------
        # 📋 โมดูลการจัดสรรตารางงานให้รถ (Multi-Vehicle Dispatch Schedule)
        # -----------------------------------------------------------------
        st.markdown("### 📋 ตารางการปฏิบัติงานแยกตามยานพาหนะ (Fleet Dispatch Schedule)")
        
        if len(routes) > max_vehicles:
             st.info(f"💡 ข้อสังเกต: ระบบมีรถ {max_vehicles} คัน แต่มีภารกิจทั้งหมด {len(routes)} รอบวิ่ง ดังนั้นรถบางคันจะต้องวิ่งมากกว่า 1 รอบ (Multi-Trip)")

        fleet_schedule = {f"🚛 รถขยะคันที่ {i+1}": [] for i in range(int(max_vehicles))}
        
        for i, r in enumerate(routes):
            vehicle_idx = i % int(max_vehicles)
            vehicle_name = f"🚛 รถขยะคันที่ {vehicle_idx + 1}"
            
            trip_info = {
                "trip_sequence": (i // int(max_vehicles)) + 1,
                "route": r,
                "vol": route_vols[i]
            }
            fleet_schedule[vehicle_name].append(trip_info)
        
        for vehicle_name, trips in fleet_schedule.items():
            total_vehicle_vol = sum([t['vol'] for t in trips])
            
            with st.expander(f"{vehicle_name} (รับผิดชอบ {len(trips)} เที่ยววิ่ง | เก็บขยะรวม {total_vehicle_vol:.2f} ลบ.ม.)", expanded=True):
                if len(trips) == 0:
                    st.write("✅ รถคันนี้ไม่มีภารกิจในรอบการทำงานนี้")
                else:
                    for t in trips:
                        st.write(f"**รอบวิ่งที่ {t['trip_sequence']} ของรถคันนี้** (ปริมาตรขยะประจำเที่ยว: {t['vol']:.2f} / {max_capacity} ลบ.ม.)")
                        st.info(f"📍 ศูนย์กลาง Depot ➡️ {' ➡️ '.join(t['route'])} ➡️ ศูนย์กลาง Depot")
