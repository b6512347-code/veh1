import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import math
import folium
from folium import plugins
from streamlit_folium import st_folium
import urllib.request
import os
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

# =====================================================================
# 🛠️ 1. การตั้งค่าทรัพยากรและฟอนต์
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
# 📡 3. การเชื่อมต่อโครงข่ายถนน (OSRM API Integration)
# =====================================================================
@st.cache_data(show_spinner=False)
def get_distance_matrix(locations):
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
            except Exception:
                pass
            time.sleep(0.5)
            
    return pd.DataFrame(distance_matrix) / 1000.0

def get_osrm_route_geometry(route_seq, coords, depot):
    full_route = [depot] + route_seq + [depot]
    route_coords = [coords[n] for n in full_route]
    coords_string = ";".join([f"{lon},{lat}" for lon, lat in route_coords])
    url = f"http://router.project-osrm.org/route/v1/driving/{coords_string}?overview=full&geometries=geojson"
    
    try:
        response = requests.get(url)
        data = response.json()
        if data.get("code") == "Ok":
            return data["routes"][0]["geometry"]["coordinates"]
    except Exception:
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
                if s_ij > 0: savings.append((s_ij, i, j))
    savings.sort(key=lambda x: x[0], reverse=True)

    routes = [[c] for c in customers]
    route_vols = [demands[c] for c in customers]

    def get_route_idx(node):
        for idx, r in enumerate(routes):
            if node in r: return idx
        return -1

    for s_ij, i, j in savings:
        idx_i, idx_j = get_route_idx(i), get_route_idx(j)
        if idx_i != idx_j and idx_i != -1 and idx_j != -1:
            if routes[idx_i][-1] == i and routes[idx_j][0] == j:
                if route_vols[idx_i] + route_vols[idx_j] <= max_capacity:
                    routes[idx_i].extend(routes[idx_j])
                    route_vols[idx_i] += route_vols[idx_j]
                    routes.pop(idx_j)
                    route_vols.pop(idx_j)
    return routes, route_vols

def run_sweep_algorithm(locations, demands, nodes, max_capacity):
    depot, depot_lat, depot_lon = nodes[0], locations[0][1], locations[0][2]
    customer_angles = []
    for i, item in enumerate(locations[1:]):
        node_name, lat, lon = item[0], item[1], item[2]
        angle = math.degrees(math.atan2(lat - depot_lat, lon - depot_lon))
        if angle < 0: angle += 360
        customer_angles.append({"node": node_name, "angle": angle, "vol": demands[node_name]})
        
    customer_angles.sort(key=lambda x: x['angle'])
    routes, route_vols, current_route, current_vol = [], [], [], 0.0
    
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
# 🗺️ 5. โมดูลสร้างแผนที่ Interactive (Folium)
# =====================================================================
def create_interactive_map(routes, locations, nodes):
    depot_coords = (locations[0][1], locations[0][2])
    m = folium.Map(location=depot_coords, zoom_start=15, tiles='CartoDB positron')
    colors = ['#FF5733', '#335BFF', '#28B463', '#9B59B6', '#E67E22', '#1ABC9C', '#34495E']
    coords_lonlat = {item[0]: (item[2], item[1]) for item in locations}

    folium.Marker(
        location=depot_coords,
        popup=folium.Popup(f"<b>DEPOT</b><br>โรงจัดการขยะ", max_width=200),
        icon=folium.Icon(color="red", icon="home", prefix="fa")
    ).add_to(m)

    for item in locations[1:]:
        folium.CircleMarker(
            location=(item[1], item[2]), radius=6,
            popup=folium.Popup(f"<b>{item[0]}</b><br>ปริมาณขยะ: {item[3]} ลบ.ม.", max_width=250),
            tooltip=item[0], color="#34495E", fill=True, fill_color="#F1C40F", fill_opacity=0.9
        ).add_to(m)

    for trip_idx, route_seq in enumerate(routes):
        route_color = colors[trip_idx % len(colors)]
        road_coords_lonlat = get_osrm_route_geometry(route_seq, coords_lonlat, nodes[0])
        road_coords_latlon = [(pt[1], pt[0]) for pt in road_coords_lonlat]
        
        plugins.AntPath(
            locations=road_coords_latlon, color=route_color, weight=5, opacity=0.8,
            dash_array=[10, 20], delay=800, tooltip=f"<b>Trip {trip_idx+1}</b>"
        ).add_to(m)
        time.sleep(0.1)
    return m

# =====================================================================
# 🖥️ 6. หน้าจอผู้ใช้งาน (Streamlit UI)
# =====================================================================
st.set_page_config(page_title="Smart Waste Collection CVRP", layout="wide")
st.title("🚛 Smart Waste Collection Routing System (Green Logistics Edition)")
st.markdown("ระบบจัดเส้นทางพร้อมการประเมินคาร์บอนฟุตพริ้นท์ตามมาตรฐานสากล ($E = A \\times EF \\times GWP$)")

with st.sidebar:
    st.header("⚙️ 1. ปรับแต่งยานพาหนะ (Fleet)")
    max_vehicles = st.number_input("จำนวนรถขยะที่มีในระบบ (คัน)", min_value=1, value=2, step=1)
    max_capacity = st.number_input("ความจุสูงสุดของรถ (ลบ.ม. / คัน)", min_value=1.0, value=4.5, step=0.5)
    
    st.header("⚙️ 2. เลือกอัลกอริทึม")
    algorithm_choice = st.selectbox("เทคนิคการจัดเส้นทาง", ("Clarke-Wright Savings", "Sweep Algorithm"))
    
    # 🌿 ส่วนใหม่: ตัวแปรคาร์บอนฟุตพริ้นท์
    st.header("🌿 3. ตัวแปรคาร์บอนฟุตพริ้นท์")
    st.latex(r"E = A \times EF \times GWP")
    
    st.markdown("**A = Activity Data (ปริมาณเชื้อเพลิงที่ใช้)**")
    fuel_economy = st.number_input("อัตราสิ้นเปลืองน้ำมันของรถ (กม./ลิตร)", value=5.0, step=0.5)
    
    st.markdown("**EF = Emission Factor**")
    ef_value = st.number_input("ค่าสัมประสิทธิ์การปล่อยก๊าซ (kgCO₂/ลิตร)", value=2.7446, step=0.0001, format="%.4f")
    
    st.markdown("**GWP = Global Warming Potential**")
    gwp_value = st.number_input("ค่าศักยภาพทำให้เกิดโลกร้อน", value=1.0, step=0.1)

    st.header("📂 4. วิธีการนำเข้าข้อมูล")
    data_mode = st.radio("เลือกรูปแบบข้อมูล:", ("ใช้ข้อมูลทดสอบ มทส.", "กรอกข้อมูลใหม่เองทั้งหมด", "อัปโหลดไฟล์ Excel/CSV"))
    uploaded_file = st.file_uploader("อัปโหลดไฟล์ (ถ้ามี)", type=["xlsx", "csv"]) if data_mode == "อัปโหลดไฟล์ Excel/CSV" else None

# --- ส่วนจัดการตารางข้อมูล ---
st.subheader("📝 ตารางจัดการข้อมูลพิกัด GPS และปริมาณขยะ")
if data_mode == "ใช้ข้อมูลทดสอบ มทส.":
    df_input = pd.DataFrame(DEFAULT_DATA, columns=["Node_Name", "Latitude", "Longitude", "Demand"])
elif data_mode == "กรอกข้อมูลใหม่เองทั้งหมด":
    df_input = pd.DataFrame([("Depot โรงจัดการขยะหลัก", 14.862939, 102.027903, 0.0), ("จุดทิ้งขยะ 1", 14.869030, 102.021350, 0.5)], columns=["Node_Name", "Latitude", "Longitude", "Demand"])
else:
    if uploaded_file is not None:
        df_input = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    else:
        st.warning("⚠️ กรุณาทำการอัปโหลดไฟล์ที่แถบเมนูด้านข้างก่อนดำเนินการ")
        st.stop()

edited_df = st.data_editor(df_input, num_rows="dynamic", use_container_width=True)
start_btn = st.button("🚀 ยืนยันข้อมูลและเริ่มการประมวลผล (Start Optimization)", type="primary")

# =====================================================================
# 🚀 7. ส่วนประมวลผลหลัก (Execution Block)
# =====================================================================
if start_btn:
    data_to_use = edited_df.values.tolist()
    nodes, demands = edited_df["Node_Name"].tolist(), dict(zip(edited_df["Node_Name"], edited_df["Demand"]))
    
    if len(nodes) < 2: st.error("❌ ต้องมี Depot และจุดเก็บอย่างน้อย 1 จุด"); st.stop()
    if max(demands.values()) > max_capacity: st.error(f"❌ มีขยะล้นความจุรถ ({max_capacity} ลบ.ม.)"); st.stop()

    osrm_input_format = [(row[0], row[1], row[2], row[3]) for row in data_to_use]

    with st.spinner("📡 กำลังดึงข้อมูลระยะทางและประมวลผล..."):
        df_dist = get_distance_matrix(osrm_input_format)
        df_dist.columns = df_dist.index = nodes

        if algorithm_choice == "Clarke-Wright Savings":
            routes, route_vols = run_savings_algorithm(df_dist, demands, nodes, max_capacity)
        else:
            routes, route_vols = run_sweep_algorithm(osrm_input_format, demands, nodes, max_capacity)
            
        grand_total_distance = sum(sum(df_dist.loc[full_route[k], full_route[k+1]] for k in range(len(full_route)-1)) for full_route in ([nodes[0]] + r + [nodes[0]] for r in routes))
        grand_total_volume = sum(route_vols)
        
        # 🌿 -------------------------------------------------------------
        # 🌿 ระบบคำนวณคาร์บอนฟุตพริ้นท์ ตามสูตร E = A * EF * GWP
        # 🌿 -------------------------------------------------------------
        # 1. คำนวณ Activity Data (A) = ระยะทางรวม / อัตราสิ้นเปลือง (Liters)
        activity_data_A = grand_total_distance / fuel_economy
        
        # 2. คำนวณ Emissions (E)
        carbon_emitted_E = activity_data_A * ef_value * gwp_value
        
        st.success("✅ ออปติไมซ์คำตอบและประเมินคาร์บอนสำเร็จ!")
        
        # แสดงผลลัพธ์เป็นแผง Dashboard
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("รอบวิ่งที่ต้องใช้", f"{len(routes)} เที่ยว")
        col2.metric("ระยะทางขับขี่จริง", f"{grand_total_distance:.2f} กม.")
        col3.metric("ปริมาตรขยะขนส่ง", f"{grand_total_volume:.2f} ลบ.ม.")
        col4.metric("คาร์บอนฟุตพริ้นท์ (E)", f"{carbon_emitted_E:.2f} kgCO₂e", delta="- Green Logistics", delta_color="inverse")
        
        # ขยายเพื่อแสดงวิธีทำแบบละเอียดให้กรรมการดู (Show detailed calculation)
        with st.expander("📊 ดูรายละเอียดการคำนวณคาร์บอนฟุตพริ้นท์ (Carbon Calculation Details)", expanded=False):
            st.markdown(f"**สมการที่ใช้:** $E = A \\times EF \\times GWP$")
            st.markdown(f"- **A (Activity Data):** ใช้น้ำมันไปทั้งหมด `{grand_total_distance:.2f} กม.` ÷ `{fuel_economy} กม./ลิตร` = **`{activity_data_A:.2f} ลิตร`**")
            st.markdown(f"- **EF (Emission Factor):** **`{ef_value:.4f} kgCO₂e/ลิตร`**")
            st.markdown(f"- **GWP (Global Warming Potential):** **`{gwp_value}`**")
            st.markdown(f"**ผลลัพธ์ (E):** `{activity_data_A:.2f}` $\\times$ `{ef_value:.4f}` $\\times$ `{gwp_value}` = **`{carbon_emitted_E:.2f} kgCO₂e`**")

        # -----------------------------------------------------------------
        # 🗺️ เรนเดอร์แผนที่ Folium
        # -----------------------------------------------------------------
        with st.spinner("🗺️ กำลังสร้างแผนที่ GPS Interactive..."):
            m = create_interactive_map(routes, osrm_input_format, nodes)
            st_folium(m, width=1200, height=600, returned_objects=[])
        
        # -----------------------------------------------------------------
        # 📋 โมดูลการจัดสรรตารางงาน
        # -----------------------------------------------------------------
        st.markdown("### 📋 ตารางการปฏิบัติงานแยกตามยานพาหนะ (Fleet Dispatch Schedule)")
        fleet_schedule = {f"🚛 รถขยะคันที่ {i+1}": [] for i in range(int(max_vehicles))}
        for i, r in enumerate(routes):
            fleet_schedule[f"🚛 รถขยะคันที่ {(i % int(max_vehicles)) + 1}"].append({"trip_sequence": (i // int(max_vehicles)) + 1, "route": r, "vol": route_vols[i]})
        
        for vehicle_name, trips in fleet_schedule.items():
            with st.expander(f"{vehicle_name} (รับผิดชอบ {len(trips)} เที่ยววิ่ง | เก็บขยะรวม {sum([t['vol'] for t in trips]):.2f} ลบ.ม.)", expanded=True):
                if not trips: st.write("✅ รถคันนี้ไม่มีภารกิจในรอบการทำงานนี้")
                for t in trips:
                    st.write(f"**รอบวิ่งที่ {t['trip_sequence']}** (ขยะ: {t['vol']:.2f} / {max_capacity} ลบ.ม.)")
                    st.info(f"📍 Depot ➡️ {' ➡️ '.join(t['route'])} ➡️ Depot")
