import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import math
import folium
from folium import plugins
from streamlit_folium import st_folium

# =====================================================================
# ⚙️ 1. ฐานข้อมูลตั้งต้นสำหรับทดสอบ (SUT Default Data)
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
    ("กาญจนาภิเษก", 14.873602, 102.026147, 0.5)
]

# =====================================================================
# 📡 2. การเชื่อมต่อโครงข่ายถนน (OSRM API Integration แบบเลือก Server ได้)
# =====================================================================
@st.cache_data(show_spinner=False)
def get_distance_matrix_osrm(locations, base_url):
    """สร้าง Distance Matrix โดยรองรับ URL เซิร์ฟเวอร์ที่หลากหลาย"""
    N = len(locations)
    distance_matrix = np.zeros((N, N))
    CHUNK_SIZE = 50
    coords = [(item[2], item[1]) for item in locations] # (Lon, Lat)
    
    for i in range(0, N, CHUNK_SIZE):
        for j in range(0, N, CHUNK_SIZE):
            src_chunk = coords[i:i+CHUNK_SIZE]
            dst_chunk = coords[j:j+CHUNK_SIZE]
            combined_coords = src_chunk + dst_chunk
            
            coords_string = ";".join([f"{lon},{lat}" for lon, lat in combined_coords])
            sources_str = ";".join([str(x) for x in range(len(src_chunk))])
            destinations_str = ";".join([str(x) for x in range(len(src_chunk), len(src_chunk) + len(dst_chunk))])
            
            # ยิง API ไปยัง Base URL ที่ผู้ใช้เลือก
            url = f"{base_url}/table/v1/driving/{coords_string}?sources={sources_str}&destinations={destinations_str}&annotations=distance"
            try:
                response = requests.get(url, timeout=10)
                data = response.json()
                if data.get("code") == "Ok":
                    distance_matrix[i:i+len(src_chunk), j:j+len(dst_chunk)] = np.array(data["distances"])
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดในการเชื่อมต่อเซิร์ฟเวอร์ OSRM: {e}")
            time.sleep(0.5)
            
    return pd.DataFrame(distance_matrix) / 1000.0

def get_osrm_route_geometry(route_seq, coords, depot, base_url):
    """ดึงพิกัดจุดเลี้ยวเพื่อวาดเส้นทางบนแผนที่ตามเซิร์ฟเวอร์ที่เลือก"""
    full_route = [depot] + route_seq + [depot]
    route_coords = [coords[n] for n in full_route]
    coords_string = ";".join([f"{lon},{lat}" for lon, lat in route_coords])
    
    url = f"{base_url}/route/v1/driving/{coords_string}?overview=full&geometries=geojson"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        if data.get("code") == "Ok":
            return data["routes"][0]["geometry"]["coordinates"]
    except Exception:
        pass
    
    # Fallback: หากเซิร์ฟเวอร์ล่มให้ลากเส้นตรง
    return route_coords

# =====================================================================
# 🧠 3. อัลกอริทึมการจัดเส้นทาง (Optimization Engines)
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
# 🗺️ 4. โมดูลสร้างแผนที่ Interactive (Folium)
# =====================================================================
def create_interactive_map(routes, locations, nodes, base_url):
    depot_coords = (locations[0][1], locations[0][2])
    m = folium.Map(location=depot_coords, zoom_start=15, tiles='CartoDB positron')
    colors = ['#FF5733', '#335BFF', '#28B463', '#9B59B6', '#E67E22', '#1ABC9C', '#34495E']

    # ปักหมุด Depot
    folium.Marker(
        location=depot_coords, 
        popup="<b>DEPOT</b>", 
        icon=folium.Icon(color="red", icon="home")
    ).add_to(m)
    
    # ปักหมุดจุดเก็บขยะ
    for item in locations[1:]:
        folium.CircleMarker(
            location=(item[1], item[2]), 
            radius=6, 
            tooltip=f"{item[0]} ({item[3]} ลบ.ม.)", 
            color="#34495E", 
            fill=True, 
            fill_color="#F1C40F"
        ).add_to(m)

    coords_lonlat = {item[0]: (item[2], item[1]) for item in locations}

    for trip_idx, route_seq in enumerate(routes):
        route_color = colors[trip_idx % len(colors)]
        
        # ดึงพิกัดแนวถนนจากเซิร์ฟเวอร์ OSRM ที่เราเลือก
        road_coords_lonlat = get_osrm_route_geometry(route_seq, coords_lonlat, nodes[0], base_url)
        road_coords_latlon = [(pt[1], pt[0]) for pt in road_coords_lonlat]
        
        plugins.AntPath(
            locations=road_coords_latlon, 
            color=route_color, 
            weight=5, 
            dash_array=[10, 20], 
            tooltip=f"Trip {trip_idx+1}"
        ).add_to(m)
        time.sleep(0.1)
        
    return m

# =====================================================================
# 🖥️ 5. หน้าจอผู้ใช้งาน (Streamlit UI)
# =====================================================================
st.set_page_config(page_title="Comparative VRP Routing", layout="wide")
st.title("🚛 Smart Waste Collection Routing (API Comparison Mode)")
st.markdown("ระบบวิเคราะห์เปรียบเทียบการจัดเส้นทางผ่านผู้ให้บริการ **OSRM API** ในรูปแบบต่างๆ")

with st.sidebar:
    st.header("⚙️ 1. เลือกเซิร์ฟเวอร์ OSRM (API Endpoint)")
    osrm_choice = st.selectbox(
        "เลือก Routing Engine ที่ต้องการเปรียบเทียบ:",
        (
            "1. Public OSRM (Global Default)", 
            "2. FOSSGIS OSRM (Alternative Public)", 
            "3. Localhost (Custom Server 5000)",
            "4. กำหนด URL เอง (Custom Endpoint)"
        )
    )
    
    # กำหนด Base URL ตามที่ผู้ใช้เลือก
    if osrm_choice == "1. Public OSRM (Global Default)":
        base_url = "http://router.project-osrm.org"
    elif osrm_choice == "2. FOSSGIS OSRM (Alternative Public)":
        base_url = "https://routing.openstreetmap.de/routed-car"
    elif osrm_choice == "3. Localhost (Custom Server 5000)":
        base_url = "http://localhost:5000"
    else:
        base_url = st.text_input("ระบุ OSRM URL (เช่น http://192.168.1.5:5000):", value="http://router.project-osrm.org")

    st.info(f"📍 **URL ปัจจุบัน:** `{base_url}`")

    st.header("⚙️ 2. ปรับแต่งยานพาหนะ")
    max_vehicles = st.number_input("จำนวนรถขยะที่มีในระบบ", min_value=1, value=2)
    max_capacity = st.number_input("ความจุสูงสุดของรถ (ลบ.ม.)", min_value=1.0, value=4.5)
    
    st.header("⚙️ 3. เลือกอัลกอริทึม")
    algorithm_choice = st.selectbox("เทคนิคการจัดเส้นทาง", ("Clarke-Wright Savings", "Sweep Algorithm"))
    
    st.header("🌿 4. ตัวแปรเศรษฐศาสตร์และคาร์บอน")
    fuel_economy = st.number_input("อัตราสิ้นเปลือง (กม./ลิตร)", value=5.0)
    fuel_price = st.number_input("ราคาน้ำมัน (บาท/ลิตร)", value=32.94)
    ef_value = st.number_input("ค่า EF (kgCO₂/ลิตร)", value=2.7446, format="%.4f")
    gwp_value = st.number_input("ค่า GWP", value=1.0)

# --- ส่วนจัดการตารางข้อมูล ---
st.subheader("📝 ตารางจัดการข้อมูลพิกัด GPS และปริมาณขยะ")
df_input = pd.DataFrame(DEFAULT_DATA, columns=["Node_Name", "Latitude", "Longitude", "Demand"])
edited_df = st.data_editor(df_input, num_rows="dynamic", use_container_width=True)
start_btn = st.button("🚀 ยืนยันข้อมูลและเริ่มการประมวลผล", type="primary")

# =====================================================================
# 🚀 6. ส่วนประมวลผลหลัก (Execution Block)
# =====================================================================
if start_btn:
    data_to_use = edited_df.values.tolist()
    nodes = edited_df["Node_Name"].tolist()
    demands = dict(zip(edited_df["Node_Name"], edited_df["Demand"]))
    osrm_input_format = [(row[0], row[1], row[2], row[3]) for row in data_to_use]

    with st.spinner(f"📡 กำลังคำนวณระยะทางจากเซิร์ฟเวอร์: {osrm_choice}..."):
        # ส่ง base_url เข้าไปในฟังก์ชัน
        df_dist = get_distance_matrix_osrm(osrm_input_format, base_url)
        df_dist.columns = df_dist.index = nodes

        if algorithm_choice == "Clarke-Wright Savings":
            routes, route_vols = run_savings_algorithm(df_dist, demands, nodes, max_capacity)
        else:
            routes, route_vols = run_sweep_algorithm(osrm_input_format, demands, nodes, max_capacity)

        grand_total_distance = sum(sum(df_dist.loc[full_route[k], full_route[k+1]] for k in range(len(full_route)-1)) for full_route in ([nodes[0]] + r + [nodes[0]] for r in routes))
        grand_total_volume = sum(route_vols)
        
        # คำนวณสิ่งแวดล้อม & เศรษฐศาสตร์
        activity_data_A = grand_total_distance / fuel_economy
        carbon_emitted_E = activity_data_A * ef_value * gwp_value
        total_fuel_cost = activity_data_A * fuel_price
        
        st.success(f"✅ ประมวลผลผ่าน {base_url} สำเร็จ!")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("รอบวิ่ง (Trips)", f"{len(routes)} เที่ยว")
        col2.metric("ระยะทางจริง", f"{grand_total_distance:.2f} กม.")
        col3.metric("ปริมาตรขยะ", f"{grand_total_volume:.2f} ลบ.ม.")
        col4.metric("คาร์บอน (CO₂e)", f"{carbon_emitted_E:.2f} kg")
        col5.metric("ต้นทุนน้ำมัน", f"฿ {total_fuel_cost:,.2f}")
        
        with st.spinner("🗺️ กำลังสร้างแผนที่ GPS Interactive..."):
            m = create_interactive_map(routes, osrm_input_format, nodes, base_url)
            st_folium(m, width=1200, height=600)
            
        st.markdown("### 📋 ตารางการปฏิบัติงานแยกตามยานพาหนะ")
        fleet_schedule = {f"🚛 รถขยะคันที่ {i+1}": [] for i in range(int(max_vehicles))}
        for i, r in enumerate(routes):
            fleet_schedule[f"🚛 รถขยะคันที่ {(i % int(max_vehicles)) + 1}"].append({"trip_sequence": (i // int(max_vehicles)) + 1, "route": r, "vol": route_vols[i]})
        
        for vehicle_name, trips in fleet_schedule.items():
            with st.expander(f"{vehicle_name} (รับผิดชอบ {len(trips)} เที่ยววิ่ง)", expanded=True):
                for t in trips:
                    st.info(f"📍 Depot ➡️ {' ➡️ '.join(t['route'])} ➡️ Depot (ขยะ: {t['vol']:.2f} ลบ.ม.)")
