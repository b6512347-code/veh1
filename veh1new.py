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

# 🌿 นำเข้าไลบรารีสำหรับสร้างโครงข่าย
import networkx as nx
import xml.etree.ElementTree as ET
import io # สำหรับอ่านไฟล์ที่อัปโหลดผ่านหน้าเว็บ

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
# 🧮 2. โมดูลประมวลผลไฟล์ .OSM (แปลงข้อมูลดิบเป็นสมการกราฟ)
# =====================================================================
def haversine_dist(lon1, lat1, lon2, lat2):
    R = 6371.0
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    return R * (2 * math.asin(math.sqrt(a)))

@st.cache_data
def build_osm_graph_manual(osm_string_data):
    """อ่านข้อมูล .osm ที่ผู้ใช้อัปโหลดและสร้างเป็นโครงข่ายเส้นทาง"""
    G = nx.Graph()
    try:
        # ใช้ io.StringIO เพื่ออ่าน String XML เป็นไฟล์
        tree = ET.parse(io.StringIO(osm_string_data))
        root = tree.getroot()
        nodes = {}
        
        # 1. ดึงพิกัด (Nodes) ทั้งหมด
        for node in root.findall('node'):
            n_id = node.get('id')
            lat, lon = float(node.get('lat')), float(node.get('lon'))
            nodes[n_id] = (lat, lon)
            G.add_node(n_id, x=lon, y=lat)
            
        # 2. ดึงเส้นทางเชื่อม (Ways) ทั้งหมด
        for way in root.findall('way'):
            nds = way.findall('nd')
            way_nodes = [nd.get('ref') for nd in nds]
            for i in range(len(way_nodes)-1):
                n1, n2 = way_nodes[i], way_nodes[i+1]
                if n1 in nodes and n2 in nodes:
                    lat1, lon1 = nodes[n1]
                    lat2, lon2 = nodes[n2]
                    dist = haversine_dist(lon1, lat1, lon2, lat2)
                    G.add_edge(n1, n2, weight=dist)
        return G
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการแปลไฟล์ .osm: {e}")
        return None

def get_nearest_node(G, lon, lat):
    """ค้นหาพิกัดถนนที่ใกล้ที่สุด"""
    nearest_n = None
    min_d = float('inf')
    for n in G.nodes:
        n_lon, n_lat = G.nodes[n]['x'], G.nodes[n]['y']
        d = haversine_dist(lon, lat, n_lon, n_lat)
        if d < min_d:
            min_d = d
            nearest_n = n
    return nearest_n

def get_distance_matrix_osm(G, locations):
    """สร้าง Distance Matrix จากโครงข่ายไฟล์ .osm"""
    N = len(locations)
    distance_matrix = np.zeros((N, N))
    snapped_nodes = [get_nearest_node(G, item[2], item[1]) for item in locations]
    
    for i in range(N):
        for j in range(N):
            if i != j:
                try:
                    dist = nx.shortest_path_length(G, source=snapped_nodes[i], target=snapped_nodes[j], weight='weight')
                    distance_matrix[i][j] = dist
                except nx.NetworkXNoPath:
                    distance_matrix[i][j] = haversine_dist(locations[i][2], locations[i][1], locations[j][2], locations[j][1])
    return pd.DataFrame(distance_matrix)

# =====================================================================
# 📡 3. โมดูลเชื่อมต่อ OSRM API (ออนไลน์)
# =====================================================================
@st.cache_data(show_spinner=False)
def get_distance_matrix_osrm(locations):
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
            sources_str = ";".join([str(x) for x in range(len(src_chunk))])
            destinations_str = ";".join([str(x) for x in range(len(src_chunk), len(src_chunk) + len(dst_chunk))])
            url = f"http://router.project-osrm.org/table/v1/driving/{coords_string}?sources={sources_str}&destinations={destinations_str}&annotations=distance"
            try:
                response = requests.get(url, timeout=10)
                data = response.json()
                if data.get("code") == "Ok":
                    distance_matrix[i:i+len(src_chunk), j:j+len(dst_chunk)] = np.array(data["distances"])
            except: pass
            time.sleep(0.5)
    return pd.DataFrame(distance_matrix) / 1000.0

# =====================================================================
# 🧠 4. อัลกอริทึมการจัดเส้นทาง
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
    routes, route_vols = [[c] for c in customers], [demands[c] for c in customers]

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
            current_route, current_vol = [c['node']], c['vol']
    if current_route:
        routes.append(current_route)
        route_vols.append(current_vol)
    return routes, route_vols

# =====================================================================
# 🗺️ 5. โมดูลสร้างแผนที่ Interactive
# =====================================================================
def create_interactive_map(routes, locations, nodes, routing_mode, G_osm=None):
    depot_coords = (locations[0][1], locations[0][2])
    m = folium.Map(location=depot_coords, zoom_start=15, tiles='CartoDB positron')
    colors = ['#FF5733', '#335BFF', '#28B463', '#9B59B6', '#E67E22', '#1ABC9C', '#34495E']

    # ปักหมุดถนนที่วาดเอง (ถ้ามี)
    if G_osm is not None:
        for u, v in G_osm.edges():
            lon1, lat1 = G_osm.nodes[u]['x'], G_osm.nodes[u]['y']
            lon2, lat2 = G_osm.nodes[v]['x'], G_osm.nodes[v]['y']
            folium.PolyLine([(lat1, lon1), (lat2, lon2)], color="#BDC3C7", weight=2, opacity=0.5, dash_array="5, 5").add_to(m)

    # ปักหมุดสถานที่
    folium.Marker(location=depot_coords, popup="<b>DEPOT</b>", icon=folium.Icon(color="red", icon="home")).add_to(m)
    for item in locations[1:]:
        folium.CircleMarker(location=(item[1], item[2]), radius=6, tooltip=item[0], color="#34495E", fill=True, fill_color="#F1C40F", fill_opacity=0.9).add_to(m)

    coords_lonlat = {item[0]: (item[2], item[1]) for item in locations}

    for trip_idx, route_seq in enumerate(routes):
        route_color = colors[trip_idx % len(colors)]
        full_route = [nodes[0]] + route_seq + [nodes[0]]
        road_coords_latlon = []

        if routing_mode == "Local Map (ออฟไลน์ผ่านไฟล์ .osm)" and G_osm is not None:
            # ลากเส้นตามกราฟ .osm ของเราเอง
            for k in range(len(full_route)-1):
                lon1, lat1 = coords_lonlat[full_route[k]][0], coords_lonlat[full_route[k]][1]
                lon2, lat2 = coords_lonlat[full_route[k+1]][0], coords_lonlat[full_route[k+1]][1]
                n1 = get_nearest_node(G_osm, lon1, lat1)
                n2 = get_nearest_node(G_osm, lon2, lat2)
                try:
                    path = nx.shortest_path(G_osm, source=n1, target=n2, weight='weight')
                    for node_id in path:
                        road_coords_latlon.append((G_osm.nodes[node_id]['y'], G_osm.nodes[node_id]['x']))
                except nx.NetworkXNoPath:
                    road_coords_latlon.extend([(lat1, lon1), (lat2, lon2)])
        else:
            # ลากเส้นด้วย OSRM
            coords_string = ";".join([f"{coords_lonlat[n][0]},{coords_lonlat[n][1]}" for n in full_route])
            url = f"http://router.project-osrm.org/route/v1/driving/{coords_string}?overview=full&geometries=geojson"
            try:
                data = requests.get(url, timeout=10).json()
                road_coords_lonlat = data["routes"][0]["geometry"]["coordinates"]
                road_coords_latlon = [(pt[1], pt[0]) for pt in road_coords_lonlat]
            except:
                road_coords_latlon = [(coords_lonlat[n][1], coords_lonlat[n][0]) for n in full_route]
        
        if len(road_coords_latlon) > 1:
            plugins.AntPath(locations=road_coords_latlon, color=route_color, weight=5, opacity=0.8, dash_array=[10, 20], delay=800, tooltip=f"Trip {trip_idx+1}").add_to(m)
        time.sleep(0.1)
    return m

# =====================================================================
# 🖥️ 6. หน้าจอผู้ใช้งาน (Streamlit UI)
# =====================================================================
st.set_page_config(page_title="Smart Waste Collection CVRP", layout="wide")
st.title("🚛 Smart Waste Collection Routing System (.OSM Edition)")
st.markdown("ระบบวิเคราะห์เส้นทางขยะ รองรับการอัปโหลด **ไฟล์ .osm จาก JOSM/QGIS** โดยตรงหน้าเว็บ")

with st.sidebar:
    st.header("⚙️ 1. เลือกโครงข่ายถนน")
    routing_mode = st.radio("Routing Engine:", ("OSRM API (ออนไลน์/สาธารณะ)", "Local Map (ออฟไลน์ผ่านไฟล์ .osm)"))
    
    G_osm = None
    if routing_mode == "Local Map (ออฟไลน์ผ่านไฟล์ .osm)":
        # ⚠️ สร้างกล่องให้อัปโหลดไฟล์ .osm บนหน้าเว็บโดยตรง
        uploaded_osm = st.file_uploader("📂 อัปโหลดไฟล์ .osm ของคุณที่นี่", type=["osm"])
        
        if uploaded_osm is not None:
            with st.spinner("⏳ กำลังอ่านโครงสร้างไฟล์ .osm..."):
                osm_content = uploaded_osm.getvalue().decode("utf-8") # อ่านเนื้อหาไฟล์
                G_osm = build_osm_graph_manual(osm_content)
            
            if G_osm is not None and len(G_osm.edges) > 0:
                st.success(f"✅ โครงข่ายสมบูรณ์! (พิกัด: {len(G_osm.nodes)} จุด | เส้นทาง: {len(G_osm.edges)} เส้น)")
            else:
                st.error("❌ ไฟล์ที่อัปโหลดไม่มีข้อมูลเส้นทาง (Way) กรุณาตรวจสอบการ Export จาก JOSM")
        else:
            st.warning("⚠️ โปรดอัปโหลดไฟล์ .osm ของคุณเพื่อดำเนินการต่อ")

    st.header("⚙️ 2. ปรับแต่งยานพาหนะ")
    max_vehicles = st.number_input("จำนวนรถขยะที่มี", min_value=1, value=2)
    max_capacity = st.number_input("ความจุสูงสุดของรถ (ลบ.ม.)", min_value=1.0, value=4.5)
    
    st.header("⚙️ 3. อัลกอริทึม & คาร์บอน")
    algorithm_choice = st.selectbox("เทคนิคการจัดเส้นทาง", ("Clarke-Wright Savings", "Sweep Algorithm"))
    fuel_economy = st.number_input("อัตราสิ้นเปลือง (กม./ลิตร)", value=5.0)
    fuel_price = st.number_input("ราคาน้ำมัน (บาท/ลิตร)", value=32.94)
    ef_value = st.number_input("ค่า EF (kgCO₂/ลิตร)", value=2.7446, format="%.4f")
    gwp_value = st.number_input("ค่า GWP", value=1.0)

# --- ส่วนจัดการตารางข้อมูล ---
st.subheader("📝 ตารางข้อมูลพิกัดและปริมาณขยะ")
df_input = pd.DataFrame(DEFAULT_DATA, columns=["Node_Name", "Latitude", "Longitude", "Demand"])
edited_df = st.data_editor(df_input, num_rows="dynamic", use_container_width=True)
start_btn = st.button("🚀 เริ่มการประมวลผล", type="primary")

if start_btn:
    if routing_mode == "Local Map (ออฟไลน์ผ่านไฟล์ .osm)" and (G_osm is None or len(G_osm.edges) == 0):
        st.error(f"❌ ไม่สามารถประมวลผลได้ เนื่องจากคุณยังไม่ได้อัปโหลดไฟล์ .osm หรือโครงข่ายถนนไม่สมบูรณ์")
        st.stop()

    data_to_use = edited_df.values.tolist()
    nodes, demands = edited_df["Node_Name"].tolist(), dict(zip(edited_df["Node_Name"], edited_df["Demand"]))
    osrm_input_format = [(row[0], row[1], row[2], row[3]) for row in data_to_use]

    with st.spinner(f"📡 กำลังคำนวณระยะทางจาก {routing_mode}..."):
        if routing_mode == "Local Map (ออฟไลน์ผ่านไฟล์ .osm)":
            df_dist = get_distance_matrix_osm(G_osm, osrm_input_format)
        else:
            df_dist = get_distance_matrix_osrm(osrm_input_format)
            
        df_dist.columns = df_dist.index = nodes
        if algorithm_choice == "Clarke-Wright Savings":
            routes, route_vols = run_savings_algorithm(df_dist, demands, nodes, max_capacity)
        else:
            routes, route_vols = run_sweep_algorithm(osrm_input_format, demands, nodes, max_capacity)

        grand_total_distance = sum(sum(df_dist.loc[full_route[k], full_route[k+1]] for k in range(len(full_route)-1)) for full_route in ([nodes[0]] + r + [nodes[0]] for r in routes))
        
        activity_data_A = grand_total_distance / fuel_economy
        carbon_emitted_E = activity_data_A * ef_value * gwp_value
        total_fuel_cost = activity_data_A * fuel_price
        
        st.success(f"✅ ประมวลผลสำเร็จ!")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("รอบวิ่ง (Trips)", f"{len(routes)} เที่ยว")
        col2.metric("ระยะทางจริง", f"{grand_total_distance:.2f} กม.")
        col3.metric("ปริมาตรขยะ", f"{sum(route_vols):.2f} ลบ.ม.")
        col4.metric("คาร์บอน (CO₂e)", f"{carbon_emitted_E:.2f} kg")
        col5.metric("ต้นทุนน้ำมัน", f"฿ {total_fuel_cost:,.2f}")
        
        with st.spinner("🗺️ กำลังเรนเดอร์แผนที่..."):
            m = create_interactive_map(routes, osrm_input_format, nodes, routing_mode, G_osm)
            st_folium(m, width=1200, height=600)
            
        st.markdown("### 📋 ตารางการปฏิบัติงานแยกตามยานพาหนะ")
        fleet_schedule = {f"🚛 รถขยะคันที่ {i+1}": [] for i in range(int(max_vehicles))}
        for i, r in enumerate(routes):
            fleet_schedule[f"🚛 รถขยะคันที่ {(i % int(max_vehicles)) + 1}"].append({"trip_sequence": (i // int(max_vehicles)) + 1, "route": r, "vol": route_vols[i]})
        
        for vehicle_name, trips in fleet_schedule.items():
            with st.expander(f"{vehicle_name} (รับผิดชอบ {len(trips)} เที่ยววิ่ง)", expanded=True):
                for t in trips:
                    st.info(f"📍 Depot ➡️ {' ➡️ '.join(t['route'])} ➡️ Depot (ขยะ: {t['vol']:.2f} ลบ.ม.)")
