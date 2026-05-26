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
import io

# Set page config ให้แสดงผลกว้างเต็มตา
st.set_page_config(page_title="Balanced VRP Routing System", layout="wide")

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

# ตารางจุดทิ้งขยะตั้งต้น (ไม่มี Depot)
DEFAULT_DATA = [
    ("ภูมิทัศน์(ใหม่)", 14.86903, 102.02135, 0.3),
    ("สวนพฤกษศาสตร์", 14.86991, 102.022113, 0.3),
    ("อุทยานผีเสื้อ", 14.871074, 102.022713, 0.3),
    ("ซินโครตรอน", 14.872731, 102.023232, 0.1),
    ("อาคารสุรพัฒน์ 2", 14.8754, 102.02286, 0.2),
    ("อาคารสุรพัฒน์ 3", 14.874078, 102.022316, 0.4),
    ("เรือนไทย", 14.875346, 102.021912, 0.3),
    ("อาคารสุรพัฒน์ 1 จุดที่ 1", 14.87584, 102.02302, 0.2),
    ("อาคารสุรพัฒน์ 1 จุดที่ 2", 14.87572, 102.02284, 0.2),
    ("เซเว่น-อีเลฟเว่น เทคโนธานี จุดที่ 1", 14.876072, 102.022745, 0.7),
    ("เซเว่น-อีเลฟเว่น เทคโนธานี จุดที่ 2", 14.876125, 102.022341, 0.4),
    ("ร้านคอกาแฟ ข้าง 7-11 เทคโนธานี", 14.876938, 102.022377, 0.1),
    ("อาคารสุรสัมนาคาร", 14.876533, 102.024665, 0.1),
    ("โรงอาหารครัวท่านท้าว", 14.877234, 102.02026, 0.2),
    ("อาคารวิจัยมันสำปะหลัง", 14.874527, 102.020047, 0.2),
    ("หอดูดาว", 14.87414, 102.027598, 0.2),
    ("กาญจนาภิเษก", 14.873602, 102.026147, 0.5)
]

# =====================================================================
# 🧮 2. โมดูลประมวลผลไฟล์ .OSM
# =====================================================================
def haversine_dist(lon1, lat1, lon2, lat2):
    R = 6371.0
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    return R * (2 * math.asin(math.sqrt(a)))

@st.cache_data
def build_osm_graph_manual(osm_string_data):
    G = nx.Graph()
    try:
        tree = ET.parse(io.StringIO(osm_string_data))
        root = tree.getroot()
        nodes = {}
        for node in root.findall('node'):
            n_id = node.get('id')
            lat, lon = float(node.get('lat')), float(node.get('lon'))
            nodes[n_id] = (lat, lon)
            G.add_node(n_id, x=lon, y=lat)
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
# 📡 3. โมดูลเชื่อมต่อ OSRM API
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
# 🧠 4. อัลกอริทึมการจัดเส้นทาง (รวม 4 ระบบ)
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

def run_balanced_savings_algorithm(df_dist, demands, nodes, max_capacity, num_vehicles):
    depot = nodes[0]
    customers = nodes[1:]
    
    total_demand = sum([demands[c] for c in customers])
    min_trips = max(1, math.ceil(total_demand / max_capacity))
    effective_vehicles = max(num_vehicles, min_trips)
    target_cap = total_demand / effective_vehicles
    
    soft_capacity = max(target_cap * 1.15, max([demands[c] for c in customers]))
    effective_capacity = min(soft_capacity, max_capacity)

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
                if route_vols[idx_i] + route_vols[idx_j] <= effective_capacity:
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

def run_balanced_sweep_algorithm(locations, demands, nodes, max_capacity):
    depot, depot_lat, depot_lon = nodes[0], locations[0][1], locations[0][2]
    customer_angles = []
    for i, item in enumerate(locations[1:]):
        node_name, lat, lon = item[0], item[1], item[2]
        angle = math.degrees(math.atan2(lat - depot_lat, lon - depot_lon))
        if angle < 0: angle += 360
        customer_angles.append({"node": node_name, "angle": angle, "vol": demands[node_name]})
    customer_angles.sort(key=lambda x: x['angle'])
    
    total_demand = sum([c['vol'] for c in customer_angles])
    num_routes = max(1, math.ceil(total_demand / max_capacity))
    target_cap = total_demand / num_routes

    routes, route_vols, current_route, current_vol = [], [], [], 0.0
    for c in customer_angles:
        if current_vol + c['vol'] > max_capacity:
            routes.append(current_route)
            route_vols.append(current_vol)
            current_route, current_vol = [c['node']], c['vol']
        elif current_vol + c['vol'] > target_cap and current_vol >= (target_cap * 0.75):
            routes.append(current_route)
            route_vols.append(current_vol)
            current_route, current_vol = [c['node']], c['vol']
        else:
            current_route.append(c['node'])
            current_vol += c['vol']
            
    if current_route:
        routes.append(current_route)
        route_vols.append(current_vol)
    return routes, route_vols

# =====================================================================
# 🗺️ 5. โมดูลสร้างแผนที่ Interactive
# =====================================================================
def create_interactive_map(routes, locations, nodes, routing_mode, G_osm=None, map_type='CartoDB positron'):
    depot_coords = (locations[0][1], locations[0][2])
    
    # ⚠️ ตรวจสอบและแมปประเภทแผนที่ให้เข้ากับ Folium
    tiles_mapping = {
        'แผนที่พื้นฐานแบบสว่าง (CartoDB Positron)': 'CartoDB positron',
        'แผนที่พื้นฐานแบบมืด (CartoDB Dark_Matter)': 'CartoDB dark_matter',
        'แผนที่ภูมิประเทศ (OpenStreetMap)': 'OpenStreetMap',
        'แผนที่ดาวเทียม (Esri Satellite)': 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        'แผนที่ภูมิประเทศ+ดาวเทียม (Esri NatGeo)': 'https://server.arcgisonline.com/ArcGIS/rest/services/NatGeo_World_Map/MapServer/tile/{z}/{y}/{x}'
    }
    
    selected_tile = tiles_mapping.get(map_type, 'CartoDB positron')
    
    # ถ้าเป็น Custom URL (อย่างดาวเทียม) ต้องใส่ attr ด้วย
    if 'http' in selected_tile:
        m = folium.Map(location=depot_coords, zoom_start=15, tiles=selected_tile, attr='Esri')
    else:
        m = folium.Map(location=depot_coords, zoom_start=15, tiles=selected_tile)
        
    colors = ['#FF5733', '#335BFF', '#28B463', '#9B59B6', '#E67E22', '#1ABC9C', '#34495E']

    if G_osm is not None:
        for u, v in G_osm.edges():
            lon1, lat1 = G_osm.nodes[u]['x'], G_osm.nodes[u]['y']
            lon2, lat2 = G_osm.nodes[v]['x'], G_osm.nodes[v]['y']
            folium.PolyLine([(lat1, lon1), (lat2, lon2)], color="#BDC3C7", weight=2, opacity=0.5, dash_array="5, 5").add_to(m)

    folium.Marker(location=depot_coords, popup="<b>DEPOT (ศูนย์กลาง)</b>", icon=folium.Icon(color="red", icon="home")).add_to(m)
    for item in locations[1:]:
        folium.CircleMarker(location=(item[1], item[2]), radius=6, tooltip=item[0], color="#34495E", fill=True, fill_color="#F1C40F", fill_opacity=0.9).add_to(m)

    coords_lonlat = {item[0]: (item[2], item[1]) for item in locations}

    for trip_idx, route_seq in enumerate(routes):
        route_color = colors[trip_idx % len(colors)]
        full_route = [nodes[0]] + route_seq + [nodes[0]]
        road_coords_latlon = []

        if routing_mode == "Local Map (ออฟไลน์ผ่านไฟล์ .osm)" and G_osm is not None:
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
st.title("🚛 Smart Waste Collection Routing System (Balanced Edition)")
st.markdown("ระบบจัดเส้นทางอัจฉริยะ พร้อมฟังก์ชัน **Load Balancing (รักษาสมดุลภาระงาน)** และปรับรูปแบบแผนที่ได้")

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.header("⚙️ 1. เลือกโครงข่ายถนน (Map Source)")
    routing_mode = st.radio("Routing Engine:", ("OSRM API (ออนไลน์/สาธารณะ)", "Local Map (ออฟไลน์ผ่านไฟล์ .osm)"))
    
    G_osm = None
    if routing_mode == "Local Map (ออฟไลน์ผ่านไฟล์ .osm)":
        uploaded_osm = st.file_uploader("📂 อัปโหลดไฟล์ .osm ของคุณที่นี่", type=["osm"])
        if uploaded_osm is not None:
            with st.spinner("⏳ กำลังอ่านโครงสร้างไฟล์ .osm..."):
                osm_content = uploaded_osm.getvalue().decode("utf-8")
                G_osm = build_osm_graph_manual(osm_content)
            if G_osm is not None and len(G_osm.edges) > 0:
                st.success(f"✅ โครงข่ายสมบูรณ์! (พิกัด: {len(G_osm.nodes)} จุด | เส้นทาง: {len(G_osm.edges)} เส้น)")
            else:
                st.error("❌ ไฟล์ที่อัปโหลดไม่มีข้อมูลเส้นทาง")
        else:
            st.warning("⚠️ โปรดอัปโหลดไฟล์ .osm ของคุณเพื่อดำเนินการต่อ")

    # ⚠️ เพิ่มตัวเลือกปรับแต่งรูปแบบแผนที่
    st.header("🗺️ 2. รูปแบบแผนที่ (Basemap Layer)")
    map_type = st.selectbox(
        "เลือกประเภทของแผนที่แสดงผล:",
        (
            "แผนที่ภูมิประเทศ (OpenStreetMap)",
            "แผนที่ดาวเทียม (Esri Satellite)",
            "แผนที่พื้นฐานแบบสว่าง (CartoDB Positron)",
            "แผนที่พื้นฐานแบบมืด (CartoDB Dark_Matter)",
            "แผนที่ภูมิประเทศ+ดาวเทียม (Esri NatGeo)"
        )
    )

    st.header("📍 3. ตั้งค่าจุดศูนย์กลาง (Depot)")
    depot_name = st.text_input("ชื่อจุด Depot", value="Depot โรงจัดการขยะ")
    depot_lat = st.number_input("ละติจูด (Latitude)", value=14.862939, format="%.6f")
    depot_lon = st.number_input("ลองจิจูด (Longitude)", value=102.027903, format="%.6f")

    st.header("⚙️ 4. ปรับแต่งยานพาหนะ")
    max_vehicles = st.number_input("จำนวนรถขยะที่มี", min_value=1, value=2)
    max_capacity = st.number_input("ความจุสูงสุดของรถ (ลบ.ม.)", min_value=1.0, value=4.5)
    
    st.header("⚙️ 5. อัลกอริทึม & คาร์บอน")
    algorithm_choice = st.selectbox("เทคนิคการจัดเส้นทาง", (
        "Balanced Clarke-Wright Savings (แนะนำ)",
        "Balanced Workload Sweep", 
        "Clarke-Wright Savings (มาตรฐาน)", 
        "Sweep Algorithm (มาตรฐาน)"
    ))
    fuel_economy = st.number_input("อัตราสิ้นเปลือง (กม./ลิตร)", value=5.0)
    fuel_price = st.number_input("ราคาน้ำมัน (บาท/ลิตร)", value=32.94)
    ef_value = st.number_input("ค่า EF (kgCO₂/ลิตร)", value=2.7446, format="%.4f")
    gwp_value = st.number_input("ค่า GWP", value=1.0)

# ----------------- MAIN AREA -----------------
st.subheader("📝 1. ข้อมูลพิกัดจุดทิ้งขยะ (Customer Nodes)")

uploaded_coord_file = st.file_uploader("📂 อัปโหลดไฟล์ Excel/CSV (เพื่อนำเข้าพิกัดและปริมาณขยะแทนตารางด้านล่าง)", type=["xlsx", "csv"])

if uploaded_coord_file is not None:
    if uploaded_coord_file.name.endswith('.csv'):
        df_input = pd.read_csv(uploaded_coord_file)
    else:
        df_input = pd.read_excel(uploaded_coord_file)
    st.success("✅ โหลดข้อมูลจากไฟล์สำเร็จ!")
else:
    df_input = pd.DataFrame(DEFAULT_DATA, columns=["Node_Name", "Latitude", "Longitude", "Demand"])

st.markdown("*(หมายเหตุ: ตารางนี้แสดงเฉพาะจุดเก็บขยะ **ไม่ต้องใส่พิกัด Depot** ลงในตารางนี้)*")
edited_df = st.data_editor(df_input, num_rows="dynamic", use_container_width=True)

if 'show_results' not in st.session_state:
    st.session_state['show_results'] = False

st.markdown("<br>", unsafe_allow_html=True)
start_btn = st.button("🚀 ยืนยันข้อมูลและเริ่มจัดเส้นทาง", type="primary", use_container_width=True)

if start_btn:
    cleaned_df = edited_df.dropna(subset=['Node_Name', 'Latitude', 'Longitude', 'Demand'])
    
    if len(cleaned_df) < 1:
        st.error("❌ ข้อมูลไม่เพียงพอ ต้องมีจุดเก็บขยะอย่างน้อย 1 จุด")
        st.session_state['show_results'] = False
    elif routing_mode == "Local Map (ออฟไลน์ผ่านไฟล์ .osm)" and (G_osm is None or len(G_osm.edges) == 0):
        st.error(f"❌ ไม่สามารถประมวลผลได้ เนื่องจากคุณยังไม่ได้อัปโหลดไฟล์แผนที่ .osm หรือโครงข่ายถนนไม่สมบูรณ์")
        st.session_state['show_results'] = False
    else:
        st.session_state['show_results'] = True
        st.session_state['process_data'] = cleaned_df

st.markdown("---")

# =====================================================================
# 🚀 7. ส่วนแสดงผลลัพธ์ (Interface สรุปผลแบบแยกส่วนชัดเจน)
# =====================================================================
if st.session_state.get('show_results', False):
    
    cleaned_df = st.session_state['process_data']
    total_customers = len(cleaned_df)
    
    depot_data = [depot_name, depot_lat, depot_lon, 0.0]
    data_to_use = [depot_data] + cleaned_df.values.tolist()
    
    nodes = [row[0] for row in data_to_use]
    demands = dict(zip([row[0] for row in data_to_use], [row[3] for row in data_to_use]))
    osrm_input_format = [(row[0], row[1], row[2], row[3]) for row in data_to_use]

    with st.spinner(f"📡 กำลังประมวลผลด้วยอัลกอริทึม {algorithm_choice}..."):
        if routing_mode == "Local Map (ออฟไลน์ผ่านไฟล์ .osm)":
            df_dist = get_distance_matrix_osm(G_osm, osrm_input_format)
        else:
            df_dist = get_distance_matrix_osrm(osrm_input_format)
            
        df_dist.columns = df_dist.index = nodes
        
        if algorithm_choice == "Clarke-Wright Savings (มาตรฐาน)":
            routes, route_vols = run_savings_algorithm(df_dist, demands, nodes, max_capacity)
        elif algorithm_choice == "Balanced Clarke-Wright Savings (แนะนำ)":
            routes, route_vols = run_balanced_savings_algorithm(df_dist, demands, nodes, max_capacity, max_vehicles)
        elif algorithm_choice == "Balanced Workload Sweep":
            routes, route_vols = run_balanced_sweep_algorithm(osrm_input_format, demands, nodes, max_capacity)
        else:
            routes, route_vols = run_sweep_algorithm(osrm_input_format, demands, nodes, max_capacity)

        route_distances = []
        for r in routes:
            full_route = [nodes[0]] + r + [nodes[0]]
            dist = sum([df_dist.loc[full_route[k], full_route[k+1]] for k in range(len(full_route)-1)])
            route_distances.append(dist)
            
        grand_total_distance = sum(route_distances)
        activity_data_A = grand_total_distance / fuel_economy
        carbon_emitted_E = activity_data_A * ef_value * gwp_value
        total_fuel_cost = activity_data_A * fuel_price
        
        # ---------------------------------------------------------
        # ระบบ Fleet Assignment ชั้นที่ 2 (Min-Max Greedy Balancing)
        # ---------------------------------------------------------
        trip_data = [{"original_idx": i+1, "route": routes[i], "vol": route_vols[i], "dist": route_distances[i]} for i in range(len(routes))]
        trip_data.sort(key=lambda x: x['dist'], reverse=True)
        
        fleet_schedule = {f"🚛 รถขยะคันที่ {i+1}": [] for i in range(int(max_vehicles))}
        vehicle_workloads = {f"🚛 รถขยะคันที่ {i+1}": 0.0 for i in range(int(max_vehicles))}
        
        for t in trip_data:
            best_vehicle = min(vehicle_workloads, key=vehicle_workloads.get)
            t['trip_sequence'] = len(fleet_schedule[best_vehicle]) + 1
            fleet_schedule[best_vehicle].append(t)
            vehicle_workloads[best_vehicle] += t['dist']

        # ---------------------------------------------------------
        # Dashboard UI
        # ---------------------------------------------------------
        st.subheader("📊 2. สรุปผลการปฏิบัติงาน (Executive Dashboard)")
        st.success("✅ วิเคราะห์และออกแบบเส้นทางเสร็จสมบูรณ์!")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("📌 จำนวนจุดเก็บขยะทั้งหมด", f"{total_customers} จุด")
        c2.metric("🚛 รอบรถที่ต้องวิ่ง (Trips)", f"{len(routes)} เที่ยว")
        c3.metric("🗑️ ปริมาตรรวม", f"{sum(route_vols):.2f} ลบ.ม.")
        
        c4, c5, c6 = st.columns(3)
        c4.metric("📍 ระยะทางรวม", f"{grand_total_distance:.2f} กม.")
        c5.metric("🌿 คาร์บอน (CO₂e)", f"{carbon_emitted_E:.2f} kg")
        c6.metric("⛽ ต้นทุนน้ำมัน", f"฿ {total_fuel_cost:,.2f}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("🗺️ 3. แผนที่จำลองการเดินรถ (Interactive GPS Map)")
        with st.spinner("กำลังเรนเดอร์แผนที่..."):
            # ⚠️ ส่ง map_type ที่ผู้ใช้เลือกเข้าไปยังฟังก์ชันสร้างแผนที่
            m = create_interactive_map(routes, osrm_input_format, nodes, routing_mode, G_osm, map_type)
            st_folium(m, width=1200, height=600, returned_objects=[])
            
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("📋 4. ตารางปฏิบัติงานของรถแต่ละคัน (Balanced Fleet Dispatch)")
        
        for vehicle_name, trips in fleet_schedule.items():
            total_dist = sum([t['dist'] for t in trips])
            total_vol = sum([t['vol'] for t in trips])
            with st.expander(f"{vehicle_name} (รับผิดชอบ {len(trips)} เที่ยว | ระยะทางรวม {total_dist:.2f} กม. | เก็บขยะรวม {total_vol:.2f} ลบ.ม.)", expanded=True):
                if not trips:
                    st.write("✅ รถคันนี้ไม่ได้ออกปฏิบัติงาน (Standby)")
                for t in trips:
                    st.info(f"📍 {nodes[0]} ➡️ {' ➡️ '.join(t['route'])} ➡️ {nodes[0]} \n\n(ปริมาตร: {t['vol']:.2f} ลบ.ม. | ระยะทางรอบ: {t['dist']:.2f} กม.)")
