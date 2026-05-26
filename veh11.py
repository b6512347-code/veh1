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
import networkx as nx
import xml.etree.ElementTree as ET
import io

st.set_page_config(page_title="Balanced VRP Routing System", layout="wide")

# =====================================================================
# 1. การตั้งค่าฟอนต์ภาษาไทย
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
    ("ภูมิทัศน์(ใหม่)",                       14.86903,  102.02135,  0.3),
    ("สวนพฤกษศาสตร์",                          14.86991,  102.022113, 0.3),
    ("อุทยานผีเสื้อ",                           14.871074, 102.022713, 0.3),
    ("ซินโครตรอน",                              14.872731, 102.023232, 0.1),
    ("อาคารสุรพัฒน์ 2",                         14.8754,   102.02286,  0.2),
    ("อาคารสุรพัฒน์ 3",                         14.874078, 102.022316, 0.4),
    ("เรือนไทย",                                14.875346, 102.021912, 0.3),
    ("อาคารสุรพัฒน์ 1 จุดที่ 1",               14.87584,  102.02302,  0.2),
    ("อาคารสุรพัฒน์ 1 จุดที่ 2",               14.87572,  102.02284,  0.2),
    ("เซเว่น-อีเลฟเว่น เทคโนธานี จุดที่ 1",   14.876072, 102.022745, 0.7),
    ("เซเว่น-อีเลฟเว่น เทคโนธานี จุดที่ 2",   14.876125, 102.022341, 0.4),
    ("ร้านคอกาแฟ ข้าง 7-11 เทคโนธานี",        14.876938, 102.022377, 0.1),
    ("อาคารสุรสัมนาคาร",                        14.876533, 102.024665, 0.1),
    ("โรงอาหารครัวท่านท้าว",                   14.877234, 102.02026,  0.2),
    ("อาคารวิจัยมันสำปะหลัง",                  14.874527, 102.020047, 0.2),
    ("หอดูดาว",                                 14.87414,  102.027598, 0.2),
    ("กาญจนาภิเษก",                             14.873602, 102.026147, 0.5),
]

# =====================================================================
# 2. โมดูล OSM Graph
# =====================================================================
def haversine_dist(lon1, lat1, lon2, lat2):
    R = 6371.0
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
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
            nds      = way.findall('nd')
            way_nodes = [nd.get('ref') for nd in nds]
            for i in range(len(way_nodes) - 1):
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
    nearest_n, min_d = None, float('inf')
    for n in G.nodes:
        d = haversine_dist(lon, lat, G.nodes[n]['x'], G.nodes[n]['y'])
        if d < min_d:
            min_d, nearest_n = d, n
    return nearest_n

def get_distance_matrix_osm(G, locations):
    N = len(locations)
    distance_matrix = np.zeros((N, N))
    snapped = [get_nearest_node(G, item[2], item[1]) for item in locations]
    for i in range(N):
        for j in range(N):
            if i != j:
                try:
                    distance_matrix[i][j] = nx.shortest_path_length(
                        G, source=snapped[i], target=snapped[j], weight='weight')
                except nx.NetworkXNoPath:
                    distance_matrix[i][j] = haversine_dist(
                        locations[i][2], locations[i][1],
                        locations[j][2], locations[j][1])
    return pd.DataFrame(distance_matrix)

# =====================================================================
# 3. OSRM Distance Matrix
# =====================================================================
@st.cache_data(show_spinner=False)
def get_distance_matrix_osrm(locations):
    N         = len(locations)
    distance_matrix = np.zeros((N, N))
    CHUNK     = 50
    coords    = [(item[2], item[1]) for item in locations]

    for i in range(0, N, CHUNK):
        for j in range(0, N, CHUNK):
            src = coords[i:i+CHUNK]
            dst = coords[j:j+CHUNK]
            combined   = src + dst
            coord_str  = ";".join(f"{lon},{lat}" for lon, lat in combined)
            src_idx    = ";".join(str(x) for x in range(len(src)))
            dst_idx    = ";".join(str(x) for x in range(len(src), len(src)+len(dst)))
            url = (f"http://router.project-osrm.org/table/v1/driving/{coord_str}"
                   f"?sources={src_idx}&destinations={dst_idx}&annotations=distance")
            try:
                resp = requests.get(url, timeout=10)
                data = resp.json()
                if data.get("code") == "Ok":
                    distance_matrix[i:i+len(src), j:j+len(dst)] = np.array(data["distances"])
            except Exception:
                pass
            time.sleep(0.5)
    return pd.DataFrame(distance_matrix) / 1000.0

# =====================================================================
# 4. อัลกอริทึม
# =====================================================================

# ---------- Clarke-Wright Savings ----------
def run_savings_algorithm(df_dist, demands, nodes, max_capacity):
    depot     = nodes[0]
    customers = nodes[1:]
    savings   = []
    for i in customers:
        for j in customers:
            if i != j:
                s = df_dist.loc[i, depot] + df_dist.loc[depot, j] - df_dist.loc[i, j]
                if s > 0:
                    savings.append((s, i, j))
    savings.sort(key=lambda x: x[0], reverse=True)

    routes     = [[c] for c in customers]
    route_vols = [demands[c] for c in customers]

    def get_idx(node):
        for k, r in enumerate(routes):
            if node in r:
                return k
        return -1

    for s, i, j in savings:
        ii, jj = get_idx(i), get_idx(j)
        if ii != jj and ii != -1 and jj != -1:
            if routes[ii][-1] == i and routes[jj][0] == j:
                if route_vols[ii] + route_vols[jj] <= max_capacity:
                    routes[ii].extend(routes[jj])
                    route_vols[ii] += route_vols[jj]
                    routes.pop(jj)
                    route_vols.pop(jj)
    return routes, route_vols


def run_balanced_savings_algorithm(df_dist, demands, nodes, max_capacity, num_vehicles):
    depot     = nodes[0]
    customers = nodes[1:]

    total_demand   = sum(demands[c] for c in customers)
    min_trips      = max(1, math.ceil(total_demand / max_capacity))
    eff_vehicles   = max(num_vehicles, min_trips)
    target_cap     = total_demand / eff_vehicles
    soft_cap       = max(target_cap * 1.15, max(demands[c] for c in customers))
    eff_capacity   = min(soft_cap, max_capacity)

    savings = []
    for i in customers:
        for j in customers:
            if i != j:
                s = df_dist.loc[i, depot] + df_dist.loc[depot, j] - df_dist.loc[i, j]
                if s > 0:
                    savings.append((s, i, j))
    savings.sort(key=lambda x: x[0], reverse=True)

    routes     = [[c] for c in customers]
    route_vols = [demands[c] for c in customers]

    def get_idx(node):
        for k, r in enumerate(routes):
            if node in r:
                return k
        return -1

    for s, i, j in savings:
        ii, jj = get_idx(i), get_idx(j)
        if ii != jj and ii != -1 and jj != -1:
            if routes[ii][-1] == i and routes[jj][0] == j:
                if route_vols[ii] + route_vols[jj] <= eff_capacity:
                    routes[ii].extend(routes[jj])
                    route_vols[ii] += route_vols[jj]
                    routes.pop(jj)
                    route_vols.pop(jj)
    return routes, route_vols


# ---------- Sweep helpers (ใหม่) ----------
def _route_dist_by_name(route_names, depot_name, df_dist):
    """ระยะทางรวม depot → stops → depot"""
    full = [depot_name] + route_names + [depot_name]
    return sum(df_dist.loc[full[k], full[k+1]] for k in range(len(full) - 1))


def _two_opt(route_names, depot_name, df_dist, max_iter=500):
    """2-opt local search — สลับลำดับ node เพื่อลดระยะทาง"""
    best = route_names[:]
    for _ in range(max_iter):
        improved = False
        for i in range(len(best) - 1):
            for j in range(i + 2, len(best)):
                candidate = best[:i+1] + best[i+1:j+1][::-1] + best[j+1:]
                if (_route_dist_by_name(candidate, depot_name, df_dist)
                        < _route_dist_by_name(best, depot_name, df_dist)):
                    best, improved = candidate, True
        if not improved:
            break
    return best


# ---------- Sweep Algorithm (แก้ไขแล้ว — มี 2-opt) ----------
def run_sweep_algorithm(locations, demands, nodes, max_capacity, df_dist):
    """
    Sweep Algorithm + 2-opt improvement
    signature เพิ่ม df_dist เพื่อให้ 2-opt คำนวณระยะทางถูกต้อง
    """
    depot_name = nodes[0]
    depot_lat  = locations[0][1]
    depot_lon  = locations[0][2]

    # 1. คำนวณ polar angle และเรียงลำดับ
    customer_angles = []
    for item in locations[1:]:
        name, lat, lon = item[0], item[1], item[2]
        angle = math.degrees(math.atan2(lat - depot_lat, lon - depot_lon)) % 360
        customer_angles.append({"node": name, "angle": angle, "vol": demands[name]})
    customer_angles.sort(key=lambda x: x["angle"])

    # 2. Greedy capacity assignment
    routes, route_vols   = [], []
    current_route, current_vol = [], 0.0

    for c in customer_angles:
        if current_vol + c["vol"] > max_capacity:
            if current_route:
                routes.append(current_route)
                route_vols.append(current_vol)
            current_route = [c["node"]]
            current_vol   = c["vol"]
        else:
            current_route.append(c["node"])
            current_vol  += c["vol"]

    if current_route:
        routes.append(current_route)
        route_vols.append(current_vol)

    # 3. 2-opt improvement
    routes = [_two_opt(r, depot_name, df_dist) for r in routes]

    return routes, route_vols


# ---------- Balanced Sweep (แก้ไขแล้ว — มี 2-opt) ----------
def run_balanced_sweep_algorithm(locations, demands, nodes, max_capacity, df_dist):
    depot_name = nodes[0]
    depot_lat  = locations[0][1]
    depot_lon  = locations[0][2]

    customer_angles = []
    for item in locations[1:]:
        name, lat, lon = item[0], item[1], item[2]
        angle = math.degrees(math.atan2(lat - depot_lat, lon - depot_lon)) % 360
        customer_angles.append({"node": name, "angle": angle, "vol": demands[name]})
    customer_angles.sort(key=lambda x: x["angle"])

    total_demand = sum(c["vol"] for c in customer_angles)
    num_routes   = max(1, math.ceil(total_demand / max_capacity))
    target_cap   = total_demand / num_routes

    routes, route_vols   = [], []
    current_route, current_vol = [], 0.0

    for c in customer_angles:
        if current_vol + c["vol"] > max_capacity:
            routes.append(current_route)
            route_vols.append(current_vol)
            current_route, current_vol = [c["node"]], c["vol"]
        elif current_vol + c["vol"] > target_cap and current_vol >= target_cap * 0.75:
            routes.append(current_route)
            route_vols.append(current_vol)
            current_route, current_vol = [c["node"]], c["vol"]
        else:
            current_route.append(c["node"])
            current_vol  += c["vol"]

    if current_route:
        routes.append(current_route)
        route_vols.append(current_vol)

    # 2-opt improvement
    routes = [_two_opt(r, depot_name, df_dist) for r in routes]

    return routes, route_vols

# =====================================================================
# 5. สร้างแผนที่ Interactive
# =====================================================================
def create_interactive_map(routes, locations, nodes, routing_mode,
                           G_osm=None, map_type='CartoDB positron'):
    depot_coords = (locations[0][1], locations[0][2])

    tiles_mapping = {
        'แผนที่พื้นฐานแบบสว่าง (CartoDB Positron)':        'CartoDB positron',
        'แผนที่พื้นฐานแบบมืด (CartoDB Dark_Matter)':       'CartoDB dark_matter',
        'แผนที่ภูมิประเทศ (OpenStreetMap)':                 'OpenStreetMap',
        'แผนที่ดาวเทียม (Esri Satellite)':
            'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        'แผนที่ภูมิประเทศ+ดาวเทียม (Esri NatGeo)':
            'https://server.arcgisonline.com/ArcGIS/rest/services/NatGeo_World_Map/MapServer/tile/{z}/{y}/{x}',
    }
    selected_tile = tiles_mapping.get(map_type, 'CartoDB positron')

    if 'http' in selected_tile:
        m = folium.Map(location=depot_coords, zoom_start=15,
                       tiles=selected_tile, attr='Esri')
    else:
        m = folium.Map(location=depot_coords, zoom_start=15, tiles=selected_tile)

    colors = ['#FF5733','#335BFF','#28B463','#9B59B6',
              '#E67E22','#1ABC9C','#34495E']

    if G_osm is not None:
        for u, v in G_osm.edges():
            lon1, lat1 = G_osm.nodes[u]['x'], G_osm.nodes[u]['y']
            lon2, lat2 = G_osm.nodes[v]['x'], G_osm.nodes[v]['y']
            folium.PolyLine([(lat1, lon1), (lat2, lon2)],
                            color="#BDC3C7", weight=2, opacity=0.5,
                            dash_array="5, 5").add_to(m)

    folium.Marker(location=depot_coords,
                  popup="<b>DEPOT (ศูนย์กลาง)</b>",
                  icon=folium.Icon(color="red", icon="home")).add_to(m)

    for item in locations[1:]:
        folium.CircleMarker(location=(item[1], item[2]), radius=6,
                            tooltip=item[0], color="#34495E", fill=True,
                            fill_color="#F1C40F", fill_opacity=0.9).add_to(m)

    coords_lonlat = {item[0]: (item[2], item[1]) for item in locations}

    for trip_idx, route_seq in enumerate(routes):
        route_color  = colors[trip_idx % len(colors)]
        full_route   = [nodes[0]] + route_seq + [nodes[0]]
        road_latlon  = []

        if routing_mode == "Local Map (ออฟไลน์ผ่านไฟล์ .osm)" and G_osm is not None:
            for k in range(len(full_route) - 1):
                lon1, lat1 = coords_lonlat[full_route[k]]
                lon2, lat2 = coords_lonlat[full_route[k+1]]
                n1 = get_nearest_node(G_osm, lon1, lat1)
                n2 = get_nearest_node(G_osm, lon2, lat2)
                try:
                    path = nx.shortest_path(G_osm, source=n1, target=n2, weight='weight')
                    for nid in path:
                        road_latlon.append((G_osm.nodes[nid]['y'], G_osm.nodes[nid]['x']))
                except nx.NetworkXNoPath:
                    road_latlon.extend([(lat1, lon1), (lat2, lon2)])
        else:
            coord_str = ";".join(
                f"{coords_lonlat[n][0]},{coords_lonlat[n][1]}" for n in full_route)
            url = (f"http://router.project-osrm.org/route/v1/driving/{coord_str}"
                   f"?overview=full&geometries=geojson")
            try:
                data = requests.get(url, timeout=10).json()
                road_latlon = [(pt[1], pt[0])
                               for pt in data["routes"][0]["geometry"]["coordinates"]]
            except Exception:
                road_latlon = [(coords_lonlat[n][1], coords_lonlat[n][0])
                               for n in full_route]

        if len(road_latlon) > 1:
            plugins.AntPath(locations=road_latlon, color=route_color,
                            weight=5, opacity=0.8,
                            dash_array=[10, 20], delay=800,
                            tooltip=f"Trip {trip_idx+1}").add_to(m)
        time.sleep(0.1)

    return m

# =====================================================================
# 6. Streamlit UI
# =====================================================================
st.title("🚛 Smart Waste Collection Routing System (Balanced Edition)")
st.markdown("ระบบจัดเส้นทางอัจฉริยะ พร้อมฟังก์ชัน **Load Balancing** และปรับรูปแบบแผนที่ได้")

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.header("⚙️ 1. เลือกโครงข่ายถนน")
    routing_mode = st.radio("Routing Engine:",
                            ("OSRM API (ออนไลน์/สาธารณะ)",
                             "Local Map (ออฟไลน์ผ่านไฟล์ .osm)"))

    G_osm = None
    if routing_mode == "Local Map (ออฟไลน์ผ่านไฟล์ .osm)":
        uploaded_osm = st.file_uploader("📂 อัปโหลดไฟล์ .osm", type=["osm"])
        if uploaded_osm is not None:
            with st.spinner("⏳ กำลังอ่านโครงสร้างไฟล์ .osm..."):
                G_osm = build_osm_graph_manual(uploaded_osm.getvalue().decode("utf-8"))
            if G_osm and len(G_osm.edges) > 0:
                st.success(f"✅ โครงข่ายสมบูรณ์! "
                           f"(พิกัด: {len(G_osm.nodes)} | เส้นทาง: {len(G_osm.edges)})")
            else:
                st.error("❌ ไฟล์ที่อัปโหลดไม่มีข้อมูลเส้นทาง")
        else:
            st.warning("⚠️ โปรดอัปโหลดไฟล์ .osm เพื่อดำเนินการต่อ")

    st.header("🗺️ 2. รูปแบบแผนที่")
    map_type = st.selectbox("เลือกประเภทแผนที่:", (
        "แผนที่ภูมิประเทศ (OpenStreetMap)",
        "แผนที่ดาวเทียม (Esri Satellite)",
        "แผนที่พื้นฐานแบบสว่าง (CartoDB Positron)",
        "แผนที่พื้นฐานแบบมืด (CartoDB Dark_Matter)",
        "แผนที่ภูมิประเทศ+ดาวเทียม (Esri NatGeo)",
    ))

    st.header("📍 3. จุดศูนย์กลาง (Depot)")
    depot_name = st.text_input("ชื่อจุด Depot", value="Depot โรงจัดการขยะ")
    depot_lat  = st.number_input("ละติจูด", value=14.862939, format="%.6f")
    depot_lon  = st.number_input("ลองจิจูด", value=102.027903, format="%.6f")

    st.header("⚙️ 4. ยานพาหนะ")
    max_vehicles = st.number_input("จำนวนรถขยะ", min_value=1, value=2)
    max_capacity = st.number_input("ความจุสูงสุด (ลบ.ม.)", min_value=1.0, value=4.5)

    st.header("⚙️ 5. อัลกอริทึม & คาร์บอน")
    algorithm_choice = st.selectbox("เทคนิคการจัดเส้นทาง", (
        "Balanced Clarke-Wright Savings (แนะนำ)",
        "Balanced Workload Sweep",
        "Clarke-Wright Savings (มาตรฐาน)",
        "Sweep Algorithm (มาตรฐาน)",
    ))
    fuel_economy = st.number_input("อัตราสิ้นเปลือง (กม./ลิตร)", value=5.0)
    fuel_price   = st.number_input("ราคาน้ำมัน (บาท/ลิตร)", value=32.94)
    ef_value     = st.number_input("ค่า EF (kgCO₂/ลิตร)", value=2.7446, format="%.4f")
    gwp_value    = st.number_input("ค่า GWP", value=1.0)

# ----------------- MAIN AREA -----------------
st.subheader("📝 1. ข้อมูลพิกัดจุดทิ้งขยะ")

uploaded_coord = st.file_uploader(
    "📂 อัปโหลดไฟล์ Excel/CSV (ถ้ามี)", type=["xlsx", "csv"])

if uploaded_coord is not None:
    df_input = (pd.read_csv(uploaded_coord)
                if uploaded_coord.name.endswith('.csv')
                else pd.read_excel(uploaded_coord))
    st.success("✅ โหลดข้อมูลจากไฟล์สำเร็จ!")
else:
    df_input = pd.DataFrame(DEFAULT_DATA,
                            columns=["Node_Name","Latitude","Longitude","Demand"])

st.markdown("*(ไม่ต้องใส่พิกัด Depot ในตารางนี้)*")
edited_df = st.data_editor(df_input, num_rows="dynamic", use_container_width=True)

if 'show_results' not in st.session_state:
    st.session_state['show_results'] = False

st.markdown("<br>", unsafe_allow_html=True)
start_btn = st.button("🚀 ยืนยันข้อมูลและเริ่มจัดเส้นทาง",
                      type="primary", use_container_width=True)

if start_btn:
    cleaned_df = edited_df.dropna(
        subset=['Node_Name','Latitude','Longitude','Demand'])

    if len(cleaned_df) < 1:
        st.error("❌ ต้องมีจุดเก็บขยะอย่างน้อย 1 จุด")
        st.session_state['show_results'] = False
    elif (routing_mode == "Local Map (ออฟไลน์ผ่านไฟล์ .osm)"
          and (G_osm is None or len(G_osm.edges) == 0)):
        st.error("❌ ยังไม่ได้อัปโหลดไฟล์ .osm หรือโครงข่ายไม่สมบูรณ์")
        st.session_state['show_results'] = False
    else:
        st.session_state['show_results'] = True
        st.session_state['process_data'] = cleaned_df

st.markdown("---")

# =====================================================================
# 7. แสดงผลลัพธ์
# =====================================================================
if st.session_state.get('show_results', False):

    cleaned_df      = st.session_state['process_data']
    total_customers = len(cleaned_df)

    depot_data   = [depot_name, depot_lat, depot_lon, 0.0]
    data_to_use  = [depot_data] + cleaned_df.values.tolist()
    nodes        = [row[0] for row in data_to_use]
    demands      = {row[0]: row[3] for row in data_to_use}
    osrm_fmt     = [(row[0], row[1], row[2], row[3]) for row in data_to_use]

    with st.spinner(f"📡 กำลังประมวลผลด้วย {algorithm_choice}..."):

        # --- Distance Matrix ---
        if routing_mode == "Local Map (ออฟไลน์ผ่านไฟล์ .osm)":
            df_dist = get_distance_matrix_osm(G_osm, osrm_fmt)
        else:
            df_dist = get_distance_matrix_osrm(osrm_fmt)
        df_dist.columns = df_dist.index = nodes

        # --- อัลกอริทึม ---
        if algorithm_choice == "Clarke-Wright Savings (มาตรฐาน)":
            routes, route_vols = run_savings_algorithm(
                df_dist, demands, nodes, max_capacity)

        elif algorithm_choice == "Balanced Clarke-Wright Savings (แนะนำ)":
            routes, route_vols = run_balanced_savings_algorithm(
                df_dist, demands, nodes, max_capacity, max_vehicles)

        elif algorithm_choice == "Balanced Workload Sweep":
            routes, route_vols = run_balanced_sweep_algorithm(
                osrm_fmt, demands, nodes, max_capacity, df_dist)

        else:  # Sweep Algorithm (มาตรฐาน)
            routes, route_vols = run_sweep_algorithm(
                osrm_fmt, demands, nodes, max_capacity, df_dist)

        # --- ระยะทางและคาร์บอน ---
        route_distances = []
        for r in routes:
            full = [nodes[0]] + r + [nodes[0]]
            route_distances.append(
                sum(df_dist.loc[full[k], full[k+1]] for k in range(len(full)-1)))

        grand_total   = sum(route_distances)
        activity_A    = grand_total / fuel_economy
        carbon_E      = activity_A * ef_value * gwp_value
        fuel_cost     = activity_A * fuel_price

        # --- Fleet Balancing (Min-Max Greedy) ---
        trip_data = [{"original_idx": i+1, "route": routes[i],
                      "vol": route_vols[i], "dist": route_distances[i]}
                     for i in range(len(routes))]
        trip_data.sort(key=lambda x: x['dist'], reverse=True)

        fleet_schedule    = {f"🚛 รถขยะคันที่ {i+1}": [] for i in range(int(max_vehicles))}
        vehicle_workloads = {f"🚛 รถขยะคันที่ {i+1}": 0.0 for i in range(int(max_vehicles))}

        for t in trip_data:
            best = min(vehicle_workloads, key=vehicle_workloads.get)
            t['trip_sequence'] = len(fleet_schedule[best]) + 1
            fleet_schedule[best].append(t)
            vehicle_workloads[best] += t['dist']

        # --- Dashboard ---
        st.subheader("📊 2. สรุปผลการปฏิบัติงาน")
        st.success("✅ วิเคราะห์และออกแบบเส้นทางเสร็จสมบูรณ์!")

        c1, c2, c3 = st.columns(3)
        c1.metric("📌 จุดเก็บขยะ",      f"{total_customers} จุด")
        c2.metric("🚛 รอบที่ต้องวิ่ง",   f"{len(routes)} เที่ยว")
        c3.metric("🗑️ ปริมาตรรวม",      f"{sum(route_vols):.2f} ลบ.ม.")

        c4, c5, c6 = st.columns(3)
        c4.metric("📍 ระยะทางรวม",       f"{grand_total:.2f} กม.")
        c5.metric("🌿 คาร์บอน (CO₂e)",   f"{carbon_E:.2f} kg")
        c6.metric("⛽ ต้นทุนน้ำมัน",     f"฿ {fuel_cost:,.2f}")

        # --- แผนที่ ---
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("🗺️ 3. แผนที่จำลองการเดินรถ")
        with st.spinner("กำลังเรนเดอร์แผนที่..."):
            m = create_interactive_map(
                routes, osrm_fmt, nodes, routing_mode, G_osm, map_type)
            st_folium(m, width=1200, height=600, returned_objects=[])

        # --- ตารางปฏิบัติงาน ---
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("📋 4. ตารางปฏิบัติงานของรถแต่ละคัน")

        for vehicle_name, trips in fleet_schedule.items():
            total_d = sum(t['dist'] for t in trips)
            total_v = sum(t['vol']  for t in trips)
            with st.expander(
                f"{vehicle_name} — {len(trips)} เที่ยว | "
                f"{total_d:.2f} กม. | {total_v:.2f} ลบ.ม.",
                expanded=True
            ):
                if not trips:
                    st.write("✅ รถคันนี้ไม่ได้ออกปฏิบัติงาน (Standby)")
                for t in trips:
                    st.info(
                        f"📍 {nodes[0]} ➡️ {' ➡️ '.join(t['route'])} ➡️ {nodes[0]}\n\n"
                        f"ปริมาตร: {t['vol']:.2f} ลบ.ม. | ระยะทาง: {t['dist']:.2f} กม."
                    )
