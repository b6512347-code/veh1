# =====================================================================
# 🗺️ 6. โมดูลสร้างแผนที่ Interactive (Folium)
# =====================================================================
def create_interactive_map(routes, locations, nodes, routing_mode, G_qgis=None, qgis_file_path=None):
    depot_coords = (locations[0][1], locations[0][2])
    m = folium.Map(location=depot_coords, zoom_start=15, tiles='CartoDB positron')
    colors = ['#FF5733', '#335BFF', '#28B463', '#9B59B6', '#E67E22', '#1ABC9C', '#34495E']
    
    # ⚠️ แก้ปัญหาที่ 2: บังคับวาดเส้นถนน QGIS ให้ปรากฏบน Web Map
    if routing_mode == "QGIS Custom Map (ออฟไลน์)" and qgis_file_path is not None and os.path.exists(qgis_file_path):
        try:
            # โหลดไฟล์ด้วย GeoPandas
            gdf_bg = gpd.read_file(qgis_file_path)
            
            # บังคับเช็คและแปลง CRS เป็น 4326 (WGS84) เสมอ
            if gdf_bg.crs is None:
                gdf_bg.set_crs(epsg=4326, inplace=True)
            elif gdf_bg.crs.to_epsg() != 4326:
                gdf_bg = gdf_bg.to_crs(epsg=4326)
            
            # บังคับแปลงเป็น JSON String ก่อนส่งให้ Folium (วิธีนี้เสถียรที่สุด)
            geo_json_data = gdf_bg.to_json()
            
            folium.GeoJson(
                geo_json_data,
                name="โครงข่ายถนน QGIS (Custom Map)",
                style_function=lambda x: {'color': '#7F8C8D', 'weight': 3, 'opacity': 0.8, 'dashArray': '4, 6'}
            ).add_to(m)
        except Exception as e:
            st.warning(f"⚠️ ไม่สามารถวาดกราฟิกถนนพื้นหลัง QGIS ได้: {e}")

    # ปักหมุด Depot และจุดทิ้งขยะ
    folium.Marker(location=depot_coords, popup="<b>DEPOT</b>", icon=folium.Icon(color="red", icon="home")).add_to(m)
    for item in locations[1:]:
        folium.CircleMarker(location=(item[1], item[2]), radius=6, tooltip=item[0], color="#34495E", fill=True, fill_color="#F1C40F", fill_opacity=0.9).add_to(m)

    coords_lonlat = {item[0]: (item[2], item[1]) for item in locations}

    for trip_idx, route_seq in enumerate(routes):
        route_color = colors[trip_idx % len(colors)]
        full_route = [nodes[0]] + route_seq + [nodes[0]]
        
        road_coords_latlon = []
        if routing_mode == "QGIS Custom Map (ออฟไลน์)" and G_qgis is not None:
            # วาดเส้นทับไปตามเส้นทางคณิตศาสตร์ NetworkX
            for k in range(len(full_route)-1):
                pt1 = coords_lonlat[full_route[k]]
                pt2 = coords_lonlat[full_route[k+1]]
                n1, _ = get_nearest_node(G_qgis, pt1)
                n2, _ = get_nearest_node(G_qgis, pt2)
                try:
                    path = nx.shortest_path(G_qgis, source=n1, target=n2, weight='weight')
                    road_coords_latlon.extend([(pt[1], pt[0]) for pt in path])
                except nx.NetworkXNoPath:
                    road_coords_latlon.extend([(pt1[1], pt1[0]), (pt2[1], pt2[0])])
        else:
            # วาดเส้นตาม OSRM API
            coords_string = ";".join([f"{coords_lonlat[n][0]},{coords_lonlat[n][1]}" for n in full_route])
            url = f"http://router.project-osrm.org/route/v1/driving/{coords_string}?overview=full&geometries=geojson"
            try:
                data = requests.get(url).json()
                road_coords_lonlat = data["routes"][0]["geometry"]["coordinates"]
                road_coords_latlon = [(pt[1], pt[0]) for pt in road_coords_lonlat]
            except:
                road_coords_latlon = [(coords_lonlat[n][1], coords_lonlat[n][0]) for n in full_route]
        
        # วาด Animation วิ่งตามถนน
        if len(road_coords_latlon) > 1:
            plugins.AntPath(locations=road_coords_latlon, color=route_color, weight=5, opacity=0.8, dash_array=[10, 20], delay=800, tooltip=f"Trip {trip_idx+1}").add_to(m)
        time.sleep(0.1)
    
    # เพิ่มตัวควบคุม Layer ให้ผู้ใช้เลือกเปิด/ปิดเส้น QGIS ได้
    folium.LayerControl().add_to(m)
    return m
