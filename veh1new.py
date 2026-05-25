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

# Set page config ให้แสดงผลกว้างเต็มตา (ต้องอยู่บนสุดเสมอ)
st.set_page_config(page_title="VRP Garbage Routing", layout="wide")

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
        st.error(f"เกิดข้อผิดพลาดในการแปล
