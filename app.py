#merci pour ton aide dans l'avancé du projet .
#à present un petit problème se pose , les action de la partie se font uniquement que si ce sont les cordonnées latittude et longitude que l'utilisateur renseigne, lors que cela doit etre fait dans les deux cas c'est à dire que ce soit les coordonnées ou le shape du lieu d'activité.

    # -*- coding: utf-8 -*-
import streamlit as st
import geopandas as gpd
import leafmap.foliumap as leafmap
from shapely.geometry import Point
from pystac_client import Client
import planetary_computer
import rasterio
import numpy as np
import base64
import rasterio
import pandas as pd
import time 
import folium
import ee
import tempfile
import zipfile
import os
import imageio
import tempfile
from pyproj import Transformer
from fonction import get_base64_image,compute_indices,fix_crs,compute_nearest,load_clients,get_color,get_name_field,get_stress_color,run_change_detection,build_dynamic_world_map
from fonction import get_client_geometry,load_uploaded_geometry,create_hls_timeseries,get_best_sentinel_pair,save_raster,compute_ndvi_raster,dynamic_world_change
from fonction import get_dynamic_world_series,dynamic_world_timelapse
    # Initialisation Earth Engine
try:
        ee.Initialize(project='ancient-lattice-491308-n6')
except:
        ee.Authenticate()
        ee.Initialize()

st.set_page_config(layout="wide")

    # Initialisation session_state
if "year" not in st.session_state:
        st.session_state["year"] = 2023

logo_base64 = get_base64_image("logo_nbci.png")


st.markdown(
        f"""
        <style>

        /* Cacher le header Streamlit (barre noire) */
        header {{
            visibility: hidden;
        }}

        /* Décaler le contenu principal */
        .block-container {{
            padding-top: 90px;
        }}

        /* Navbar fixe */
        .navbar {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 70px;
            background-color: #0a1f44;
            display: flex;
            align-items: center;
            z-index: 9999;
            padding: 0 20px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }}

        .navbar img {{
            max-width: 500px;
            object-fit: contain; 
            height: 45px;
            margin-right: 15px;
        }}

        .navbar-title {{
            color: white;
            font-size: 24px;
            font-weight: bold;
            margin-right: 40px;
        }}

        .marquee {{
            color: white;
            font-size: 20px;
            font-weight: 500;
            white-space: nowrap;
            overflow: hidden;
            width: 100%;
        }}

        .marquee span {{
            display: inline-block;
            padding-left: 100%;
            animation: scroll-left 15s linear infinite;
        }}

        @keyframes scroll-left {{
            0% {{ transform: translateX(0); }}
            100% {{ transform: translateX(-100%); }}
        }}

        /* Sidebar alignée */
        section[data-testid="stSidebar"] {{
            top: 70px;
            height: calc(100vh - 70px);
        }}

        section[data-testid="stSidebar"] > div {{
            padding-top: 20px;
        }}

        </style>

        <div class="navbar">
            <img src="data:image/png;base64,{logo_base64}">
            <div class="navbar-title">NSIAGEO</div>
            <div class="marquee">
                <span>
                Bienvenue sur NSIAGEO - Suivez l’impact environnemental des clients à financer en temps réel
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown(
        "<h1 style='text-align: center;'>Analyse environnementale</h1>",
        unsafe_allow_html=True
    )

    # -------------------------
    # Animation en haut de page (juste sous le titre)
    # -------------------------
    #st.subheader("Animations en direct")
col_anim1, col_anim2 = st.columns(2)
with col_anim1:
        st.image(
            "https://github.com/giswqs/data/raw/main/timelapse/goes.gif",
            caption="Analyse environnementale en action",
            width=700
        )
with col_anim2:
        st.image(
            "https://github.com/giswqs/data/raw/main/timelapse/fire.gif",
            caption="Timelapse satellite",
            width=700
        )
    # -------------------------
    # Chargement données
    # -------------------------
forest_path = r"C:\Users\Yao Simlin\Downloads\GeoDataAnalystProject\BDGEO_Base_de donnée_CI\disk_2\aires-protegees\forest.shp"
water_path = r"D:\GeoDataAnalystProject\BDGEO_Base_de donnée_CI\Shape CI\waterways\waterways.shp"
bassin_path = r"C:\Users\Yao Simlin\Documents\Bassin.shp"
lake_path =r"C:\Users\Yao Simlin\Documents\Lake.shp"
parcs_path = r"C:\Users\Yao Simlin\Documents\parcs.shp"
hydro_path = r"C:\Users\Yao Simlin\Documents\hydrographie.shp"
parcs_path = r"C:\Users\Yao Simlin\Documents\parcs.shp"
integrale_path = r"C:\Users\Yao Simlin\Documents\integrale.shp"

@st.cache_data(ttl=3600)
def load_data():

        lake = fix_crs(gpd.read_file(lake_path))
        waterways = fix_crs(gpd.read_file(water_path))
        parcs = fix_crs(gpd.read_file(parcs_path))
        forets = fix_crs(gpd.read_file(forest_path))
        bassin = fix_crs(gpd.read_file(bassin_path))
        hydro = fix_crs(gpd.read_file(hydro_path))
        integrale = fix_crs(gpd.read_file(integrale_path))

        # -------------------------
        # 🌍 KBA GLOBAL → FILTRE CI
        # -------------------------
        kba_path = r"C:\Users\Yao Simlin\Downloads\KBA_Data\KBAsGlobal_2025_September_02\KBAsGlobal_2025_September_02_POL.shp"
        kba = gpd.read_file(kba_path)

        # Bounding box Côte d'Ivoire
        bbox = (-8.6, 4.3, -2.5, 10.8)

        # Filtrage rapide
        kba = kba.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]

        kba = kba.to_crs(epsg=4326)

        stress_path = r"C:\Users\Yao Simlin\Downloads\Stress_hydrique.shp"
        stress_hydrique = gpd.read_file(stress_path)
        stress_hydrique = stress_hydrique.to_crs(epsg=4326)
        stress_hydrique = stress_hydrique[[
            "bws_score",
            "bws_label",
            "geometry"
        ]]
        stress_hydrique = stress_hydrique.rename(columns={
            "bws_score": "Score stress hydrique",
            "bws_label": "Niveau stress hydrique"
        })
        stress_hydrique["color"] = stress_hydrique["Niveau stress hydrique"].apply(get_stress_color)
        return lake, waterways, parcs, forets, bassin, hydro, integrale, kba, stress_hydrique

lake, waterways, parcs, forets, bassin, hydro, integrale, kba, stress_hydrique = load_data()


@st.cache_data(ttl=3600)
def create_timelapse(lat, lon):
        catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

        bbox = [lon - 0.05, lat - 0.05, lon + 0.05, lat + 0.05]

        search = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime="2023-01-01/2026-03-31",
            query={"eo:cloud_cover": {"lt": 20}},
        )

        items = list(search.items())[:5]

        images = []

        for item in items:
            try:
                item = planetary_computer.sign(item)

                with rasterio.open(item.assets["visual"].href) as src:
                    img = src.read([1,2,3])
                    img = np.transpose(img, (1,2,0))
                    img = (img / img.max() * 255).astype(np.uint8)

                    images.append(img)

            except:
                continue

        gif_path = "timelapse.gif"
        imageio.mimsave(gif_path, images, duration=1)

        return gif_path

# On intègre ici les critères ESG restant

ESG_RULES = {
    "Mines": {
        "zones": [(1, "very high"), (10, "high"), (50, "medium")],
        "screening": 50
    },
    "Energie thermique": {
        "zones": [(1, "very high"), (10, "high"), (50, "medium")],
        "screening": 50
    },
    "Energie renouvelable": {
        "zones": [(0.5, "very high"), (5, "high"), (10, "medium")],
        "screening": 50
    },
    "Industrie": {
        "zones": [(0.5, "very high"), (5, "high"), (10, "medium")],
        "screening": 50
    },
    "Agriculture": {
        "zones": [(0.5, "very high"), (5, "high"), (10, "medium")],
        "screening": 50
    },
    "BTP bâtiment": {
        "zones": [(0.2, "very high"), (1, "high"), (5, "medium")],
        "screening": 10
    },
    "Infrastructure linéaire": {
        "zones": [(0.5, "very high"), (5, "high"), (20, "medium")],
        "screening": 50
    }
}

def map_secteur(secteur):
    if secteur in ["Metals & Mining", "Oil, Gas & Consumable Fuels"]:
        return "Mines"

    elif secteur in ["Construction Materials Production"]:
        return "Industrie"

    elif secteur in ["Agriculture (Plant products)", "Food & Beverage Production"]:
        return "Agriculture"

    elif secteur in ["Transportation Services"]:
        return "Infrastructure linéaire"

    elif secteur in ["Land developpement & Construction"]:
        return "BTP bâtiment"

    else:
        return "Industrie"

def get_risk_level(distance_km, secteur):
        
    secteur_map = map_secteur(secteur)
    config = ESG_RULES.get(secteur_map)

    if config is None:
        return "low", False

    zones = config["zones"]
    screening_limit = config["screening"]

    # 🎯 Risque principal
    for threshold, level in zones:
        if distance_km <= threshold:
            return level, False

    # 🛰️ Filet de screening
    if distance_km <= screening_limit:
        return "screening", True

    return "low", False
    # -------------------------
    # Saisie utilisateur
    # -------------------------
lat = st.sidebar.number_input("Latitude", value=5.36, format="%.6f")
lon = st.sidebar.number_input("Longitude", value=-4.01, format="%.6f")

geom = None
use_shape = False

secteurs = [
        "Agriculture (Plant products)",
        "Appliances & General Goods Manufacturing",
        "Automotive, Electrical Equipment & Machine Production",
        "Chemicals & Electrical Equipment & Machinery Production",
        "Chemicals & Other Materiels Production",
        "Construction Materials Production",
        "Food & Beverage Production",
        "Food Retailing",
        "General or Speciality Retailing",
        "Health Care, Pharmaceuticals and Biotechnology",
        "Land developpement & Construction",
        "Metals & Mining",
        "Offices & professional services",
        "Oil, Gas & Consumable Fuels",
        "Telecommunication services",
        "Transportation Services",
        "Water utilities / Water Service Providers",
        "Other"
    ]

    # ✅ Ajout valeur par défaut
secteur = st.sidebar.selectbox(
        "Secteur d'activité",
        ["-- Sélectionner un secteur --"] + secteurs
    )

uploaded_shape = st.sidebar.file_uploader(
    "Uploader zone d'activité (shapefile .zip ou geojson)",
    type=["zip", "geojson", "json","shp"]
)

analyser = st.sidebar.button("Analyser")

geometry = None
gdf_client = None
    # ✅ Validation + stockage
if analyser:

    if secteur == "-- Sélectionner un secteur --":
        st.warning("Veuillez sélectionner un secteur d'activité")
        st.stop()

    # -------------------------
    # 🔥 CONSTRUCTION GEOMETRIE ICI (IMPORTANT)
    # -------------------------

    if uploaded_shape is not None:

        gdf = load_uploaded_geometry(uploaded_shape)

        if gdf is None or gdf.empty:
            st.error("Shapefile invalide")
            st.stop()

        gdf["geometry"] = gdf["geometry"].buffer(0)

        geom = gdf.unary_union
        geom_m = gpd.GeoSeries([geom], crs=4326).to_crs(32630).iloc[0]

        use_shape = True
        gdf_client = gdf

    else:
        geom = Point(lon, lat)
        geom_m = gpd.GeoSeries([geom], crs=4326).to_crs(32630).iloc[0]

        use_shape = False
        gdf_client = None

    # -------------------------
    # 💾 STOCKAGE SESSION
    # -------------------------
    st.session_state["geometry"] = geom
    st.session_state["geom_m"] = geom_m
    st.session_state["use_shape"] = use_shape
    st.session_state["gdf_client"] = gdf_client

    st.session_state["secteur"] = secteur
    st.session_state["lat"] = lat
    st.session_state["lon"] = lon
    st.session_state["analyzed"] = True
#     # =========================
# # 🎯 CONSTRUCTION GEOMETRIE UNIQUE
# # =========================

# # Définir geom par défaut
# geom = Point(lon, lat)
# use_shape = False

# geom = None
# geom_m = None
# use_shape = False
# gdf_client = None

# if uploaded_shape is not None:
#     gdf = load_uploaded_geometry(uploaded_shape)
    
#     if gdf is None or gdf.empty:
#         st.error("Shapefile invalide")
#         st.stop()
    
#     # nettoyage
#     gdf["geometry"] = gdf["geometry"].buffer(0)
    
#     # Crée la géométrie unique du shapefile
#     geom = gdf.unary_union
#     geom_m = gpd.GeoSeries([geom], crs=4326).to_crs(32630).iloc[0]
#     use_shape = True
#     gdf_client = gdf

# else:
#     # Mode point
#     geom = Point(lon, lat)
#     geom_m = gpd.GeoSeries([geom], crs=4326).to_crs(32630).iloc[0]
#     use_shape = False
#     gdf_client = None

# # Stockage dans session_state
# st.session_state["geometry"] = geom
# st.session_state["geom_m"] = geom_m
# st.session_state["use_shape"] = use_shape
# st.session_state["gdf_client"] = gdf_client
    # Animation sidebar (en bas)
    # -------------------------
st.sidebar.markdown("---")
#st.sidebar.image(
#        "https://github.com/giswqs/data/raw/main/timelapse/fire.gif",
#        caption="Timelapse satellite",
#        width=250
#    )

mode = st.sidebar.radio(
        "Navigation",
        ["🗺️ Carte & Analyse", "📈 Impact environnemental", "📍 Position"]
    )
if mode == "🗺️ Carte & Analyse":

        m = leafmap.Map(center=[7.5, -5.5], zoom=15)
        # -------------------------
        # Carte Split map
        # -------------------------
        #m.split_map(
        #    left_layer="ESA WorldCover 2020 S2 FCC",
        #    right_layer="ESA WorldCover 2020"
        #)
        def style_protected_areas(feature):
            nature = feature["properties"].get("NATURE", "")

            if "Parcs Nationaux" in nature:
                return {
                    "color": "purple",
                    "fillColor": "purple",
                    "fillOpacity": 0.5,
                    "weight": 1
                }

            elif "Forêts classées" in nature or "Foret classee" in nature:
                return {
                    "color": "gray",
                    "fillColor": "gray",
                    "fillOpacity": 0.2,
                    "weight": 1,
                    "dashArray": "5, 5"  # effet rayé (simulation)
                }

            else:
                return {
                    "color": "green",
                    "fillOpacity": 0.3
                }
        m.add_gdf(
            stress_hydrique,
            layer_name="Stress hydrique",
            style_function=lambda x: {
                "fillColor": x["properties"]["color"],
                "color": "black",
                "weight": 1,
                "fillOpacity": 0.7
            },
            info_mode="on_hover"
        )
        m.add_gdf(
            bassin,
            layer_name ="Bassin versant",
            style={"color": "orange"}
        )
        m.add_gdf(hydro, layer_name="Reseau hydro", style={"color": "gray"})
        m.add_gdf(
            forets,
            layer_name="Aires protégées",
            style={"color": "green"}
        )
        m.add_gdf(integrale, layer_name="Reserves integrale", style={"color": "red"})
        m.add_gdf(parcs, layer_name="Reserves", style={"color": "red"})
        m.add_gdf(waterways, layer_name="Cours d'eau", style={"color": "blue"})
        m.add_gdf(
        kba,
        layer_name="Zones KBA",
        style={
            "color": "purple",
            "fillColor": "purple",
            "fillOpacity": 0.2
        }
            )
        m.add_gdf(lake, layer_name="Lacs", style={"color": "navy"})
        
        # -------------------------
        # 🌍 ESRI LULC
        # -------------------------
        
        #m.add_legend(title="Occupation du sol", builtin_legend="ESA_WorldCover")

        if st.session_state.get("geometry") is not None:
            geom = st.session_state["geometry"]
            geom_m = st.session_state["geom_m"]
            use_shape = st.session_state["use_shape"]
        else:
            geom = Point(lon, lat)
            geom_m = gpd.GeoSeries([geom], crs=4326).to_crs(32630).iloc[0]
            use_shape = False
                    # -------------------------
        # Analyse client
        # -------------------------
        if "analyzed" not in st.session_state:
            st.session_state["analyzed"] = False
        if analyser:
            st.session_state["analyzed"] = True
        if st.session_state.get("analyzed", False):
            point_client = Point(lon, lat)
            forets_m = forets.to_crs(epsg=32630)
            water_m = waterways.to_crs(epsg=32630)
            parcs_m = parcs.to_crs(epsg=32630)
            lake_m = lake.to_crs(epsg=32630)
            kba_m = kba.to_crs(epsg=32630)
            point_m = gpd.GeoSeries([point_client], crs=4326).to_crs(32630)
            if "geometry" not in st.session_state:
                st.warning("Veuillez lancer une analyse")
                st.stop()

            geom = st.session_state["geometry"]
            geom_m = st.session_state["geom_m"]
            use_shape = st.session_state["use_shape"]

            forets_m["dist"] = forets_m.distance(geom_m)
            idx_f = forets_m["dist"].idxmin()
            distance_foret_km = forets_m.loc[idx_f, "dist"] / 1000
            foret_nom = str(forets.loc[idx_f, "NOM_FORET"])

            kba_m["dist"] = kba_m.distance(geom_m)
            idx_kba = kba_m["dist"].idxmin()
            distance_kba_km = kba_m.loc[idx_kba, "dist"] / 1000
            kba_nom = str(kba.loc[idx_kba, "NatName"])

            parcs_m["dist"] = parcs_m.distance(geom_m)
            idx_parc = parcs_m["dist"].idxmin()
            distance_parcs_km = parcs_m.loc[idx_parc, "dist"] / 1000
            name_field_parc = get_name_field(parcs)
            if name_field_parc:
                parc_nom = str(parcs.loc[idx_parc, name_field_parc])
            else:
                parc_nom = "Non renseigné"

            if forets_m.intersects(geom_m).any():
                distance_foret_km = 0
                foret_nom = "Zone intersectée ⚠️"
            else:
                forets_m["dist"] = forets_m.distance(geom_m)
                idx_foret = forets_m["dist"].idxmin()
                distance_foret_km = forets_m.loc[idx_foret, "dist"] / 1000
                foret_nom = str(forets.loc[idx_foret, "NOM_FORET"])

            water_m["dist"] = water_m.distance(geom_m)
            idx_w = water_m["dist"].idxmin()
            distance_water_km = water_m.loc[idx_w, "dist"] / 1000
            water_name = str(waterways.loc[idx_w, "name"]) if waterways.loc[idx_w, "name"] else "Non renseigne"

            lake_m["dist"] = lake_m.distance(geom_m)
            idx_w = lake_m["dist"].idxmin()
            distance_lac_km = lake_m.loc[idx_w, "dist"] / 1000
            name_field_lake = get_name_field(lake)
            lake_name = str(lake.loc[idx_w, name_field_lake]) if name_field_lake else "Non renseigné"

            tooltip_text = (
                f"Client\n"
                f"Lat: {lat:.5f}, Lon: {lon:.5f}\n"
                f"Forêt: {foret_nom} ({distance_foret_km:.2f} km)\n"
                f"Parc: {parc_nom} ({distance_parcs_km:.2f} km)\n"
                f"Eau: {water_name} ({distance_water_km:.2f} km)\n"
                f"Lac: {lake_name} ({distance_lac_km:.2f} km)\n"
                f"KBA:{kba_nom} ({distance_kba_km:.2f} km)"
            )
            # Nettoyage UTF-8 sécurisé
            if use_shape:
                tooltip_text = (
                    f"Zone client\n"
                    f"Forêt: {foret_nom} ({distance_foret_km:.2f} km)\n"
                    f"Parc: {parc_nom} ({distance_parcs_km:.2f} km)"
                )
            else:
                tooltip_text = (
                    f"Client\n"
                    f"Lat: {lat:.5f}, Lon: {lon:.5f}\n"
                    f"Forêt: {foret_nom} ({distance_foret_km:.2f} km)"
                )
            # -------------------------
            # Résultats
            # -------------------------
            st.subheader("Résultats environnementaux")
            st.write(f"Foret clasée la plus proche : {foret_nom} ({distance_foret_km:.2f} km)")
            st.write(f"Cours d'eau le plus proche : {water_name} ({distance_water_km:.2f} km)")
            st.write(f"Parc le plus proche : {parc_nom} ({distance_parcs_km:.2f} km)")
            st.write(f"Lac le plus proche : {lake_name} ({distance_lac_km:.2f} km)")
            st.write(f"zone de kbas la plus proche : {kba_nom} ({distance_kba_km:.2f} km)")

            st.subheader("Diagnostic RSE")

            secteur = st.session_state.get("secteur", "Other")

            distances = [
                distance_foret_km,
                distance_parcs_km,
                distance_kba_km,
                distance_water_km
            ]

            min_distance = min(distances)

            risk, screening_flag = get_risk_level(min_distance, secteur)

            if risk == "very high":
                st.error("🔴 Risque ESG très élevé (zone directe)")
                
            elif risk == "high":
                st.warning("🟠 Risque ESG élevé (zone indirecte)")
                
            elif risk == "medium":
                st.info("🟡 Risque ESG modéré (effet cumulatif)")
                
            elif risk == "screening":
                st.info("🛰️ Zone de vigilance ESG (screening 50 km)")
                
            else:
                st.success("🟢 Risque ESG faible")

            # -------------------------
            # Analyse satellite
            # -------------------------
            with st.spinner("Analyse satellite en cours..."):
                # Géométrie en mètres
                geom = st.session_state["geometry"]
                geom_m = st.session_state["geom_m"]
                use_shape = st.session_state["use_shape"]

                if use_shape:
                    zone_analysis = geom
                else:
                    buffer_m = geom_m.buffer(1000)
                    zone_analysis = gpd.GeoSeries([buffer_m], crs=32630).to_crs(4326).iloc[0]

                ndvi_value, ndwi_value = compute_indices(zone_analysis)
                st.subheader("Analyse NDVI / NDWI")
                st.write(f"NDVI moyen : {ndvi_value:.2f}" if ndvi_value is not None else "NDVI non disponible")
                st.write(f"NDWI moyen : {ndwi_value:.2f}" if ndwi_value is not None else "NDWI non disponible")
        # =========================
        # AJOUT SHAPE CLIENT
        # =========================
        if st.session_state.get("gdf_client") is not None:

            gdf_client = st.session_state["gdf_client"]

            st.write("DEBUG - shape chargé:", len(gdf_client))

            m.add_gdf(
                gdf_client,
                layer_name="Zone client",
                style={
                    "color": "black",
                    "weight": 3,
                    "fillColor": "black",
                    "fillOpacity": 0.1
                }
            )

            centroid = gdf_client.unary_union.centroid

            m.add_marker(
                location=[centroid.y, centroid.x],
                popup="Centre zone client",
                icon=folium.Icon(color="black")
            )

            m.center = [centroid.y, centroid.x]
            m.zoom = 14

        else:
            m.add_marker(location=[lat, lon])
            m.center = [lat, lon]
            m.zoom = 15
            
        m.to_streamlit(height=700)
        
elif mode == "📈 Impact environnemental":

    if "geometry" not in st.session_state or st.session_state["geometry"] is None:
        st.error("Veuillez d'abord analyser une zone (point ou shapefile)")
        st.stop()

    geometry = st.session_state["geometry"]
    lat = st.session_state["lat"]
    lon = st.session_state["lon"]
    use_shape = st.session_state["use_shape"]
    gdf_client = st.session_state.get("gdf_client", None)

    if st.button("▶️ Lancer l'animation spatio-temporelle"):

        with st.spinner("Animation en cours..."):

            images, ee_geom = get_dynamic_world_series(geometry)

            if images is None:
                st.error("Impossible de générer l'animation (géométrie invalide)")
                st.stop()

            vis = {
                "min": 0,
                "max": 8,
                "palette": [
                    "419BDF","397D49","88B053","7A87C6",
                    "E49635","DFC35A","C4281B","A59B8F","B39FE1"
                ]
            }

            progress_bar = st.progress(0)
            year_text = st.empty()
            map_placeholder = st.empty()

            total = len(images)

            for i, (year, img) in enumerate(images):

                try:
                    ee.Initialize(project='ancient-lattice-491308-n6')
                except Exception:
                    ee.Authenticate()
                    ee.Initialize(project='ancient-lattice-491308-n6')

                import geemap.foliumap as geemap
                m = geemap.Map()
                coords = ee_geom.bounds().getInfo()["coordinates"][0]
                minx = min([c[0] for c in coords])
                maxx = max([c[0] for c in coords])
                miny = min([c[1] for c in coords])
                maxy = max([c[1] for c in coords])

                m.fit_bounds([[miny, minx], [maxy, maxx]])
                m.addLayer(img, vis, f"Dynamic World {year}")

                # ✅ CORRECTION ICI
                with map_placeholder:
                    m.to_streamlit(height=600)

                # progression
                progress_bar.progress((i + 1) / total)

                year_text.markdown(f"### Année : {year}")

                time.sleep(1.5)

    #st.markdown("---")
    #st.subheader("🎬 GIF D'EVOLUTION DE L'OCCUPATION")

    # 🔥 IMPORTANT : vérifier analyse
    if not st.session_state.get("analyzed", False):
        st.warning("Veuillez d'abord analyser un client")
        st.stop()

    # 🔥 récupérer geometry UNE SEULE FOIS
    geometry = st.session_state.get("geometry", None)

    if geometry is None:
        st.error("Aucune géométrie disponible")
        st.stop()

    if st.button("▶️ Générer le film d'évolution"):

        with st.spinner("Génération du timelapse Dynamic World..."):

            gif_url = dynamic_world_timelapse(geometry)

            if gif_url:
                st.image(gif_url, caption="Évolution occupation du sol (2020-2026)")
            else:
                st.warning("Impossible de générer l'animation")

        st.subheader("Détection des changements environnementaux")

        if not st.session_state.get("analyzed", False):
            st.warning("Veuillez d'abord analyser un client")
            st.stop()

        lat = st.session_state["lat"]
        lon = st.session_state["lon"]
        geometry = st.session_state["geometry"]
        use_shape = st.session_state["use_shape"]
        gdf_client = st.session_state.get("gdf_client", None)


    if st.button("🌍 Analyse Occupation du sol"):

        with st.spinner("Analyse IA du sol en cours..."):

            # =========================
            # 🌍 DATA DYNAMIC WORLD
            # =========================
            dw_start, dw_end, change, stats, stats_start, stats_end = dynamic_world_change(geometry)

            if stats is None or stats_start is None or stats_end is None:
                st.error("Aucune donnée disponible pour cette zone")
                st.stop()

            start_dict = stats_start.getInfo().get("label", {})
            end_dict = stats_end.getInfo().get("label", {})

            if len(start_dict) == 0 or len(end_dict) == 0:
                st.warning("Zone sans données exploitables")
                st.stop()

            total = sum(start_dict.values()) + 1e-6

            # =========================
            # 🌍 CLASSES DYNAMIC WORLD
            # =========================
            dw_classes = {
                0: "Eau",
                1: "Forêt (arbres)",
                2: "Herbe",
                3: "Zone inondée",
                4: "Cultures",
                5: "Arbustes",
                6: "Urbain",
                7: "Sol nu",
                8: "Neige"
            }

            vegetation = [1, 2, 5]

            # =========================
            # 🔥 CHANGEMENTS PAR CLASSE
            # =========================
            losses = []
            gains = []

            for k in range(9):

                before = start_dict.get(str(k), 0)
                after = end_dict.get(str(k), 0)

                diff_pct = ((after - before) / total) * 100

                if abs(diff_pct) > 1:

                    label = dw_classes[k]

                    if diff_pct < 0:
                        losses.append((label, diff_pct))
                    else:
                        gains.append((label, diff_pct))

            # =========================
            # 🌿 DEGRADATION VEGETATION
            # =========================
            veg_loss = sum(
                abs(pct) for (label, pct) in losses
                if "Forêt" in label or "Herbe" in label or "Arbustes" in label
            )

            urb_gain = sum(
                pct for (label, pct) in gains
                if "Urbain" in label or "Sol nu" in label
            )

            # =========================
            # 📊 INDICATEURS
            # =========================
            deg_pct = veg_loss
            urb_pct = urb_gain

            col1, col2 = st.columns(2)
            col1.metric("Perte végétation (%)", f"{deg_pct:.2f}")
            col2.metric("Artificialisation (%)", f"{urb_pct:.2f}")

            # =========================
            # 🧠 INTERPRÉTATION MÉTIER
            # =========================
            if deg_pct < 5:
                deg_level = "faible"
            elif deg_pct < 15:
                deg_level = "modérée"
            else:
                deg_level = "forte"

            if urb_pct < 5:
                urb_level = "faible"
            elif urb_pct < 20:
                urb_level = "modérée"
            else:
                urb_level = "forte"

            # =========================
            # 🧾 TEXTE MÉTIER CLAIR
            # =========================
            st.markdown("### Analyse métier des changements")

            message = ""

            if len(losses) > 0:
                message += "\n🔴 **Perte de couverture naturelle :**\n"
                for label, pct in losses:
                    message += f"- {label} ({pct:.1f}%)\n"

            if len(gains) > 0:
                message += "\n🟢 **Gains observés :**\n"
                for label, pct in gains:
                    message += f"- {label} (+{pct:.1f}%)\n"

            # logique métier
            if any("Forêt" in l[0] or "Herbe" in l[0] for l in losses):
                message += "\n🌿 ➜ Perte de végétation naturelle détectée"

            if any("Urbain" in g[0] or "Sol nu" in g[0] for g in gains):
                message += "\n🏗️ ➜ Artificialisation du sol en cours"

            # conclusion
            if deg_pct > 15 and urb_pct > 20:
                conclusion = "⚠️ Transformation intense : pression urbaine + perte végétale"
            elif deg_pct > 15:
                conclusion = "⚠️ Dégradation environnementale importante"
            elif urb_pct > 20:
                conclusion = "🏗️ Urbanisation rapide"
            else:
                conclusion = "✅ Zone globalement stable"

            message += f"\n\n📌 **Conclusion : {conclusion}**"

            st.info(message)

            # =========================
            # 🗺️ CARTE
            # =========================
            vis = {
                "min": 0,
                "max": 8,
                "palette": [
                    "419BDF",
                    "397D49",
                    "88B053",
                    "7A87C6",
                    "E49635",
                    "DFC35A",
                    "C4281B",
                    "A59B8F",
                    "B39FE1"
                ]
            }

            m = build_dynamic_world_map(
                dw_start,
                dw_end,
                change,
                vis,
                lat,
                lon
            )

            if use_shape and gdf_client is not None:
                m.add_gdf(
                    gdf_client,
                    layer_name="Zone analysée",
                    style={
                        "color": "black",
                        "weight": 3,
                        "fillOpacity": 0
                    }
                )

                centroid = gdf_client.unary_union.centroid
                m.center = [centroid.y, centroid.x]

            else:
                m.center = [lat, lon]

            m.to_streamlit(height=600)

elif mode == "📍 Position":

        st.subheader("📍 Localisation des clients")

        uploaded_file = st.file_uploader(
            "Importer fichier clients (Excel ou CSV)",
            type=["xlsx", "csv"]
        )

        # Carte principale
        m3 = leafmap.Map(center=[7.5, -5.5], zoom=7)

        # -------------------------
        # 🌍 AJOUT DES COUCHES SIG
        # -------------------------
        m3.add_gdf(
            bassin,
            layer_name ="Bassin versant",
            style={"color": "orange"}
        )
        m3.add_gdf(hydro, layer_name="Reseau hydro", style={"color": "gray"})
        m3.add_gdf(
            forets,
            layer_name="Aires protégées",
            style={"color": "green"})
        m3.add_gdf(integrale, layer_name="Reserves integrale", style={"color": "red"})
        m3.add_gdf(parcs, layer_name="Reserves", style={"color": "red"})
        m3.add_gdf(waterways, layer_name="Cours d'eau", style={"color": "blue"})
        m3.add_gdf(lake, layer_name="Lacs", style={"color": "navy"})
        m3.add_gdf(
        kba,
        layer_name="Zones KBA",
        style={
            "color": "purple",
            "fillColor": "purple",
            "fillOpacity": 0.2
        }
    )

        if uploaded_file is not None:

            # Lecture fichier
            if uploaded_file.name.endswith(".csv"):
                df_clients = pd.read_csv(uploaded_file)
            else:
                df_clients = pd.read_excel(uploaded_file, engine="openpyxl")

            df_clients = df_clients.dropna(subset=["latitude", "longitude"])

            st.success(f"{len(df_clients)} clients chargés")

            afficher = st.checkbox("Afficher les clients", value=True)
            # convertir clients en GeoDataFrame
            gdf_clients = gpd.GeoDataFrame(
                df_clients,
                geometry=gpd.points_from_xy(df_clients.longitude, df_clients.latitude),
                crs="EPSG:4326"
            )

            # intersection avec KBA
            clients_kba = gpd.sjoin(gdf_clients, kba, how="inner", predicate="intersects")

            st.warning(f"{len(clients_kba)} clients situés dans une zone KBA ⚠️")

            if afficher:
                # ✅ Ajout des clients (cluster)
                for _, row in df_clients.iterrows():

                    color = get_color(row["secteur"])

                    m3.add_marker(
                        location=[row["latitude"], row["longitude"]],
                        popup=f"{row.get('nom','Client')}<br>Secteur: {row['secteur']}",
                        icon=folium.Icon(color=color, icon="info-sign")
                    )

            # Centrage auto
            if len(df_clients) > 0:
                m3.center = [
                    df_clients["latitude"].mean(),
                    df_clients["longitude"].mean()
                ]
                m3.zoom = 8

            # Tableau
            with st.expander("Voir les données clients"):
                st.dataframe(df_clients)

        else:
            st.info("Veuillez importer un fichier pour afficher les clients")

        # ✅ IMPORTANT : affichage final UNE SEULE FOIS
            m3.add_legend(
            title="Secteurs",
            legend_dict={
                "Risque élevé": "red",
                "Agriculture": "green",
                "Transport / Construction": "orange",
                "Services": "blue",
                "Autres": "gray"
            }
        )
        m3.to_streamlit(height=700)