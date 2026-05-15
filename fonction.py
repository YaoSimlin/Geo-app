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
import streamlit as st
from pyproj import Transformer
import rasterio.features
import imageio
import requests
#from geoai.change_detection import ChangeDetection
import tempfile
from rasterio.mask import mask
from shapely.geometry import shape


def get_base64_image(image_path):
        with open(image_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()



@st.cache_data(show_spinner=False)
def compute_indices(_geometry):


    try:
        catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

        bounds = _geometry.bounds  # xmin, ymin, xmax, ymax

        from datetime import datetime, timedelta

        end_date = datetime.today() - timedelta(days=10)  # 🔥 buffer ingestion
        start_date = end_date - timedelta(days=60)

        datetime_range = f"{start_date:%Y-%m-%d}/{end_date:%Y-%m-%d}"

        search = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=bounds,
            datetime=datetime_range,
            query={"eo:cloud_cover": {"lt": 20}},
        )

        items = list(search.items())

        if len(items) == 0:
            return None, None

        # 🔥 Filtrer uniquement les images qui intersectent vraiment la géométrie
        valid_items = []

        geom_gdf = gpd.GeoSeries([_geometry], crs="EPSG:4326")

        for item in items:
            footprint = shape(item.geometry)
            footprint_gdf = gpd.GeoSeries([footprint], crs="EPSG:4326")

            if geom_gdf.intersects(footprint_gdf).any():
                valid_items.append(item)

        if len(valid_items) == 0:
            return None, None

        # 🔥 Prendre la meilleure image (moins de nuages)
        item = planetary_computer.sign(
            sorted(valid_items, key=lambda x: x.properties["eo:cloud_cover"])[0]
        )

        def read_band(asset):
            with rasterio.open(asset.href) as src:

                # 🔥 reprojection geometry → CRS raster
                geom_proj = gpd.GeoSeries([_geometry], crs="EPSG:4326").to_crs(src.crs)
                geom_json = [geom_proj.iloc[0].__geo_interface__]

                try:
                    out_image, _ = mask(src, geom_json, crop=True)
                    return out_image[0].astype(float)
                except ValueError:
                    return None  # pas d'intersection réelle

        red = read_band(item.assets["B04"])
        nir = read_band(item.assets["B08"])
        green = read_band(item.assets["B03"])

        # 🔥 sécurité
        if red is None or nir is None or green is None:
            return None, None

        # 🔥 masque pixels valides
        mask_valid = (red > 0) & (nir > 0) & (green > 0)

        if not np.any(mask_valid):
            return None, None

        # 🔥 indices
        ndvi = np.where(mask_valid, (nir - red) / (nir + red + 1e-10), np.nan)
        ndwi = np.where(mask_valid, (green - nir) / (green + nir + 1e-10), np.nan)

        return float(np.nanmean(ndvi)), float(np.nanmean(ndwi))

    except Exception as e:
        st.error(f"Erreur satellite : {e}")
        return None, None
    

    # -------------------------
    # Correction CRS
    # -------------------------
def fix_crs(gdf):
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=32630, allow_override=True)
        else:
            sample = gdf.geometry.iloc[0].centroid
            x, y = sample.x, sample.y
            if abs(x) > 180 or abs(y) > 90:
                gdf = gdf.set_crs(epsg=32630, allow_override=True)
        return gdf.to_crs(epsg=4326)

def compute_nearest(gdf, geom_m, name_field):
        gdf = gdf.to_crs(epsg=32630).copy()
        
        gdf["dist"] = gdf.distance(geom_m)
        idx = gdf["dist"].idxmin()
        
        distance_km = gdf.loc[idx, "dist"] / 1000
        try:
            name = str(gdf.loc[idx, name_field])
        except:
            name = "Non renseigné"
        
        return name, distance_km

@st.cache_data(show_spinner=False)
def load_clients(excel_path):
        df = pd.read_excel(excel_path)

        # Nettoyage basique
        df = df.dropna(subset=["latitude", "longitude"])

        return df

def get_client_geometry(lat, lon):
    gdf_client = st.session_state.get("gdf_client", None)

    if gdf_client is not None:
        geom = gdf_client.unary_union
        geom_m = gpd.GeoSeries([geom], crs=4326).to_crs(32630).iloc[0]
        return geom, geom_m, True
    else:
        point = Point(lon, lat)
        geom_m = gpd.GeoSeries([point], crs=4326).to_crs(32630).iloc[0]
        return point, geom_m, False
    
def load_uploaded_geometry(uploaded_file):
    try:
        tmpdir = tempfile.mkdtemp()
        file_path = os.path.join(tmpdir, uploaded_file.name)

        # Sauvegarde du fichier
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        # -------------------------
        # CAS 1 : SHAPEFILE (.zip)
        # -------------------------
        if uploaded_file.name.endswith(".zip"):

            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(tmpdir)

            shp_files = [f for f in os.listdir(tmpdir) if f.endswith(".shp")]

            if not shp_files:
                st.error("Aucun fichier .shp trouvé")
                return None

            shp_path = os.path.join(tmpdir, shp_files[0])
            gdf = gpd.read_file(shp_path)

        # -------------------------
        # CAS 2 : GEOJSON
        # -------------------------
        elif uploaded_file.name.endswith((".geojson", ".json")):

            gdf = gpd.read_file(file_path)

        else:
            st.error("Format non supporté")
            return None

        # 🔥 nettoyage
        gdf = gdf[gdf.geometry.notnull()]
        gdf = gdf[~gdf.geometry.is_empty]

        if len(gdf) == 0:
            st.error("Fichier vide ou invalide")
            return None

        # 🔄 correction CRS
        gdf = fix_crs(gdf)

        return gdf

    except Exception as e:
        st.error(f"Erreur chargement fichier : {e}")
        return None

    except Exception as e:
        st.error(f"Erreur shapefile : {e}")
        return None

@st.cache_data
def create_hls_timeseries(lat, lon):
        import geemap

        point = ee.Geometry.Point([lon, lat])

        collection = ee.ImageCollection("NASA/HLS/HLSL30/v002") \
            .filterBounds(point) \
            .filterDate("2020-01-01", "2026-03-25") \
            .filter(ee.Filter.lt("CLOUD_COVERAGE", 20))

        # Calcul NDVI
        def add_ndvi(image):
            ndvi = image.normalizedDifference(['B5', 'B4']).rename('NDVI')
            return image.addBands(ndvi)

        collection = collection.map(add_ndvi)

        # Réduction temporelle (médiane mensuelle)
        timeseries = geemap.create_timeseries(
            collection.select('NDVI'),
            start_date='2020-01-01',
            end_date='2026-03-25',
            frequency='quarter',
            reducer='mean',
            region=point.buffer(5000)  # zone autour du client
        )

        return timeseries

def get_color(secteur):

    if secteur in ["Oil, Gas & Consumable Fuels", "Metals & Mining"]:
        return "red"
    
    elif secteur in ["Agriculture (Plant products)", "Food & Beverage Production"]:
        return "green"
    
    elif secteur in ["Transportation Services", "Construction Materials Production"]:
        return "orange"
    
    elif secteur in ["Offices & professional services", "Telecommunication services"]:
        return "blue"
    
    else:
        return "gray"

def get_name_field(gdf):
        possible_fields = ["NOM", "NAME", "nom", "name", "LIBELLE", "DESIGNATION"]
        
        for field in possible_fields:
            if field in gdf.columns:
                return field
        
        return None

def get_dynamic_world(year):
    collection = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1") \
        .filterDate(f"{year}-01-01", f"{year}-12-31")

    if collection.size().getInfo() == 0:
        return None

    return collection.median().select("label")

def get_stress_color(label):
    if label == "Low":
        return "#2ECC71"  # vert
    elif label == "Low - Medium":
        return "#F1C40F"
    elif label == "Medium - High":
        return "#E67E22"
    elif label == "High":
        return "#E74C3C"
    elif label == "Extremely High":
        return "#8E0000"
    else:
        return "gray"
    

# =========================
# 📡 Récupération Sentinel optimisée
# =========================
def get_best_sentinel_pair(lat, lon):

    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

    bbox = [lon - 0.02, lat - 0.02, lon + 0.02, lat + 0.02]

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime="2023-01-01/2026-01-01",
        query={"eo:cloud_cover": {"lt": 20}},
    )

    items = list(search.items())

    if len(items) < 2:
        return None, None

    # 🔥 tri intelligent
    def same_season(item1, item2):
        return abs(item1.datetime.month - item2.datetime.month) <= 1

    items_sorted = sorted(items, key=lambda x: x.datetime)

    t0 = items_sorted[0]

    # 🔥 chercher image récente même saison
    t1 = None
    for item in reversed(items_sorted):
        if same_season(t0, item):
            t1 = item
            break

    if t1 is None:
        t1 = items_sorted[-1]

    return t0.assets["visual"].href, t1.assets["visual"].href


# =========================
# 💾 Sauvegarde raster
# =========================
def save_raster(url):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
    r = requests.get(url)
    with open(tmp.name, "wb") as f:
        f.write(r.content)
    return tmp.name


# =========================
# 🌱 NDVI raster
# =========================
def compute_ndvi_raster(path):

    with rasterio.open(path) as src:
        red = src.read(3).astype(float)
        nir = src.read(4).astype(float)

        ndvi = (nir - red) / (nir + red + 1e-6)

    return ndvi


# =========================
# 🧠 Change Detection
# =========================
def run_change_detection(lat, lon):

    t0_url, t1_url = get_best_sentinel_pair(lat, lon)

    if t0_url is None:
        return None, None, None, None

    t0_path = save_raster(t0_url)
    t1_path = save_raster(t1_url)

    detector = ChangeDetection(sam_model_type="vit_h")

    detector.set_hyperparameters(
        change_confidence_threshold=145,
        use_normalized_feature=True,
        bitemporal_match=True,
    )

    detector.set_mask_generator_params(
        points_per_side=16,  # ⚡ adapté Sentinel
        stability_score_thresh=0.90,
    )

    results = detector.detect_changes(
        t0_path,
        t1_path,
        output_path="mask.tif",
        export_probability=True,
        probability_output_path="proba.tif",
        return_detailed_results=True,
    )

    # NDVI
    ndvi_t0 = compute_ndvi_raster(t0_path)
    ndvi_t1 = compute_ndvi_raster(t1_path)

    ndvi_change = ndvi_t1 - ndvi_t0

    return results, "mask.tif", "proba.tif", ndvi_change


def dynamic_world_change(geometry, start_date="2020-01-01", end_date="2026-03-31"):
    import ee
    import geemap

    try:
        ee.Initialize(project='ancient-lattice-491308-n6')
    except:
        ee.Authenticate()
        ee.Initialize(project='ancient-lattice-491308-n6')

    # =========================
    # 🔥 CONVERSION EN EE GEOMETRY  
    # =========================
    try:
        if geometry.geom_type == "Point":
            # 👉 cas coordonnées
            ee_geom = ee.Geometry.Point([geometry.x, geometry.y]).buffer(2000)
        else:
            # 👉 cas shapefile (Polygon / MultiPolygon)
            ee_geom = ee.Geometry(geometry.__geo_interface__)
            # ✅ AJOUT ICI
        ee_geom = ee_geom.simplify(100)
        ee_geom = ee_geom.buffer(100)
    except Exception as e:
        print("Erreur conversion géométrie :", e)
        return None, None, None, None

    # =========================
    # 📡 COLLECTION DYNAMIC WORLD
    # =========================
    dw = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1") \
        .filterBounds(ee_geom)

    dw_start_col = dw.filterDate(start_date, "2020-12-31")
    dw_end_col = dw.filterDate("2026-02-28", end_date)

    # =========================
    # ✅ VÉRIFICATION DATA
    # =========================
    if dw_start_col.size().getInfo() == 0:
        return None, None, None, None

    if dw_end_col.size().getInfo() == 0:
        return None, None, None, None

    # =========================
    # 🖼️ IMAGES
    # =========================
    dw_start = dw_start_col.select("label").mode().clip(ee_geom)
    dw_end = dw_end_col.select("label").mode().clip(ee_geom)

    # =========================
    # 🔒 SÉCURITÉ BANDES
    # =========================
    band_start = dw_start.bandNames().size().getInfo()
    band_end = dw_end.bandNames().size().getInfo()

    if band_start == 0 or band_end == 0:
        return None, None, None, None

    # =========================
    # 🔄 CHANGEMENT
    # =========================
    # Détection brute
    change_mask = dw_start.neq(dw_end)

    # Classes
    veg = [1, 2, 5]
    urban = [6, 7]

    start_veg = dw_start.remap(veg, [1]*len(veg), 0)
    end_veg   = dw_end.remap(veg, [1]*len(veg), 0)

    start_urban = dw_start.remap(urban, [1]*len(urban), 0)
    end_urban   = dw_end.remap(urban, [1]*len(urban), 0)

    # Typologie
    deg = start_veg.And(end_urban)       # 🔴 dégradation
    imp = start_urban.And(end_veg)       # 🟢 amélioration
    other = change_mask.And(deg.Not()).And(imp.Not())  # 🟡 autres

    # Carte finale
    change = deg.multiply(-1) \
        .add(imp) \
        .add(other.multiply(2))

    # =========================
    # 📊 STATISTIQUES
    # =========================
    stats = change.reduceRegion(
    reducer=ee.Reducer.frequencyHistogram(),
    geometry=ee_geom,
    scale=10,
    maxPixels=1e9,
    bestEffort=True
        )

    stats_start = dw_start.reduceRegion(
    reducer=ee.Reducer.frequencyHistogram(),
    geometry=ee_geom,
    scale=10,
    maxPixels=1e9,
    bestEffort=True
    )

    stats_end = dw_end.reduceRegion(
    reducer=ee.Reducer.frequencyHistogram(),
    geometry=ee_geom,
    scale=10,
    maxPixels=1e9,
    bestEffort=True
    )

    return dw_start, dw_end, change, stats, stats_start, stats_end


def build_dynamic_world_map(dw_start, dw_end, change, vis, lat, lon, geometry=None):
    try:
        import ee

        # 🔥 FIX CRITIQUE (juste ça manquait)
        try:
            ee.Initialize(project='ancient-lattice-491308-n6')
        except Exception:
            ee.Authenticate()
            ee.Initialize(project='ancient-lattice-491308-n6')

        import geemap.foliumap as geemap

    except Exception as e:
        import streamlit as st
        st.error(f"Erreur geemap import : {e}")
        st.stop()

    m = geemap.Map(center=[lat, lon], zoom=13)

    m.addLayer(dw_start, vis, "Occupation du sol (avant)")
    m.addLayer(dw_end, vis, "Occupation du sol (après)")
    m.addLayer(change, {"min": -5, "max": 5, "palette": ["red", "white", "green"]}, "Changement")

    # -------------------------
    # 🧾 Légende Dynamic World
    # -------------------------
    dw_legend = {
        "Eau": "419BDF",
        "Arbres": "397D49",
        "Herbe": "88B053",
        "Zone inondée": "7A87C6",
        "Cultures": "E49635",
        "Arbustes": "DFC35A",
        "Urbain": "C4281B",
        "Sol nu": "A59B8F",
        "Neige / glace": "B39FE1"
    }

    m.add_legend(
        title="Occupation du sol (Dynamic World)",
        legend_dict=dw_legend
    )

    # -------------------------
    # 🔄 Légende Changement
    # -------------------------
    change_legend = {
            "Dégradation": "red",
            "Stable": "white",
            "Amélioration": "green",
            "Autres changements": "yellow"
        }

    m.add_legend(
        title="Changement",
        legend_dict=change_legend
    )

    # -------------------------
    # 🧭 AFFICHAGE SHAPE
    # -------------------------
    if geometry is not None:
        try:
            import geopandas as gpd

            gdf = gpd.GeoDataFrame(geometry=[geometry], crs="EPSG:4326")

            m.add_gdf(
                gdf,
                layer_name="Zone analysée",
                style={
                    "color": "black",
                    "weight": 3,
                    "fillOpacity": 0
                }
            )
        except:
            pass

    return m


def get_dynamic_world_series(geometry, start_year=2020, end_year=2026):
    import ee

    # 🔥 sécurité CRITIQUE
    if geometry is None:
        return None, None

    try:
        geom_type = geometry.geom_type
    except:
        return None, None

    # 🔥 conversion geometry
    if geom_type == "Point":
        ee_geom = ee.Geometry.Point([geometry.x, geometry.y]).buffer(2000)
    else:
        ee_geom = ee.Geometry(geometry.__geo_interface__)

    ee_geom = ee_geom.simplify(100).buffer(100)

    years = list(range(start_year, end_year + 1))
    images = []

    for year in years:
        img = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1") \
            .filterBounds(ee_geom) \
            .filterDate(f"{year}-01-01", f"{year}-12-31") \
            .select("label") \
            .mode() \
            .clip(ee_geom)

        images.append((year, img))

    return images, ee_geom


def dynamic_world_timelapse(geometry, start_year=2020, end_year=2026):
    import ee
    import geemap
    from PIL import Image, ImageDraw, ImageFont

    if geometry is None:
        return None

    try:
        ee.Initialize(project='ancient-lattice-491308-n6')
    except:
        #ee.Authenticate()
        ee.Initialize(project='ancient-lattice-491308-n6')

    # =========================
    # 🔥 GEOMETRY
    # =========================
    if geometry.geom_type == "Point":
        ee_geom = ee.Geometry.Point([geometry.x, geometry.y]).buffer(2000)
    else:
        ee_geom = ee.Geometry(geometry.__geo_interface__)

    ee_geom = ee_geom.simplify(100).buffer(100)

    # =========================
    # 📡 COLLECTION
    # =========================
    dw = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1") \
        .filterBounds(ee_geom)

    years = list(range(start_year, end_year + 1))
    frames = []

    palette = [
        "419BDF","397D49","88B053","7A87C6",
        "E49635","DFC35A","C4281B","A59B8F","B39FE1"
    ]

    from io import BytesIO

    for year in years:

        img = dw.filterDate(f"{year}-01-01", f"{year}-12-31") \
                .select("label") \
                .mode() \
                .clip(ee_geom)

        url = img.getThumbURL({
            "region": ee_geom,
            "dimensions": 1024,
            "min": 0,
            "max": 8,
            "palette": palette
        })

        response = requests.get(url)

        img_pil = Image.open(BytesIO(response.content)).convert("RGB")

        draw = ImageDraw.Draw(img_pil)

        try:
            font = ImageFont.truetype("arial.ttf", 30)
        except:
            font = ImageFont.load_default()

        draw.text((20, 20), f"Année : {year}", fill="white", font=font)

        frames.append(np.array(img_pil))
    # =========================
    # 🎬 GIF FINAL
    # =========================
    gif_path = os.path.join(tempfile.gettempdir(), "dynamic_world.gif")

    imageio.mimsave(gif_path, frames, fps=2, loop=0)

    return gif_path


