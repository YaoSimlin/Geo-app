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
import tempfile
import json
from rasterio.mask import mask
from shapely.geometry import shape



from google.oauth2 import service_account

# =========================
# INITIALISATION EARTH ENGINE
# =========================
def init_ee():

    try:

        credentials = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=["https://www.googleapis.com/auth/earthengine"]
        )

        ee.Initialize(
            credentials,
            project=st.secrets["gcp_service_account"]["project_id"]
        )

        return True

    except Exception as e:
        st.error(f"Erreur Earth Engine : {e}")
        return False

    except Exception as e:
        st.error(f"Erreur Earth Engine : {e}")
        return False
    
init_ee()


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
# def run_change_detection(lat, lon):

#     t0_url, t1_url = get_best_sentinel_pair(lat, lon)

#     if t0_url is None:
#         return None, None, None, None

#     t0_path = save_raster(t0_url)
#     t1_path = save_raster(t1_url)

#     #detector = ChangeDetection(sam_model_type="vit_h")

#     detector.set_hyperparameters(
#         change_confidence_threshold=145,
#         use_normalized_feature=True,
#         bitemporal_match=True,
#     )

#     detector.set_mask_generator_params(
#         points_per_side=16,  # ⚡ adapté Sentinel
#         stability_score_thresh=0.90,
#     )

#     results = detector.detect_changes(
#         t0_path,
#         t1_path,
#         output_path="mask.tif",
#         export_probability=True,
#         probability_output_path="proba.tif",
#         return_detailed_results=True,
#     )

#     # NDVI
#     ndvi_t0 = compute_ndvi_raster(t0_path)
#     ndvi_t1 = compute_ndvi_raster(t1_path)

#     ndvi_change = ndvi_t1 - ndvi_t0

#     return results, "mask.tif", "proba.tif", ndvi_change


def dynamic_world_change(geometry, start_date="2020-01-01", end_date="2026-03-31"):
    import ee
    import geemap

    try:
        ee.Number(1).getInfo()
    except:
        init_ee()

    # try:
    #     ee.Initialize(project='ancient-lattice-491308-n6')
    # except:
    #     ee.Authenticate()
    #     ee.Initialize(project='ancient-lattice-491308-n6')

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

    import ee
    import leafmap.foliumap as leafmap
    import geopandas as gpd

    try:
        ee.Number(1).getInfo()
    except:
        init_ee()

    m = leafmap.Map(center=[lat, lon], zoom=13)

    # =========================
    # 🌍 Occupation du sol
    # =========================
    m.add_ee_layer(
        ee_object=dw_start,
        vis_params=vis,
        name="Occupation du sol (avant)"
    )

    m.add_ee_layer(
        ee_object=dw_end,
        vis_params=vis,
        name="Occupation du sol (après)"
    )

    # =========================
    # 🔄 Changements
    # =========================
    change_vis = {
        "min": -1,
        "max": 2,
        "palette": [
            "red",      # dégradation
            "white",    # stable
            "green"     # amélioration
        ]
    }

    m.add_ee_layer(
        ee_object=change,
        vis_params=change_vis,
        name="Changements"
    )

    # =========================
    # 📍 Géométrie analysée
    # =========================
    if geometry is not None:

        try:

            gdf = gpd.GeoDataFrame(
                geometry=[geometry],
                crs="EPSG:4326"
            )

            m.add_gdf(
                gdf,
                layer_name="Zone analysée",
                style={
                    "color": "black",
                    "weight": 3,
                    "fillColor": "black",
                    "fillOpacity": 0
                }
            )

        except Exception as e:
            print("Erreur ajout géométrie :", e)

    # =========================
    # 🌍 Contrôle couches
    # =========================
    m.add_layer_control()

        # =========================
    # 🗂️ LÉGENDE DYNAMIC WORLD
    # =========================
    legend_dict = {
        "Eau": "419BDF",
        "Forêt": "397D49",
        "Herbe": "88B053",
        "Zone inondée": "7A87C6",
        "Cultures": "E49635",
        "Arbustes": "DFC35A",
        "Urbain": "C4281B",
        "Sol nu": "A59B8F",
        "Neige": "B39FE1"
    }

    m.add_legend(
        title="Occupation du sol",
        legend_dict=legend_dict
    )

    # =========================
    # 🔄 LÉGENDE CHANGEMENTS
    # =========================
    change_legend = {
        "Dégradation": "red",
        "Stable": "white",
        "Amélioration": "green"
    }

    m.add_legend(
        title="Changements environnementaux",
        legend_dict=change_legend
    )

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

    #try:
    #ee.Initialize(project='ancient-lattice-491308-n6')
    #except:
        #ee.Authenticate()
        #ee.Initialize(project='ancient-lattice-491308-n6')

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

        draw.text((20, 20), f"An: {year}", fill="white", font=font)

        frames.append(np.array(img_pil))
    # =========================
    # 🎬 GIF FINAL
    # =========================
    gif_path = os.path.join(tempfile.gettempdir(), "dynamic_world.gif")

    imageio.mimsave(gif_path, frames, fps=2, loop=0)

    return gif_path


#--------------------------------------------------------------------------------------------
# Integration des modifcations

# =========================
# 🦎 ESPÈCES MENACÉES / ENDÉMIQUES PAR KBA
# =========================
def normalize_name(s):
    import unicodedata, re
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace("'", "").replace("’", "")
    s = re.sub(r"[-–—]", " ", s)          # tous les tirets → espace
    s = re.sub(r"\s+", " ", s)             # espaces multiples → un seul
    return s.strip().lower()



def get_kba_species_key(kba_name):
    """
    Convertit le NatName de la couche KBA vers la clé canonique
    (même espace de normalisation que la colonne Zone de l'Excel).
    """
    key = normalize_name(kba_name)

    aliases = {
        "parc national de tai": "tai nzo",
        "reserve de faune du nzo": "tai nzo",
        "parc national de tai et reserve de faune du nzo": "tai nzo",
        "adiopodoume": "adiopodoume",
        "foret classee de bossematie": "bossematie",
        "foret classee de yapo et mambo": "yapo mambo",
        "parc national du banco": "banco",
        "foret marecageuse de la tanoe": "tanoe ehy",
        "parc national dazagny": "azagny",
        "foret classee de moprie": "moprie",
        "station de recherche ecologique de lamto": "lamto",
        "foret classee de cavally et goin debe": "cavally goin debe",
        "parc national du mont peko": "mont peko",
        "parc national de la marahoue": "marahoue",
        "foret classee des monts gueoule et mont glo reserves": "gueoule glo",
        "reserve integrale du mont nimba": "mont nimba",
        "parc national de la comoe": "comoe",
        "parc national du mont sangbe": "sangbe",
    }

    # On re-normalise la valeur de l'alias pour garantir
    # qu'elle est dans le même espace que _zone_key
    return normalize_name(aliases.get(key, key))


@st.cache_data(show_spinner=False)
def load_kba_species(excel_path):
    """
    Charge les feuilles espèces et référence KBA.
    """

    try:
        especes = pd.read_excel(
            excel_path,
            sheet_name="Especes_menacees",
            engine="openpyxl"
        )

        kba_ref = pd.read_excel(
            excel_path,
            sheet_name="KBA_17",
            engine="openpyxl"
        )

    except Exception as e:
        st.error(
            f"Erreur chargement fichier espèces : {e}"
        )
        return None, None

    # Suppression des lignes d'en-tête répétées
    if "Code KBA" in especes.columns:
        especes = especes[
            especes["Code KBA"] != "Code KBA"
        ].copy()

    if "Code" in kba_ref.columns:
        kba_ref = kba_ref[
            kba_ref["Code"] != "Code"
        ].copy()

    especes = especes.reset_index(drop=True)
    kba_ref = kba_ref.reset_index(drop=True)

    return especes, kba_ref



def enrich_with_kba_species(clients_df, kba_name_col, species_path):

    COLONNES_ESPECES = [
        "Code KBA",
        "Espèce / taxon",
        "Groupe",
        "Statut UICN / statut actuel",
        "Endémisme / restriction",
    ]

    # --- Sécurité 1 : entrée vide ---
    if clients_df is None or clients_df.empty:
        for c in COLONNES_ESPECES:
            clients_df[c] = ""
        return clients_df

    try:
        especes_df, _ = load_kba_species(species_path)

        # --- Sécurité 2 : fichier Excel illisible ---
        if especes_df is None:
            st.error(f"❌ Fichier espèces introuvable ou illisible : {species_path}")
            for c in COLONNES_ESPECES:
                clients_df[c] = ""
            return clients_df

        especes_df = especes_df.copy()
        clients_df = clients_df.copy()
        especes_df.columns = especes_df.columns.astype(str).str.strip()
        clients_df.columns = clients_df.columns.astype(str).str.strip()

        # --- Sécurité 3 : colonnes Excel attendues ---
        colonnes_manquantes = [c for c in ["Zone"] + COLONNES_ESPECES
                               if c not in especes_df.columns]
        if colonnes_manquantes:
            st.error(f"❌ Colonnes manquantes dans l'Excel : {colonnes_manquantes}")
            st.write("Colonnes disponibles :", list(especes_df.columns))
            for c in COLONNES_ESPECES:
                clients_df[c] = ""
            return clients_df

        # --- Clés normalisées ---
        especes_df["_zone_key"] = especes_df["Zone"].fillna("").astype(str).apply(normalize_name)
        clients_df["_kba_key"] = clients_df[kba_name_col].fillna("").astype(str).apply(get_kba_species_key)

        # --- Matching exact + fallback flou ---
        import difflib
        zone_keys = especes_df["_zone_key"].unique().tolist()

        def trouver_especes(kba_key):
            sp = especes_df[especes_df["_zone_key"] == kba_key]
            if not sp.empty:
                return sp, "exact"
            proches = difflib.get_close_matches(kba_key, zone_keys, n=1, cutoff=0.75)
            if proches:
                return especes_df[especes_df["_zone_key"] == proches[0]], f"flou (~{proches[0]})"
            return sp, None

        # --- Diagnostic du matching (visible dans l'app) ---
        # avec_match = clients_df["_kba_key"].apply(lambda k: trouver_especes(k)[1] is not None).sum()
        # st.write(f"🦎 Matching espèces : {avec_match}/{len(clients_df)} KBA appariées")

        lignes = []
        for _, client in clients_df.iterrows():
            base = client.drop(labels=["_kba_key"], errors="ignore").to_dict()
            sp, _ = trouver_especes(client["_kba_key"])

            if sp.empty:
                for c in COLONNES_ESPECES:
                    base[c] = "Aucune espèce référencée" if c == "Espèce / taxon" else ""
                lignes.append(base)
            else:
                for _, espece in sp.iterrows():
                    ligne = dict(base)
                    for c in COLONNES_ESPECES:
                        ligne[c] = espece[c]
                    lignes.append(ligne)

        result = pd.DataFrame(lignes)

        # --- Sécurité 4 : garantir la présence des colonnes ---
        for c in COLONNES_ESPECES:
            if c not in result.columns:
                result[c] = ""

        return result

    except Exception as e:
        st.exception(e)   # <-- affiche la vraie erreur au lieu d'échouer silencieusement
        for c in COLONNES_ESPECES:
            clients_df[c] = ""
        return clients_df

    # =====================================================
    # NORMALISATION DES ZONES EXCEL
    # =====================================================

    especes_df["_zone_key"] = (
        especes_df["Zone"]
        .fillna("")
        .astype(str)
        .apply(normalize_name)
    )

    # =====================================================
    # NORMALISATION DES KBA DE LA COUCHE SIG
    # =====================================================

    clients_df["_kba_key"] = (
        clients_df[kba_name_col]
        .fillna("")
        .astype(str)
        .apply(get_kba_species_key)
    )

    # =====================================================
    # CONSTRUCTION DES LIGNES
    # =====================================================

    lignes_resultat = []

    for _, client in clients_df.iterrows():

        kba_key = client["_kba_key"]

        if not kba_key:
            continue

        # -------------------------------------------------
        # Toutes les espèces de cette KBA
        # -------------------------------------------------

        sp = especes_df[
            especes_df["_zone_key"] == kba_key
        ].copy()

        if sp.empty:

            # On conserve quand même le client
            ligne = client.drop(
                labels=["_kba_key"],
                errors="ignore"
            ).to_dict()

            ligne["Espèce / taxon"] = "Aucune espèce référencée"
            ligne["Groupe"] = ""
            ligne["Statut UICN / statut actuel"] = ""
            ligne["Endémisme / restriction"] = ""

            lignes_resultat.append(ligne)

            continue

        # -------------------------------------------------
        # UNE LIGNE PAR ESPECE
        # -------------------------------------------------

        for _, espece in sp.iterrows():

            ligne = client.drop(
                labels=["_kba_key"],
                errors="ignore"
            ).to_dict()

            ligne["Code KBA"] = espece["Code KBA"]
            ligne["Espèce / taxon"] = espece["Espèce / taxon"]
            ligne["Groupe"] = espece["Groupe"]
            ligne["Statut UICN / statut actuel"] = (
                espece["Statut UICN / statut actuel"]
            )
            ligne["Endémisme / restriction"] = (
                espece["Endémisme / restriction"]
            )

            lignes_resultat.append(ligne)

    # =====================================================
    # RESULTAT FINAL
    # =====================================================

    if not lignes_resultat:
        return clients_df.drop(
            columns=["_kba_key"],
            errors="ignore"
        )

    result = pd.DataFrame(lignes_resultat)

    return result


# =========================
# 🔎 EXTRACTIONS SUR ZONES SENSIBLES
# =========================
def extract_clients_inside_zones(gdf_clients, zones_gdf, name_field=None):
    """
    Retourne les clients situés À L'INTÉRIEUR des zones sensibles.
    """
    cols = [name_field] if name_field and name_field in zones_gdf.columns else []
    zones = zones_gdf[cols + ["geometry"]].copy()

    result = gpd.sjoin(
        gdf_clients, zones, how="inner", predicate="intersects"
    )

    # 🔥 Un client dans plusieurs zones = une seule ligne (on garde la 1ère zone)
    result = result.drop_duplicates(subset=["latitude", "longitude"])

    return result


def extract_clients_near_zones(
    gdf_clients,
    zones_gdf,
    name_field=None,
    distance_m=2000):
    cols = [name_field] if name_field and name_field in zones_gdf.columns else []

    # Zones dans un CRS métrique
    zones_m = zones_gdf[cols + ["geometry"]].copy().to_crs(epsg=32630)

    # Buffer uniquement pour sélectionner les clients à moins de 2 km
    zones_buf = zones_m.copy()
    zones_buf["geometry"] = zones_buf.geometry.buffer(distance_m)

    # Clients en mètres
    clients_m = gdf_clients.to_crs(epsg=32630)

    # Extraction
    result = gpd.sjoin(
        clients_m,
        zones_buf,
        how="inner",
        predicate="intersects"
    )

    # On récupère la vraie géométrie de la zone
    zone_geometries = zones_m[["geometry"]].copy()

    # index_right correspond à l'index de la zone originale
    result["_zone_geometry"] = result["index_right"].map(
        zone_geometries["geometry"]
    )

    # Distance réelle en mètres
    result["distance_m"] = result.geometry.distance(
        result["_zone_geometry"]
    )

    # Conversion en kilomètres
    result["distance_km"] = result["distance_m"] / 1000

    # Nettoyage
    result = result.drop(columns=["_zone_geometry"], errors="ignore")

    # Retour en WGS84
    result = result.to_crs(epsg=4326)

    # Arrondi
    result["distance_m"] = result["distance_m"].round(1)
    result["distance_km"] = result["distance_km"].round(3)

    return result


def show_kba_species(clients_in_zones, kba_name_col, species_path):
    """
    Affiche automatiquement les espèces menacées/endémiques
    des KBA contenant des clients.
    """
    especes_df, _ = load_kba_species(species_path)

    if especes_df is None:
        st.error("Impossible de charger la base d'espèces.")
        return

    st.markdown("#### 🦎 Espèces menacées / endémiques des KBA concernés")

    especes_df["_key"] = especes_df["Zone"].apply(normalize_name)

    zones_concernees = clients_in_zones[kba_name_col].dropna().unique()

    for zone in zones_concernees:
        key = normalize_name(zone)
        sp = especes_df[especes_df["_key"] == key].drop(columns="_key")

        if len(sp) == 0:
            with st.expander(f"🌍 {zone} — aucune espèce référencée"):
                st.info("Cette zone KBA n'a pas d'espèce listée dans la base.")
            continue

        nb_critique = sp["Statut UICN / statut actuel"].str.contains(
            "En danger critique", na=False).sum()
        nb_endemique = sp["Endémisme / restriction"].str.contains(
            "Endémique", na=False).sum()

        resume = (
            f"🌍 {zone} — {len(sp)} espèces | "
            f"🚨 {nb_critique} en danger critique | "
            f"⭐ {nb_endemique} endémique(s)"
        )

        with st.expander(resume):
            st.dataframe(sp)
            csv_sp = sp.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                f"⬇️ Télécharger espèces - {zone}",
                csv_sp,
                f"especes_{key}.csv",
                "text/csv",
                key=f"dl_{key}"
            )


def clean_coord(series):
                    """Nettoie les coordonnées : espaces insécables, virgules décimales, etc."""
                    return pd.to_numeric(
                        series.astype(str)
                            .str.replace("\xa0", "", regex=False)   # espace insécable
                            .str.replace("\u202f", "", regex=False)  # espace fine insécable
                            .str.replace(" ", "", regex=False)
                            .str.replace(",", ".", regex=False)      # virgule → point
                            .str.strip(),
                        errors="coerce"
                    )

def read_shp_safe(path):
    """Lit un shapefile en gérant l'encodage des attributs."""
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return gpd.read_file(path, encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    # dernier recours : lecture sans lever d'erreur
    return gpd.read_file(path, encoding="latin-1", ignore_errors=True)


def sanitize_strings(gdf):
    """Supprime les caractères problématiques des colonnes texte."""
    gdf = gdf.copy()
    for col in gdf.columns:
        if col != "geometry" and gdf[col].dtype == object:
            gdf[col] = (
                gdf[col]
                .astype(str)
                .str.encode("utf-8", errors="replace")
                .str.decode("utf-8")
                .str.replace("\xa0", " ", regex=False)
            )
    return gdf
