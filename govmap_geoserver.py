import requests
import geopandas as gpd
import pandas as pd
import json
import time
import os
import logging
from pathlib import Path
import xml.etree.ElementTree as ET

# Settings
CHUNK_SIZE = 2000
CHUNK_DELAY = 0.1     # seconds between chunk requests
LAYER_DELAY = 1.0     # seconds between layers
FORCE_OVERWRITE = False
OUTPUT_DIR = Path("output_geopackages")
OUTPUT_DIR.mkdir(exist_ok=True)
TEMP_DIR = Path("temp_chunks")
TEMP_DIR.mkdir(exist_ok=True)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("govmap_geoserver.log", mode="a", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)

# Load layers from CSV
layers_df = pd.read_csv("wfs_layers_summary.csv")
layer_names = layers_df["Layer Name"].tolist()

def get_feature_count(layer_name):
    url = (
        f"https://www.govmap.gov.il/api/geoserver/wfs?"
        f"SERVICE=WFS&REQUEST=GetFeature&resultType=hits&typeNames={layer_name}&VERSION=2.0.0"
    )
    logger.info(f"Counting features for {layer_name}: {url}")
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        # Parse XML and extract numberMatched
        root = ET.fromstring(resp.content)
        number_matched = root.attrib.get("numberMatched")
        if number_matched is not None:
            return int(number_matched)
        else:
            logger.warning(f"numberMatched attribute not found in response for {layer_name}")
            return 0
    except Exception as e:
        logger.error(f"Failed to get count for {layer_name}: {e}")
        return 0

def fetch_features(layer_name, start_index, count=CHUNK_SIZE):
    url = (
        f"https://www.govmap.gov.il/api/geoserver/wfs?"
        f"SERVICE=WFS&REQUEST=GetFeature&count={count}&resultType=results"
        f"&typeNames={layer_name}&startIndex={start_index}"
        f"&sortBy=objectid&VERSION=2.0.0&outputFormat=application/json"
    )
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.Timeout:
        logger.warning(f"Timeout fetching chunk {start_index}-{start_index+count} for {layer_name}: {url}")
        raise
    except Exception as e:
        logger.error(f"Failed to fetch chunk {start_index}-{start_index+count} for {layer_name}: {e} | URL: {url}")
        return None

def fetch_features_with_retry(layer_name, start_index, count=CHUNK_SIZE, min_chunk=1):
    """
    Try to fetch features, and if a timeout occurs, retry with smaller chunks (10% of original size).
    Returns a list of feature dicts (may be empty).
    """
    try:
        data = fetch_features(layer_name, start_index, count)
        if data and "features" in data and data["features"]:
            return data["features"]
        else:
            return []
    except requests.exceptions.Timeout:
        pass  # Will handle below
    except Exception as e:
        logger.error(f"Error fetching chunk {start_index}-{start_index+count} for {layer_name}: {e}")
        return []

    # If we get here, there was a timeout or error
    if count <= min_chunk:
        logger.error(f"Minimum chunk size reached for {layer_name} at {start_index}. Skipping.")
        return []

    small_count = max(int(count * 0.1), min_chunk)
    features = []
    logger.warning(f"Timeout. Retrying {layer_name} chunk {start_index}-{start_index+count} in smaller pieces of {small_count}...")
    for sub_start in range(start_index, start_index + count, small_count):
        sub_count = min(small_count, start_index + count - sub_start)
        try:
            sub_features = fetch_features_with_retry(layer_name, sub_start, sub_count, min_chunk)
            features.extend(sub_features)
        except Exception as e:
            url = (
                f"https://www.govmap.gov.il/api/geoserver/wfs?"
                f"SERVICE=WFS&REQUEST=GetFeature&count={sub_count}&resultType=results"
                f"&typeNames={layer_name}&startIndex={sub_start}"
                f"&sortBy=objectid&VERSION=2.0.0&outputFormat=application/json"
            )
            logger.error(f"Missing chunk {sub_start}-{sub_start+sub_count} for {layer_name}. URL: {url} | Error: {e}")
    return features

def check_existing_geopackage(layer_name, expected_count):
    gpkg_path = OUTPUT_DIR / f"{layer_name.replace(':', '_')}.gpkg"
    if not gpkg_path.exists():
        return False
    try:
        gdf = gpd.read_file(gpkg_path, layer=layer_name.split(":")[-1])
        actual_count = len(gdf)
        if actual_count == expected_count:
            logger.info(f"Existing GeoPackage for {layer_name} has {actual_count} features (matches expected). Skipping.")
            return True
        else:
            logger.info(f"Existing GeoPackage for {layer_name} has {actual_count} features (expected {expected_count}). Will re-download.")
            return False
    except Exception as e:
        logger.warning(f"Could not read existing GeoPackage for {layer_name}: {e}")
        return False

def process_layer(layer_name):
    logger.info(f"--- Starting download for {layer_name} ---")
    count = get_feature_count(layer_name)
    logger.info(f"{layer_name}: {count} features to download.")

    if count == 0:
        logger.warning(f"No features found for {layer_name}. Skipping.")
        return

    if not FORCE_OVERWRITE and check_existing_geopackage(layer_name, count):
        return

    tmp_files = []
    total_fetched = 0
    missing_chunks = []

    for start in range(0, count, CHUNK_SIZE):
        try:
            features = fetch_features_with_retry(layer_name, start, CHUNK_SIZE)
        except Exception as e:
            url = (
                f"https://www.govmap.gov.il/api/geoserver/wfs?"
                f"SERVICE=WFS&REQUEST=GetFeature&count={CHUNK_SIZE}&resultType=results"
                f"&typeNames={layer_name}&startIndex={start}"
                f"&sortBy=objectid&VERSION=2.0.0&outputFormat=application/json"
            )
            logger.error(f"No data returned for layer {layer_name} chunk starting at {start}: {e} | URL: {url}")
            missing_chunks.append((start, url))
            continue

        if not features:
            url = (
                f"https://www.govmap.gov.il/api/geoserver/wfs?"
                f"SERVICE=WFS&REQUEST=GetFeature&count={CHUNK_SIZE}&resultType=results"
                f"&typeNames={layer_name}&startIndex={start}"
                f"&sortBy=objectid&VERSION=2.0.0&outputFormat=application/json"
            )
            logger.warning(f"No features found in chunk starting at {start}. Skipping... URL: {url}")
            missing_chunks.append((start, url))
            continue
        else:
            logger.info(f"Fetched {len(features)} features for chunk starting at {start}")

        # Write to TEMP_DIR, under layer name
        tmp_file = TEMP_DIR / f"{layer_name.replace(':', '_')}_chunk_{start}.geojson"
        with open(tmp_file, "w", encoding="utf-8") as f:
            geojson = {
                "type": "FeatureCollection",
                "features": features
            }
            f.write(json.dumps(geojson, indent=2, ensure_ascii=False))
        tmp_files.append(tmp_file)
        total_fetched += len(features)
        logger.info(f"Fetched chunk {start}–{min(start+CHUNK_SIZE, count)} ({len(features)} features)")

        time.sleep(CHUNK_DELAY)

    # Merge all chunks
    gdfs = []
    for path in tmp_files:
        try:
            gdf = gpd.read_file(path)
            gdfs.append(gdf)
        except Exception as e:
            logger.error(f"Error reading {path}: {e}")
        finally:
            os.remove(path)

    if not gdfs:
        logger.warning(f"No valid GeoDataFrames collected for {layer_name}.")
        return

    combined = pd.concat(gdfs, ignore_index=True)
    gdf_combined = gpd.GeoDataFrame(combined, geometry="geometry", crs="EPSG:3857")
    layer_out = OUTPUT_DIR / f"{layer_name.replace(':', '_')}.gpkg"
    gdf_combined.to_file(layer_out, layer=layer_name.split(":")[-1], driver="GPKG")
    logger.info(f"Saved GeoPackage: {layer_out}")
    logger.info(f"{layer_name}: Downloaded {len(gdf_combined)} features (expected {count})")

    if missing_chunks:
        logger.warning(f"Missing chunks for {layer_name}:")
        for start, url in missing_chunks:
            logger.warning(f"  Chunk starting at {start}: {url}")

if __name__ == "__main__":
    for lyr in layer_names:
        try:
            process_layer(lyr)
        except Exception as e:
            logger.error(f"Unexpected error with {lyr}: {e}")
        time.sleep(LAYER_DELAY)