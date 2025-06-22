import os
import zipfile
import requests
import pandas as pd
from tqdm import tqdm
import logging
from pathlib import Path
import shutil
import glob

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
BASE_URL = "https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2024/resstock_amy2018_release_2"
STATE = "TX"
METADATA_PATH = "data/raw/metadata/TX_baseline_metadata_and_annual_results.parquet"
BUILDING_TIMESERIES_DIR = "data/raw/building_timeseries"
WEATHER_DIR = "data/raw/weather"
SCHEDULES_DIR = "data/raw/schedules"
TEMP_DIR = "temp_downloads"
BUILDING_SAMPLE_SIZE = 10  # Number of buildings to process. Set to 0 to process all.

def get_building_ids_from_timeseries(timeseries_dir: str) -> pd.Series:
    """Scan timeseries files to get a list of unique building IDs."""
    files = glob.glob(os.path.join(timeseries_dir, '*.parquet'))
    if not files:
        logging.error(f"No building timeseries files found in {timeseries_dir}")
        return pd.Series([], dtype='int64')

    logging.info(f"Reading {len(files)} timeseries files to get unique building IDs...")
    all_bldg_ids = set()
    for file in tqdm(files, desc="Scanning building IDs"):
        try:
            # Extract building ID from filename (e.g., "926-0.parquet" -> 926)
            filename = os.path.basename(file)
            bldg_id = int(filename.split('-')[0])
            all_bldg_ids.add(bldg_id)
        except Exception as e:
            logging.error(f"Failed to extract building ID from {file}. Error: {e}")
    
    return pd.Series(list(all_bldg_ids), dtype='int64')

def download_weather_data(metadata_df: pd.DataFrame, building_ids: pd.Series):
    """Download weather files for counties corresponding to the given building IDs."""
    weather_dir = Path(WEATHER_DIR)
    weather_dir.mkdir(exist_ok=True)
    
    # Filter metadata for the buildings we are actually using
    # The building ID is stored as the index in the metadata DataFrame
    relevant_metadata = metadata_df[metadata_df.index.isin(building_ids)]
    county_ids = relevant_metadata['in.county'].unique()
    
    logging.info(f"Found {len(county_ids)} unique counties for {len(building_ids)} buildings. Downloading weather data...")
    
    for county_id in tqdm(county_ids, desc="Downloading weather files"):
        weather_file_name = f"{county_id}_2018.csv"
        weather_url = f"{BASE_URL}/weather/state%3D{STATE}/{weather_file_name}"
        save_path = weather_dir / weather_file_name
        
        if save_path.exists():
            continue
            
        try:
            response = requests.get(weather_url, stream=True)
            response.raise_for_status()
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to download {weather_url}. Error: {e}")

def download_and_extract_schedules(building_ids: pd.Series):
    """Download and extract occupancy schedules for a list of building IDs."""
    schedules_dir = Path(SCHEDULES_DIR)
    schedules_dir.mkdir(exist_ok=True)
    temp_dir = Path(TEMP_DIR)
    
    logging.info(f"Processing schedules for {len(building_ids)} buildings...")
    
    for bldg_id in tqdm(building_ids, desc="Downloading and extracting schedules"):
        bldg_id_formatted = f"bldg{int(bldg_id):07d}"
        zip_file_name = f"{bldg_id_formatted}-up00.zip"
        schedule_url = f"{BASE_URL}/model_and_schedule_files/building_energy_models/upgrade%3D0/{zip_file_name}"
        
        final_csv_path = schedules_dir / f"{bldg_id_formatted}_schedules.csv"
        if final_csv_path.exists():
            continue

        try:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            temp_dir.mkdir(exist_ok=True)

            zip_save_path = temp_dir / zip_file_name
            
            response = requests.get(schedule_url, stream=True)
            response.raise_for_status()
            with open(zip_save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            with zipfile.ZipFile(zip_save_path, 'r') as zip_ref:
                # Assuming the schedule file is always named 'in.schedules.csv'
                zip_ref.extract('in.schedules.csv', temp_dir)
                extracted_schedule_path = temp_dir / "in.schedules.csv"
                shutil.move(extracted_schedule_path, final_csv_path)

        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to download schedule for {bldg_id_formatted}. Error: {e}")
        except (zipfile.BadZipFile, KeyError):
            logging.error(f"Failed to extract 'in.schedules.csv' from {zip_file_name}.")
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

def main():
    """Main function to download additional weather and schedule data."""
    try:
        metadata_df = pd.read_parquet(METADATA_PATH)
    except Exception as e:
        logging.error(f"Failed to load metadata from {METADATA_PATH}. Error: {e}")
        return
        
    # Get the list of building IDs from the timeseries data we have
    building_ids_in_data = get_building_ids_from_timeseries(BUILDING_TIMESERIES_DIR)
    if building_ids_in_data.empty:
        logging.warning("No building IDs found in the timeseries data. Exiting.")
        return

    # --- Download Weather Data ---
    download_weather_data(metadata_df, building_ids_in_data)
    
    # --- Download Occupancy Schedules ---
    if BUILDING_SAMPLE_SIZE > 0 and BUILDING_SAMPLE_SIZE < len(building_ids_in_data):
        building_ids_to_process = building_ids_in_data.head(BUILDING_SAMPLE_SIZE)
        logging.info(f"Processing a sample of {BUILDING_SAMPLE_SIZE} buildings for schedules.")
    else:
        building_ids_to_process = building_ids_in_data
        logging.info(f"Processing all {len(building_ids_in_data)} buildings for schedules.")

    download_and_extract_schedules(building_ids_to_process)
    
    logging.info("Additional data download process completed.")

if __name__ == "__main__":
    main() 