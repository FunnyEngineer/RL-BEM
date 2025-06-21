"""
Download ResStock Time Series Data

This script downloads individual building time series data from the ResStock dataset.
It reads building IDs from the metadata file and downloads time series for multiple buildings.
"""

import pandas as pd
import requests
import os
import logging
import time
from typing import List, Optional
from urllib.parse import quote
import concurrent.futures
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ResStockTimeSeriesDownloader:
    """Downloader for ResStock time series data."""
    
    def __init__(self, metadata_path: str, output_dir: str = 'data/raw'):
        """
        Initialize the downloader.
        
        Args:
            metadata_path: Path to metadata file containing building IDs
            output_dir: Directory to save downloaded files
        """
        self.metadata_path = metadata_path
        self.output_dir = output_dir
        self.base_url = "https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2024/resstock_amy2018_release_2/timeseries_individual_buildings/by_state/upgrade%3D0/state%3DTX"
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def load_building_ids(self, n_buildings: int = 100) -> List[int]:
        """Load building IDs from metadata file."""
        logging.info(f"Loading {n_buildings} building IDs from metadata...")
        
        try:
            # Load metadata
            metadata_df = pd.read_parquet(self.metadata_path)
            logging.info(f"Loaded metadata with {len(metadata_df)} buildings")
            
            # Get building IDs (assuming index contains building IDs)
            building_ids = list(metadata_df.index)[:n_buildings]
            
            logging.info(f"Selected {len(building_ids)} building IDs: {building_ids[:5]}...")
            return building_ids
            
        except Exception as e:
            logging.error(f"Error loading metadata: {e}")
            # Fallback: return a range of building IDs
            logging.info("Using fallback building IDs")
            return list(range(1, n_buildings + 1))
    
    def get_download_url(self, building_id: int) -> str:
        """Generate download URL for a building ID."""
        # URL encode the building ID
        encoded_id = quote(f"{building_id}-0")
        return f"{self.base_url}/{encoded_id}.parquet"
    
    def download_building_timeseries(self, building_id: int) -> Optional[str]:
        """Download time series data for a single building."""
        url = self.get_download_url(building_id)
        output_path = os.path.join(self.output_dir, f"{building_id}-0.parquet")
        
        # Skip if file already exists
        if os.path.exists(output_path):
            logging.debug(f"File already exists: {output_path}")
            return output_path
        
        try:
            logging.debug(f"Downloading building {building_id} from {url}")
            
            # Download with timeout and retry logic
            for attempt in range(3):
                try:
                    response = requests.get(url, timeout=30)
                    response.raise_for_status()
                    
                    # Save file
                    with open(output_path, 'wb') as f:
                        f.write(response.content)
                    
                    logging.debug(f"Successfully downloaded building {building_id}")
                    return output_path
                    
                except requests.exceptions.RequestException as e:
                    if attempt < 2:  # Retry up to 2 times
                        logging.warning(f"Attempt {attempt + 1} failed for building {building_id}: {e}")
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    else:
                        logging.error(f"Failed to download building {building_id} after 3 attempts: {e}")
                        return None
                        
        except Exception as e:
            logging.error(f"Error downloading building {building_id}: {e}")
            return None
    
    def download_multiple_buildings(self, building_ids: List[int], max_workers: int = 5) -> List[str]:
        """Download time series data for multiple buildings using parallel processing."""
        logging.info(f"Starting download of {len(building_ids)} buildings with {max_workers} workers")
        
        successful_downloads = []
        
        # Use ThreadPoolExecutor for parallel downloads
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all download tasks
            future_to_building = {
                executor.submit(self.download_building_timeseries, building_id): building_id 
                for building_id in building_ids
            }
            
            # Process completed downloads with progress bar
            with tqdm(total=len(building_ids), desc="Downloading buildings") as pbar:
                for future in concurrent.futures.as_completed(future_to_building):
                    building_id = future_to_building[future]
                    try:
                        result = future.result()
                        if result:
                            successful_downloads.append(result)
                            pbar.set_postfix({"Success": len(successful_downloads)})
                        else:
                            pbar.set_postfix({"Failed": building_id})
                    except Exception as e:
                        logging.error(f"Exception for building {building_id}: {e}")
                        pbar.set_postfix({"Error": building_id})
                    
                    pbar.update(1)
        
        logging.info(f"Download completed: {len(successful_downloads)} successful out of {len(building_ids)}")
        return successful_downloads
    
    def verify_downloads(self, building_ids: List[int]) -> dict:
        """Verify downloaded files and their sizes."""
        logging.info("Verifying downloaded files...")
        
        verification_results = {
            'total_buildings': len(building_ids),
            'downloaded': 0,
            'missing': 0,
            'file_sizes': {},
            'missing_files': []
        }
        
        for building_id in building_ids:
            file_path = os.path.join(self.output_dir, f"{building_id}-0.parquet")
            
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                verification_results['downloaded'] += 1
                verification_results['file_sizes'][building_id] = file_size
                
                # Log file size in MB
                size_mb = file_size / (1024 * 1024)
                logging.debug(f"Building {building_id}: {size_mb:.2f} MB")
            else:
                verification_results['missing'] += 1
                verification_results['missing_files'].append(building_id)
        
        # Print summary
        logging.info(f"Verification Summary:")
        logging.info(f"  Total buildings: {verification_results['total_buildings']}")
        logging.info(f"  Downloaded: {verification_results['downloaded']}")
        logging.info(f"  Missing: {verification_results['missing']}")
        
        if verification_results['file_sizes']:
            avg_size = sum(verification_results['file_sizes'].values()) / len(verification_results['file_sizes'])
            avg_size_mb = avg_size / (1024 * 1024)
            logging.info(f"  Average file size: {avg_size_mb:.2f} MB")
        
        if verification_results['missing_files']:
            logging.warning(f"Missing files: {verification_results['missing_files'][:10]}...")
        
        return verification_results
    
    def create_download_summary(self, building_ids: List[int], downloaded_files: List[str]) -> None:
        """Create a summary file of downloaded buildings."""
        summary_path = os.path.join(self.output_dir, 'download_summary.txt')
        
        with open(summary_path, 'w') as f:
            f.write("ResStock Time Series Download Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Download Date: {pd.Timestamp.now()}\n")
            f.write(f"Total Buildings Requested: {len(building_ids)}\n")
            f.write(f"Successfully Downloaded: {len(downloaded_files)}\n")
            f.write(f"Base URL: {self.base_url}\n\n")
            
            f.write("Building IDs:\n")
            for building_id in building_ids:
                status = "✓" if f"{building_id}-0.parquet" in [os.path.basename(f) for f in downloaded_files] else "✗"
                f.write(f"  {status} {building_id}\n")
            
            f.write(f"\nDownloaded Files:\n")
            for file_path in downloaded_files:
                f.write(f"  {os.path.basename(file_path)}\n")
        
        logging.info(f"Download summary saved to {summary_path}")

def main():
    """Main function to download ResStock time series data."""
    
    # Configuration
    metadata_path = 'data/raw/TX_baseline_metadata_and_annual_results.parquet'
    output_dir = 'data/raw'
    n_buildings = 100
    max_workers = 5  # Number of parallel downloads
    
    # Check if metadata file exists
    if not os.path.exists(metadata_path):
        logging.error(f"Metadata file not found: {metadata_path}")
        return
    
    # Initialize downloader
    downloader = ResStockTimeSeriesDownloader(metadata_path, output_dir)
    
    # Load building IDs
    building_ids = downloader.load_building_ids(n_buildings)
    
    if not building_ids:
        logging.error("No building IDs found")
        return
    
    # Download time series data
    logging.info(f"Starting download of {len(building_ids)} buildings...")
    downloaded_files = downloader.download_multiple_buildings(building_ids, max_workers)
    
    # Verify downloads
    verification_results = downloader.verify_downloads(building_ids)
    
    # Create summary
    downloader.create_download_summary(building_ids, downloaded_files)
    
    logging.info("Download process completed!")
    
    # Print final statistics
    success_rate = (verification_results['downloaded'] / verification_results['total_buildings']) * 100
    logging.info(f"Success rate: {success_rate:.1f}%")

if __name__ == '__main__':
    main() 