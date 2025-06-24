#!/usr/bin/env python3
"""
Script to find unique counties based on building IDs in timeseries data.
Extracts building IDs from timeseries filenames and matches them with county data in metadata.
"""

import os
import pandas as pd
import glob
from pathlib import Path

def extract_building_ids_from_timeseries():
    """Extract building IDs from timeseries filenames."""
    timeseries_dir = Path("data/raw/building_timeseries")
    parquet_files = glob.glob(str(timeseries_dir / "*.parquet"))
    
    building_ids = []
    for file_path in parquet_files:
        # Extract building ID from filename (e.g., "4-0.parquet" -> 4)
        filename = os.path.basename(file_path)
        building_id = int(filename.split('-')[0])
            building_ids.append(building_id)
    
    return sorted(list(set(building_ids)))  # Remove duplicates and sort

def get_unique_counties(building_ids):
    """Get unique counties for the given building IDs from metadata."""
    metadata_file = "data/raw/metadata/TX_baseline_metadata_and_annual_results.parquet"
    
    # Load metadata
    print(f"Loading metadata from {metadata_file}...")
    metadata_df = pd.read_parquet(metadata_file)
    
    # Filter metadata to only include buildings that have timeseries data
    print(f"Filtering metadata for {len(building_ids)} buildings with timeseries data...")
    filtered_metadata = metadata_df.loc[building_ids]
    
    # Get unique counties
    unique_counties = filtered_metadata['in.county'].unique()
    unique_county_names = filtered_metadata['in.county_name'].unique()
    
    return unique_counties, unique_county_names, filtered_metadata

def main():
    print("=== Finding Unique Counties from Building Timeseries Data ===\n")
    
    # Step 1: Extract building IDs from timeseries filenames
    print("Step 1: Extracting building IDs from timeseries filenames...")
    building_ids = extract_building_ids_from_timeseries()
    print(f"Found {len(building_ids)} unique building IDs with timeseries data")
    print(f"Building IDs: {building_ids[:10]}{'...' if len(building_ids) > 10 else ''}")
    print()
    
    # Step 2: Get unique counties for these buildings
    print("Step 2: Finding unique counties for these buildings...")
    unique_counties, unique_county_names, filtered_metadata = get_unique_counties(building_ids)
    
    # Step 3: Display results
    print("\n=== RESULTS ===")
    print(f"Total buildings with timeseries data: {len(building_ids)}")
    print(f"Unique counties (codes): {len(unique_counties)}")
    print(f"Unique county names: {len(unique_county_names)}")
    print()
    
    print("Unique County Codes:")
    for county in sorted(unique_counties):
        print(f"  {county}")
    print()
    
    print("Unique County Names:")
    for county_name in sorted(unique_county_names):
        print(f"  {county_name}")
    print()
    
    # Step 4: Show building distribution by county
    print("Building Distribution by County:")
    county_counts = filtered_metadata['in.county_name'].value_counts().sort_values(ascending=False)
    for county_name, count in county_counts.items():
        print(f"  {county_name}: {count} buildings")
    
    # Step 5: Save results to file
    output_file = "data/unique_counties_from_timeseries.csv"
    print(f"\nSaving results to {output_file}...")
    
    # Create a summary DataFrame
    summary_data = []
    for county_code, county_name in zip(unique_counties, unique_county_names):
        buildings_in_county = filtered_metadata[filtered_metadata['in.county'] == county_code]
        summary_data.append({
            'county_code': county_code,
            'county_name': county_name,
            'building_count': len(buildings_in_county)
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('building_count', ascending=False)
    summary_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    return building_ids, unique_counties, unique_county_names

if __name__ == "__main__":
    building_ids, unique_counties, unique_county_names = main()
