"""
Feature Selection for ResStock Time Series ML Surrogate Model

This script analyzes the ResStock datasets to select relevant features and targets
for training a time series surrogate model for EnergyPlus simulations.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ResStockFeatureSelector:
    """Feature selector for ResStock time series data."""
    
    def __init__(self, timeseries_path: str, metadata_path: str):
        """
        Initialize the feature selector.
        
        Args:
            timeseries_path: Path to the time series data (1-0.parquet)
            metadata_path: Path to the metadata (TX_baseline_metadata_and_annual_results.parquet)
        """
        self.timeseries_path = timeseries_path
        self.metadata_path = metadata_path
        self.timeseries_df = None
        self.metadata_df = None
        
    def load_data(self) -> None:
        """Load both datasets."""
        logging.info("Loading time series data...")
        self.timeseries_df = pd.read_parquet(self.timeseries_path)
        logging.info(f"Time series data loaded: {self.timeseries_df.shape}")
        
        logging.info("Loading metadata...")
        self.metadata_df = pd.read_parquet(self.metadata_path)
        logging.info(f"Metadata loaded: {self.metadata_df.shape}")
        
    def analyze_timeseries_features(self) -> Dict[str, List[str]]:
        """Analyze and categorize time series features."""
        logging.info("Analyzing time series features...")
        
        features = {
            'target_variables': [],
            'weather_features': [],
            'energy_features': [],
            'temperature_features': [],
            'emissions_features': [],
            'time_features': []
        }
        
        # Target variables (primary outputs we want to predict)
        target_candidates = [
            'out.zone_mean_air_temp.conditioned_space.c',  # Indoor temperature
            'out.site_energy.total.energy_consumption',    # Total energy
            'out.electricity.total.energy_consumption',    # Electricity
            'out.load.cooling.energy_delivered.kbtu',      # Cooling load
            'out.load.heating.energy_delivered.kbtu'       # Heating load
        ]
        
        for target in target_candidates:
            if target in self.timeseries_df.columns:
                features['target_variables'].append(target)
        
        # Weather features (external conditions)
        weather_candidates = [
            'out.outdoor_air_dryblub_temp.c'  # Outdoor temperature
        ]
        
        for weather in weather_candidates:
            if weather in self.timeseries_df.columns:
                features['weather_features'].append(weather)
        
        # Energy consumption features (can be used as features or targets)
        energy_cols = [col for col in self.timeseries_df.columns 
                      if 'energy_consumption' in col and 'intensity' not in col]
        features['energy_features'] = energy_cols
        
        # Temperature features (zone temperatures)
        temp_cols = [col for col in self.timeseries_df.columns 
                    if 'zone_mean_air_temp' in col]
        features['temperature_features'] = temp_cols
        
        # Emissions features
        emissions_cols = [col for col in self.timeseries_df.columns 
                         if 'co2e_kg' in col]
        features['emissions_features'] = emissions_cols
        
        # Time features (derived from timestamp)
        features['time_features'] = ['timestamp']
        
        logging.info(f"Found {len(features['target_variables'])} target variables")
        logging.info(f"Found {len(features['weather_features'])} weather features")
        logging.info(f"Found {len(features['energy_features'])} energy features")
        
        return features
    
    def analyze_metadata_features(self) -> Dict[str, List[str]]:
        """Analyze and categorize metadata features."""
        logging.info("Analyzing metadata features...")
        
        features = {
            'building_characteristics': [],
            'hvac_system': [],
            'envelope_features': [],
            'location_features': [],
            'occupancy_features': [],
            'appliance_features': []
        }
        
        # Building characteristics
        building_cols = [
            'in.geometry_building_type_acs',
            'in.geometry_floor_area',
            'in.geometry_stories',
            'in.geometry_foundation_type',
            'in.vintage',
            'in.bedrooms',
            'in.occupants'
        ]
        
        for col in building_cols:
            if col in self.metadata_df.columns:
                features['building_characteristics'].append(col)
        
        # HVAC system features
        hvac_cols = [
            'in.hvac_heating_type',
            'in.hvac_cooling_type',
            'in.hvac_heating_efficiency',
            'in.hvac_cooling_efficiency',
            'in.heating_fuel',
            'in.heating_setpoint',
            'in.cooling_setpoint'
        ]
        
        for col in hvac_cols:
            if col in self.metadata_df.columns:
                features['hvac_system'].append(col)
        
        # Building envelope features
        envelope_cols = [
            'in.insulation_wall',
            'in.insulation_ceiling',
            'in.insulation_floor',
            'in.windows',
            'in.infiltration'
        ]
        
        for col in envelope_cols:
            if col in self.metadata_df.columns:
                features['envelope_features'].append(col)
        
        # Location features
        location_cols = [
            'in.ashrae_iecc_climate_zone_2004',
            'in.state',
            'in.city',
            'in.county'
        ]
        
        for col in location_cols:
            if col in self.metadata_df.columns:
                features['location_features'].append(col)
        
        # Occupancy features
        occupancy_cols = [
            'in.tenure',
            'in.income',
            'in.usage_level'
        ]
        
        for col in occupancy_cols:
            if col in self.metadata_df.columns:
                features['occupancy_features'].append(col)
        
        # Appliance features
        appliance_cols = [
            'in.clothes_dryer',
            'in.clothes_washer',
            'in.dishwasher',
            'in.refrigerator',
            'in.cooking_range'
        ]
        
        for col in appliance_cols:
            if col in self.metadata_df.columns:
                features['appliance_features'].append(col)
        
        logging.info(f"Found {len(features['building_characteristics'])} building characteristics")
        logging.info(f"Found {len(features['hvac_system'])} HVAC features")
        logging.info(f"Found {len(features['envelope_features'])} envelope features")
        
        return features
    
    def get_recommended_features(self) -> Dict[str, List[str]]:
        """Get recommended feature sets for the ML model."""
        logging.info("Generating feature recommendations...")
        
        timeseries_features = self.analyze_timeseries_features()
        metadata_features = self.analyze_metadata_features()
        
        # Recommended feature sets
        recommended_features = {
            # Primary target (indoor temperature)
            'primary_target': ['out.zone_mean_air_temp.conditioned_space.c'],
            
            # Secondary targets (energy consumption)
            'secondary_targets': [
                'out.site_energy.total.energy_consumption',
                'out.electricity.total.energy_consumption'
            ],
            
            # Static features (building characteristics)
            'static_features': [
                'in.geometry_building_type_acs',
                'in.geometry_floor_area',
                'in.geometry_stories',
                'in.vintage',
                'in.hvac_heating_type',
                'in.hvac_cooling_type',
                'in.hvac_heating_efficiency',
                'in.hvac_cooling_efficiency',
                'in.insulation_wall',
                'in.insulation_ceiling',
                'in.ashrae_iecc_climate_zone_2004'
            ],
            
            # Dynamic features (weather, time)
            'dynamic_features': [
                'out.outdoor_air_dryblub_temp.c',
                'timestamp'
            ],
            
            # Additional energy features for multi-output prediction
            'energy_features': [
                'out.electricity.cooling.energy_consumption',
                'out.electricity.heating.energy_consumption',
                'out.electricity.hot_water.energy_consumption'
            ]
        }
        
        # Filter to only include features that exist in the datasets
        filtered_features = {}
        for category, features_list in recommended_features.items():
            if category in ['static_features']:
                # Check against metadata
                filtered_features[category] = [
                    f for f in features_list if f in self.metadata_df.columns
                ]
            else:
                # Check against timeseries
                filtered_features[category] = [
                    f for f in features_list if f in self.timeseries_df.columns
                ]
        
        return filtered_features
    
    def print_feature_summary(self) -> None:
        """Print a summary of available features."""
        logging.info("=== FEATURE SELECTION SUMMARY ===")
        
        recommended = self.get_recommended_features()
        
        print("\n🎯 PRIMARY TARGET:")
        for target in recommended['primary_target']:
            print(f"  • {target}")
        
        print("\n⚡ SECONDARY TARGETS:")
        for target in recommended['secondary_targets']:
            print(f"  • {target}")
        
        print("\n🏠 STATIC FEATURES (Building Characteristics):")
        for feature in recommended['static_features']:
            print(f"  • {feature}")
        
        print("\n🌤️ DYNAMIC FEATURES (Weather & Time):")
        for feature in recommended['dynamic_features']:
            print(f"  • {feature}")
        
        print("\n💡 ENERGY FEATURES (for Multi-output):")
        for feature in recommended['energy_features']:
            print(f"  • {feature}")
        
        print(f"\n📊 DATASET SIZES:")
        print(f"  • Time series: {self.timeseries_df.shape}")
        print(f"  • Metadata: {self.metadata_df.shape}")
        
        # Check for building ID overlap
        timeseries_buildings = set(self.timeseries_df.index.get_level_values('bldg_id'))
        metadata_buildings = set(self.metadata_df.index)
        overlap = len(timeseries_buildings.intersection(metadata_buildings))
        
        print(f"\n🔗 BUILDING OVERLAP:")
        print(f"  • Time series buildings: {len(timeseries_buildings)}")
        print(f"  • Metadata buildings: {len(metadata_buildings)}")
        print(f"  • Overlapping buildings: {overlap}")

def main():
    """Main function to run feature selection analysis."""
    
    # File paths
    timeseries_path = 'data/raw/1-0.parquet'
    metadata_path = 'data/raw/TX_baseline_metadata_and_annual_results.parquet'
    
    # Check if files exist
    if not os.path.exists(timeseries_path):
        logging.error(f"Time series file not found: {timeseries_path}")
        return
    
    if not os.path.exists(metadata_path):
        logging.error(f"Metadata file not found: {metadata_path}")
        return
    
    # Initialize feature selector
    selector = ResStockFeatureSelector(timeseries_path, metadata_path)
    
    # Load data
    selector.load_data()
    
    # Analyze features
    selector.print_feature_summary()
    
    # Save feature recommendations
    recommended_features = selector.get_recommended_features()
    
    # Create output directory
    os.makedirs('config', exist_ok=True)
    
    # Save feature configuration
    import json
    with open('config/feature_config.json', 'w') as f:
        json.dump(recommended_features, f, indent=2)
    
    logging.info("Feature configuration saved to config/feature_config.json")

if __name__ == '__main__':
    main() 