"""
Time Series Dataset Preparation for ResStock ML Surrogate Model

This script preprocesses the ResStock time series data for training a sequence model.
It handles data cleaning, feature engineering, sliding window creation, and train/val/test splitting for multiple buildings.
"""

import pandas as pd
import numpy as np
import logging
import os
import json
from typing import Dict, List, Tuple
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import warnings
import glob
from tqdm import tqdm
from pathlib import Path

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TimeSeriesDataPreprocessor:
    """Preprocessor for ResStock time series data."""
    
    def __init__(self, config_path: str = 'config/feature_config.json'):
        self.config_path = config_path
        self.config = self._load_config()
        self.scalers = {}
        self.label_encoders = {}
        self.one_hot_encoders = {}
        
    def _load_config(self) -> Dict:
        """Load feature configuration."""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return json.load(f)
        else:
            logging.warning(f"Config file not found: {self.config_path}")
            return {}
    
    def load_building_timeseries(self, timeseries_dir: str) -> pd.DataFrame:
        """Load and combine multiple building time series data files."""
        files = glob.glob(os.path.join(timeseries_dir, '*.parquet'))
        logging.info(f"Loading and combining {len(files)} building time series files...")
        all_dfs = [pd.read_parquet(file) for file in tqdm(files, desc="Loading building time series")]
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
        logging.info(f"Combined building time series data shape: {combined_df.shape}")
        return combined_df

    def load_weather_data(self, weather_dir: str, metadata_df: pd.DataFrame) -> pd.DataFrame:
        """Load and combine weather data for all relevant counties."""
        county_ids = metadata_df['in.county'].unique()
        logging.info(f"Loading weather data for {len(county_ids)} counties...")
        
        weather_dfs = []
        for county_id in tqdm(county_ids, desc="Loading weather data"):
            weather_file = Path(weather_dir) / f"{county_id}_2018.csv"
            if weather_file.exists():
                df = pd.read_csv(weather_file)
                df['in.county'] = county_id
                weather_dfs.append(df)
        
        if not weather_dfs:
            logging.warning("No weather data found.")
            return pd.DataFrame()
            
        combined_weather_df = pd.concat(weather_dfs, ignore_index=True)
        combined_weather_df['timestamp'] = pd.to_datetime(combined_weather_df['date_time'])
        combined_weather_df.drop(columns=['date_time'], inplace=True)
        logging.info(f"Combined weather data shape: {combined_weather_df.shape}")
        return combined_weather_df

    def load_schedule_data(self, schedule_dir: str, building_ids: pd.Series) -> pd.DataFrame:
        """Load and combine schedule data for all relevant buildings."""
        logging.info(f"Loading schedule data for {len(building_ids)} buildings...")
        
        schedule_dfs = []
        for bldg_id in tqdm(building_ids, desc="Loading schedule data"):
            bldg_id_formatted = f"bldg{int(bldg_id):07d}"
            schedule_file = Path(schedule_dir) / f"{bldg_id_formatted}_schedules.csv"
            if schedule_file.exists():
                df = pd.read_csv(schedule_file, index_col=0)
                # Schedules are hourly for a year, need to create timestamps
                start_date = '2018-01-01 01:00:00'
                end_date = '2019-01-01 00:00:00'
                df['timestamp'] = pd.to_datetime(pd.date_range(start=start_date, end=end_date, freq='h'))
                df['bldg_id'] = bldg_id
                schedule_dfs.append(df)

        if not schedule_dfs:
            logging.warning("No schedule data found.")
            return pd.DataFrame()

        combined_schedule_df = pd.concat(schedule_dfs, ignore_index=True)
        logging.info(f"Combined schedule data shape: {combined_schedule_df.shape}")
        return combined_schedule_df

    def load_metadata(self, metadata_path: str) -> pd.DataFrame:
        """Load metadata."""
        logging.info(f"Loading metadata from {metadata_path}")
        metadata_df = pd.read_parquet(metadata_path)
        logging.info(f"Loaded metadata shape: {metadata_df.shape}")
        return metadata_df

    def engineer_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features from timestamp."""
        logging.info("Engineering time features")
        df_copy = df.copy()
        ts = df_copy['timestamp']
        df_copy['hour_sin'] = np.sin(2 * np.pi * ts.dt.hour / 24)
        df_copy['hour_cos'] = np.cos(2 * np.pi * ts.dt.hour / 24)
        df_copy['day_sin'] = np.sin(2 * np.pi * ts.dt.dayofyear / 365)
        df_copy['day_cos'] = np.cos(2 * np.pi * ts.dt.dayofyear / 365)
        return df_copy
    
    def merge_data(self, building_df: pd.DataFrame, 
                   metadata_df: pd.DataFrame, 
                   weather_df: pd.DataFrame, 
                   schedule_df: pd.DataFrame) -> pd.DataFrame:
        """Merge all data sources."""
        logging.info("Merging all data sources...")
        
        # Merge building data with metadata
        merged_df = pd.merge(building_df, metadata_df, on='bldg_id', how='left')
        logging.info(f"Shape after merging with metadata: {merged_df.shape}")
        
        # Merge with weather data
        if not weather_df.empty:
            merged_df = pd.merge(merged_df, weather_df, on=['timestamp', 'in.county'], how='left')
            logging.info(f"Shape after merging with weather data: {merged_df.shape}")

        # Merge with schedule data
        if not schedule_df.empty:
            merged_df = pd.merge(merged_df, schedule_df, on=['timestamp', 'bldg_id'], how='left')
            logging.info(f"Shape after merging with schedule data: {merged_df.shape}")
            
        return merged_df
    
    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select features based on configuration."""
        logging.info("Selecting features based on configuration")
        if not self.config:
            logging.warning("No config found, using all columns.")
            return df
        
        features = (
            ['bldg_id', 'timestamp'] + 
            self.config.get('static_features', []) + 
            self.config.get('dynamic_features', []) + 
            self.config.get('weather_features', []) +
            self.config.get('schedule_features', []) +
            self.config.get('primary_target', []) + 
            self.config.get('secondary_targets', [])
        )
        time_features = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']
        features.extend(time_features)
        
        available_features = [f for f in list(dict.fromkeys(features)) if f in df.columns]
        selected_df = pd.DataFrame(df[available_features])
        logging.info(f"Selected {len(selected_df.columns)} features.")
        return selected_df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        logging.info("Handling missing values")
        df_copy = df.copy()
        missing_before = df_copy.isnull().sum().sum()
        if missing_before > 0:
            # Group by building and forward/backward fill time-series data
            df_copy = df_copy.groupby('bldg_id', group_keys=False).apply(lambda group: group.ffill().bfill())
            missing_after = df_copy.isnull().sum().sum()
            if missing_after > 0:
                logging.warning(f"Filling {missing_after} remaining NaNs with global median/mode.")
                # For remaining NaNs, use median for numeric and mode for object columns
                for col in df_copy.columns:
                    if df_copy[col].isnull().any():
                        if pd.api.types.is_numeric_dtype(df_copy[col]):
                            df_copy[col].fillna(df_copy[col].median(), inplace=True)
                        else:
                            # Get the mode, and if there are multiple, take the first one.
                            mode_val = df_copy[col].mode()
                            if not mode_val.empty:
                                df_copy[col].fillna(mode_val[0], inplace=True)
                            else:
                                # Handle cases where mode is empty (e.g., all NaNs)
                                df_copy[col].fillna("Unknown", inplace=True)

        logging.info(f"Handled {missing_before} missing values.")
        return df_copy
    
    def encode_and_scale(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical and scale numerical features."""
        logging.info(f"Encoding and scaling features (fit={fit})")
        df_copy = df.copy()
        
        # Identify categorical and numerical columns from config
        categorical_cols = [col for col in self.config.get('static_features', []) if col in df_copy.columns and df_copy[col].dtype == 'object']
        numerical_cols = [col for col in df_copy.columns if pd.api.types.is_numeric_dtype(df_copy[col]) and col not in ['bldg_id', 'timestamp']]
        
        # One-Hot Encode categorical features
        if fit:
            self.one_hot_encoders = {}
            for col in categorical_cols:
                ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                transformed = ohe.fit_transform(df_copy[[col]])
                ohe_df = pd.DataFrame(transformed, columns=ohe.get_feature_names_out([col]))
                df_copy = pd.concat([df_copy.drop(col, axis=1), ohe_df], axis=1)
                self.one_hot_encoders[col] = ohe
        else:
            for col in categorical_cols:
                if col in self.one_hot_encoders:
                    ohe = self.one_hot_encoders[col]
                    transformed = ohe.transform(df_copy[[col]])
                    ohe_df = pd.DataFrame(transformed, columns=ohe.get_feature_names_out([col]))
                    df_copy = pd.concat([df_copy.drop(col, axis=1), ohe_df], axis=1)

        # Scale numerical features
        if fit:
            scaler = MinMaxScaler()
            df_copy[numerical_cols] = scaler.fit_transform(df_copy[numerical_cols])
            self.scalers['features'] = scaler
        elif 'features' in self.scalers:
            df_copy[numerical_cols] = self.scalers['features'].transform(df_copy[numerical_cols])
            
        return df_copy

    def create_windows_for_buildings(self, df: pd.DataFrame, 
                                     building_ids: List[int],
                                     input_window: int = 168, 
                                     output_window: int = 24,
                                     step: int = 24) -> Tuple[np.ndarray, np.ndarray]:
        """Create sliding windows for a list of buildings."""
        logging.info(f"Creating sliding windows for {len(building_ids)} buildings...")
        
        all_X, all_y = [], []
        
        target_cols = self.config.get('primary_target', []) + self.config.get('secondary_targets', [])
        # Ensure all columns are numeric before creating windows
        df_numeric = df.drop(columns=['bldg_id', 'timestamp']).apply(pd.to_numeric)
        feature_cols = [col for col in df_numeric.columns if col not in target_cols]

        for building_id in tqdm(building_ids, desc="Creating windows"):
            building_df = df[df['bldg_id'] == building_id].sort_values(by='timestamp')
            if len(building_df) < input_window + output_window: continue

            building_numeric_df = building_df.drop(columns=['bldg_id', 'timestamp']).apply(pd.to_numeric)
            X_data = building_numeric_df[feature_cols].to_numpy()
            y_data = building_numeric_df[target_cols].to_numpy()

            for i in range(0, len(building_df) - input_window - output_window + 1, step):
                all_X.append(X_data[i:i + input_window])
                all_y.append(y_data[i + input_window:i + input_window + output_window])

        if not all_X:
            return np.array([]), np.array([])
            
        X_windows, y_windows = np.array(all_X, dtype=np.float32), np.array(all_y, dtype=np.float32)
        
        logging.info(f"Created {len(X_windows)} windows. X shape: {X_windows.shape}, y shape: {y_windows.shape}")
        return X_windows, y_windows
    
    def save_data(self, data_dict: Dict, output_dir: str) -> None:
        """Save processed data as numpy arrays and save scalers/encoders."""
        logging.info(f"Saving processed data to {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        for name, data in data_dict.items():
            np.save(os.path.join(output_dir, f"{name}.npy"), data)
        
        import pickle
        with open(os.path.join(output_dir, 'scalers.pkl'), 'wb') as f:
            pickle.dump(self.scalers, f)
        with open(os.path.join(output_dir, 'encoders.pkl'), 'wb') as f:
            pickle.dump({'one_hot_encoders': self.one_hot_encoders}, f)
        logging.info("Saved all artifacts.")

def main():
    """Main function to run the time series data preprocessing pipeline."""
    
    # Configuration
    raw_data_dir = 'data/raw'
    output_dir = 'data/processed'
    
    # --- Preprocessing ---
    preprocessor = TimeSeriesDataPreprocessor()
    
    # --- Load Data ---
    metadata_df = preprocessor.load_metadata(os.path.join(raw_data_dir, 'metadata', 'TX_baseline_metadata_and_annual_results.parquet'))
    building_df = preprocessor.load_building_timeseries(os.path.join(raw_data_dir, 'building_timeseries'))
    weather_df = preprocessor.load_weather_data(os.path.join(raw_data_dir, 'weather'), metadata_df)
    schedule_df = preprocessor.load_schedule_data(os.path.join(raw_data_dir, 'schedules'), metadata_df['bldg_id'])

    # --- Merge and Process ---
    merged_df = preprocessor.merge_data(building_df, metadata_df, weather_df, schedule_df)
    engineered_df = preprocessor.engineer_time_features(merged_df)
    selected_df = preprocessor.select_features(engineered_df)
    filled_df = preprocessor.handle_missing_values(selected_df)
    processed_df = preprocessor.encode_and_scale(filled_df)
    
    # --- Split and Create Windows ---
    building_ids = processed_df['bldg_id'].unique()
    train_ids, test_ids = train_test_split(building_ids, test_size=0.2, random_state=42)
    train_ids, val_ids = train_test_split(train_ids, test_size=0.15, random_state=42)
    
    logging.info(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)} buildings")
    
    window_params = {'input_window': 168, 'output_window': 24, 'step': 24}
    X_train, y_train = preprocessor.create_windows_for_buildings(processed_df, train_ids, **window_params)
    X_val, y_val = preprocessor.create_windows_for_buildings(processed_df, val_ids, **window_params)
    X_test, y_test = preprocessor.create_windows_for_buildings(processed_df, test_ids, **window_params)
    
    # --- Save Data ---
    data_splits = {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test
    }
    preprocessor.save_data(data_splits, output_dir)
    logging.info("Time series data preprocessing complete!")

if __name__ == '__main__':
    main()
