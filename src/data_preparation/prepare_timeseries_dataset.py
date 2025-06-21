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
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
import glob
from tqdm import tqdm

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
        
    def _load_config(self) -> Dict:
        """Load feature configuration."""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return json.load(f)
        else:
            logging.warning(f"Config file not found: {self.config_path}")
            return {}
    
    def load_and_combine_timeseries_data(self, timeseries_files: List[str]) -> pd.DataFrame:
        """Load and combine multiple time series data files."""
        logging.info(f"Loading and combining {len(timeseries_files)} time series files...")
        all_dfs = [pd.read_parquet(file).reset_index() for file in tqdm(timeseries_files, desc="Loading time series data")]
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
        logging.info(f"Combined time series data shape: {combined_df.shape}")
        return combined_df
    
    def load_metadata(self, metadata_path: str) -> pd.DataFrame:
        """Load metadata."""
        logging.info(f"Loading metadata from {metadata_path}")
        metadata_df = pd.read_parquet(metadata_path).reset_index()
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
    
    def merge_data(self, timeseries_df: pd.DataFrame, 
                                     metadata_df: pd.DataFrame) -> pd.DataFrame:
        """Merge metadata with time series data."""
        logging.info("Merging metadata with time series data")
        merged_df = pd.merge(timeseries_df, metadata_df, on='bldg_id', how='left')
        logging.info(f"Merged data shape: {merged_df.shape}")
        return merged_df
    
    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select features based on configuration."""
        logging.info("Selecting features based on configuration")
        if not self.config:
            logging.warning("No config found, using all columns.")
            return df
        
        features = ['bldg_id', 'timestamp'] + self.config.get('static_features', []) + self.config.get('dynamic_features', []) + self.config.get('primary_target', []) + self.config.get('secondary_targets', [])
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
            df_copy = df_copy.groupby('bldg_id', group_keys=False).apply(lambda group: group.ffill().bfill())
            missing_after = df_copy.isnull().sum().sum()
            if missing_after > 0:
                logging.warning(f"Filling {missing_after} remaining NaNs with global median.")
                df_copy.fillna(df_copy.median(numeric_only=True), inplace=True)
        logging.info(f"Handled {missing_before} missing values.")
        return df_copy
    
    def encode_and_scale(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical and scale numerical features."""
        logging.info(f"Encoding and scaling features (fit={fit})")
        df_copy = df.copy()
        
        # Explicitly define categorical columns from config
        categorical_cols = [col for col in self.config.get('static_features', []) if col in df_copy.columns and df_copy[col].dtype == 'object']
        
        for col in categorical_cols:
            if fit:
                le = LabelEncoder()
                df_copy[col] = le.fit_transform(df_copy[col].astype(str))
                self.label_encoders[col] = le
            elif col in self.label_encoders:
                df_copy[col] = self.label_encoders[col].transform(df_copy[col].astype(str))
        
        # Identify numerical columns AFTER encoding
        numerical_cols = [col for col in df_copy.columns if pd.api.types.is_numeric_dtype(df_copy[col]) and col not in ['bldg_id', 'timestamp']]
        
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
        feature_cols = [col for col in df.columns if col not in target_cols + ['bldg_id', 'timestamp']]

        for building_id in tqdm(building_ids, desc="Creating windows"):
            building_df = df[df['bldg_id'] == building_id].sort_values(by='timestamp')
            if len(building_df) < input_window + output_window: continue

            X_data = building_df[feature_cols].values
            y_data = building_df[target_cols].values

            for i in range(0, len(building_df) - input_window - output_window + 1, step):
                all_X.append(X_data[i:i + input_window])
                all_y.append(y_data[i + input_window:i + input_window + output_window])

        X_windows, y_windows = np.array(all_X), np.array(all_y)
        
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
            pickle.dump(self.label_encoders, f)
        logging.info("Saved all artifacts.")

def main():
    """Main function to run the time series data preprocessing pipeline."""
    
    # Configuration
    raw_data_dir = 'data/raw'
    output_dir = 'data/processed'
    metadata_path = os.path.join(raw_data_dir, 'TX_baseline_metadata_and_annual_results.parquet')
    
    # Get all downloaded time series files
    timeseries_files = glob.glob(os.path.join(raw_data_dir, '*-0.parquet'))
    if not timeseries_files:
        logging.error("No time series files found.")
        return
        
    # Preprocessing
    preprocessor = TimeSeriesDataPreprocessor()
    
    # Load and merge
    ts_df = preprocessor.load_and_combine_timeseries_data(timeseries_files)
    meta_df = preprocessor.load_metadata(metadata_path)
    merged_df = preprocessor.merge_data(ts_df, meta_df)
    
    # Process
    engineered_df = preprocessor.engineer_time_features(merged_df)
    selected_df = preprocessor.select_features(engineered_df)
    filled_df = preprocessor.handle_missing_values(selected_df)
    processed_df = preprocessor.encode_and_scale(filled_df)
    
    # Split buildings into train/val/test sets
    building_ids = processed_df['bldg_id'].unique()
    train_ids, test_ids = train_test_split(building_ids, test_size=0.2, random_state=42)
    train_ids, val_ids = train_test_split(train_ids, test_size=0.15, random_state=42)
    
    logging.info(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)} buildings")
    
    # Create windows for each set
    window_params = {'input_window': 168, 'output_window': 24, 'step': 24}
    X_train, y_train = preprocessor.create_windows_for_buildings(processed_df, train_ids, **window_params)
    X_val, y_val = preprocessor.create_windows_for_buildings(processed_df, val_ids, **window_params)
    X_test, y_test = preprocessor.create_windows_for_buildings(processed_df, test_ids, **window_params)
    
    data_splits = {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test
    }
    
    # Save processed data
    preprocessor.save_data(data_splits, output_dir)
    logging.info("Time series data preprocessing complete!")


if __name__ == '__main__':
    main()
