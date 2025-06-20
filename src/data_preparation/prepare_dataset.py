import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import logging
import os
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(filepath):
    """Loads data from a specified file path."""
    logging.info(f"Loading data from {filepath}...")
    # Assuming the data is in a CSV file. Modify as needed for other formats like Parquet.
    df = pd.read_csv(filepath)
    logging.info("Data loaded successfully.")
    return df

def clean_data(df):
    """Cleans the dataset by handling missing values."""
    logging.info("Cleaning data...")
    # Simple strategy: fill missing numerical values with the median and categorical with the mode.
    for column in df.select_dtypes(include=['number']).columns:
        if df[column].isnull().sum() > 0:
            median_val = df[column].median()
            df[column].fillna(median_val, inplace=True)
            logging.info(f"Filled missing values in numerical column '{column}' with median ({median_val}).")

    for column in df.select_dtypes(include=['object']).columns:
        if df[column].isnull().sum() > 0:
            mode_val = df[column].mode()[0]
            df[column].fillna(mode_val, inplace=True)
            logging.info(f"Filled missing values in categorical column '{column}' with mode ({mode_val}).")
    
    logging.info("Data cleaning complete.")
    return df

def preprocess_data(df, numerical_features, categorical_features, target_variable):
    """Preprocesses the data by scaling numerical features and encoding categorical features."""
    logging.info("Preprocessing data...")
    
    # Normalize numerical features
    scaler = MinMaxScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    logging.info(f"Normalized numerical features: {numerical_features}")

    # One-hot encode categorical features
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded_cats = encoder.fit_transform(df[categorical_features])
    encoded_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(categorical_features))
    
    # Drop original categorical columns and concatenate encoded ones
    df = df.drop(columns=categorical_features)
    df = pd.concat([df.reset_index(drop=True), encoded_df], axis=1)
    logging.info(f"One-hot encoded categorical features: {categorical_features}")
    
    logging.info("Data preprocessing complete.")
    return df

def split_data(df, target_variable):
    """Splits the data into training, validation, and test sets."""
    logging.info("Splitting data into training, validation, and test sets...")
    
    X = df.drop(columns=[target_variable])
    y = df[target_variable]
    
    # Split into training pool (80%) and a temporary set (20%)
    X_train_pool, X_temp, y_train_pool, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Split the temporary set into validation (10% of total) and test (10% of total)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    logging.info(f"Data split complete. Shapes: Train Pool ({X_train_pool.shape}), Validation ({X_val.shape}), Test ({X_test.shape})")
    
    return X_train_pool, y_train_pool, X_val, y_val, X_test, y_test

def save_data(data_dict, output_dir):
    """Saves the processed datasets to the specified directory as Parquet files."""
    logging.info(f"Saving processed data to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    for name, data in data_dict.items():
        # Combine features and target for saving if they are separate
        if isinstance(data, tuple):
            df_to_save = pd.concat(data, axis=1)
        else:
            df_to_save = data
        
        output_path = os.path.join(output_dir, f"{name}.parquet")
        df_to_save.to_parquet(output_path)
        logging.info(f"Saved {name} to {output_path}")
        
    logging.info("All data saved successfully.")

def main():
    """Main function to run the data preparation pipeline."""
    # Define paths
    raw_data_path = 'data/raw/resstock_dataset.csv' # Placeholder path
    processed_data_dir = 'data/processed'

    # Define features and target. These are placeholders and should be updated.
    # Based on ResStock, but you should verify with your actual dataset.
    numerical_features = ['SQFT', 'Insulation_R_Value', 'Window_U_Factor'] 
    categorical_features = ['Building_Type', 'Vintage', 'HVAC_System_Type', 'Climate_Zone']
    target_variable = 'Total_Annual_Energy_Consumption'

    # Check if raw data exists
    if not os.path.exists(raw_data_path):
        logging.warning(f"Raw data not found at '{raw_data_path}'.")
        logging.warning("Please place your ResStock dataset there and update feature/target lists in the script.")
        # Create a dummy file for demonstration purposes
        os.makedirs('data/raw', exist_ok=True)
        np.random.seed(42)
        
        # Create a larger dummy dataset with more realistic values
        n_samples = 1000
        dummy_df = pd.DataFrame({
            'SQFT': np.random.randint(800, 4000, n_samples),
            'Insulation_R_Value': np.random.randint(10, 50, n_samples),
            'Window_U_Factor': np.random.uniform(0.2, 0.8, n_samples),
            'Building_Type': np.random.choice(['Single-Family Detached', 'Mobile Home', 'Multi-Family', 'Townhouse'], n_samples),
            'Vintage': np.random.randint(1950, 2020, n_samples),
            'HVAC_System_Type': np.random.choice(['Central Air', 'Window AC', 'Heat Pump', 'Baseboard'], n_samples),
            'Climate_Zone': np.random.choice(['1A', '2A', '2B', '3A', '3B', '3C', '4A', '4B', '4C', '5A', '5B', '6A', '6B', '7A', '8A'], n_samples),
            'Total_Annual_Energy_Consumption': np.random.randint(8000, 20000, n_samples)
        })
        dummy_df.to_csv(raw_data_path, index=False)
        logging.info(f"Created a dummy dataset with {n_samples} samples at '{raw_data_path}'. Please replace it with your actual data.")

    # --- Pipeline ---
    # 1. Load data
    df = load_data(raw_data_path)
    
    # 2. Clean data
    df_cleaned = clean_data(df)
    
    # 3. Preprocess data
    df_processed = preprocess_data(df_cleaned, numerical_features, categorical_features, target_variable)
    
    # 4. Split data
    X_train_pool, y_train_pool, X_val, y_val, X_test, y_test = split_data(df_processed, target_variable)
    
    # 5. Save data
    # For simplicity, saving features and targets together.
    train_pool_df = pd.concat([X_train_pool, y_train_pool], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    data_to_save = {
        'train_pool': train_pool_df,
        'validation': val_df,
        'test': test_df
    }
    save_data(data_to_save, processed_data_dir)

if __name__ == '__main__':
    main() 