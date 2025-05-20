# pylint: disable=too-many-arguments, too-many-locals, invalid-name
"""
Reusable script for training and evaluating time-series forecasting models
for load prediction, adapted from a Jupyter Notebook.
"""

import argparse
import logging
from typing import Tuple, Dict, Any, List, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split # Example if not using sequential split
from sklearn.ensemble import RandomForestRegressor # Example for an alternative model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# --- Configuration ---
DATA_FILE_PATH = "data/final/Load_weather_history.csv"
ZONE_ID_TO_FILTER = 1
TARGET_COLUMN = "load_kw"
TIME_COLUMN = "time" # Keep track of the original time column for potential analysis
DROP_COLUMNS_PRE_SPLIT = [TARGET_COLUMN, TIME_COLUMN, "original_time_col"] # Columns to drop before splitting into X, y

# Train/Validation/Test split ratios for sequential data
TRAIN_RATIO = 0.7
VALIDATION_RATIO = 0.15
# TEST_RATIO is implicitly 1 - TRAIN_RATIO - VALIDATION_RATIO

# Model configurations
# Add more model configurations here or load from a config file
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "xgboost": {
        "model_class": xgb.XGBRegressor,
        "params": {
            "n_estimators": 1000,
            "learning_rate": 0.01,
            "max_depth": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
            #tree_method='hist' # Faster for large datasets, good for GPU too
        },
        "fit_params": {
            
            "verbose": False # Set to True or an integer for more verbose output during training
        }
    },
    "random_forest": {
        "model_class": RandomForestRegressor,
        "params": {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42,
            "n_jobs": -1,
            "min_samples_split": 5,
            "min_samples_leaf": 3
        },
        "fit_params": {} # RandomForestRegressor doesn't have early stopping in the same way
    }
    # Add other models like LinearRegression, SVR, etc.
    # "linear_regression": {
    #     "model_class": LinearRegression,
    #     "params": {},
    #     "fit_params": {}
    # }
}

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_and_prepare_data(file_path: str, zone_id: int) -> pd.DataFrame:
    """
    Loads data from a CSV file, filters by zone_id, and performs initial cleaning.

    Args:
        file_path (str): Path to the CSV data file.
        zone_id (int): Zone ID to filter the data by.

    Returns:
        pd.DataFrame: Processed DataFrame.
    """
    logging.info(f"Loading data from {file_path} for zone_id {zone_id}...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        logging.error(f"Error: Data file not found at {file_path}")
        raise
    
    logging.info(f"Initial DataFrame shape: {df.shape}")
    
    df = df[df["zone_id"] == zone_id].copy() # Use .copy() to avoid SettingWithCopyWarning
    if df.empty:
        logging.warning(f"No data found for zone_id {zone_id}. Exiting.")
        return pd.DataFrame()
        
    df.drop(columns=["zone_id"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    logging.info(f"DataFrame shape after filtering and cleaning: {df.shape}")
    return df


def engineer_features(df: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
    """
    Performs feature engineering on the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        time_col (str): Name of the time column.

    Returns:
        pd.DataFrame: DataFrame with engineered features.
    """
    logging.info("Starting feature engineering...")
    df_processed = df.copy()

    # Ensure time column is datetime
    try:
        df_processed[time_col] = pd.to_datetime(df_processed[time_col])
    except Exception as e:
        logging.error(f"Error converting '{time_col}' to datetime: {e}")
        raise

    # Feature: Hour of the day
    df_processed["hour"] = df_processed[time_col].dt.hour
    
    # Feature: Month (as requested in notebook markdown, though not implemented in original code)
    # df_processed["month"] = df_processed[time_col].dt.month 
    # df_processed = pd.get_dummies(df_processed, columns=["month"], prefix="month", dummy_na=False)


    # Feature: Day of the week (as originally in the notebook)
    if "day_of_week" in df_processed.columns:
        df_processed = pd.get_dummies(df_processed, columns=["day_of_week"], prefix="day", dummy_na=False)
    else:
        logging.warning("'day_of_week' column not found. Skipping day_of_week dummies.")

    # Feature: Cyclical features for hour (optional, but good for time series)
    # df_processed['hour_sin'] = np.sin(2 * np.pi * df_processed['hour'] / 24)
    # df_processed['hour_cos'] = np.cos(2 * np.pi * df_processed['hour'] / 24)

    # Store original time column if needed for later, then drop it if it's not a feature
    df_processed['original_time_col'] = df_processed[time_col]
    
    logging.info("Feature engineering complete.")
    logging.info(f"DataFrame shape after feature engineering: {df_processed.shape}")
    logging.info(f"Columns: {df_processed.columns.tolist()}")
    return df_processed


def split_data_sequential(
    df: pd.DataFrame,
    target_col: str,
    feature_cols_to_drop: List[str],
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VALIDATION_RATIO
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Splits data sequentially into training, validation, and test sets.

    Args:
        df (pd.DataFrame): DataFrame to split.
        target_col (str): Name of the target variable column.
        feature_cols_to_drop (List[str]): Columns to drop to create the feature set (X).
                                          This typically includes the target and any non-feature time columns.
        train_ratio (float): Proportion of data for training.
        val_ratio (float): Proportion of data for validation.

    Returns:
        Tuple: X_train, y_train, X_val, y_val, X_test, y_test
    """
    logging.info("Splitting data sequentially...")
    
    y_all = df[target_col]
    # Ensure all columns in feature_cols_to_drop exist in df before dropping
    cols_to_drop_present = [col for col in feature_cols_to_drop if col in df.columns]
    X_all = df.drop(columns=cols_to_drop_present)

    n_total = len(df)
    train_end_idx = int(n_total * train_ratio)
    val_end_idx = int(n_total * (train_ratio + val_ratio))

    X_train = X_all.iloc[:train_end_idx]
    y_train = y_all.iloc[:train_end_idx]

    X_val = X_all.iloc[train_end_idx:val_end_idx]
    y_val = y_all.iloc[train_end_idx:val_end_idx]

    X_test = X_all.iloc[val_end_idx:]
    y_test = y_all.iloc[val_end_idx:]

    logging.info(f"Train set: X_train shape {X_train.shape}, y_train shape {y_train.shape}")
    logging.info(f"Validation set: X_val shape {X_val.shape}, y_val shape {y_val.shape}")
    logging.info(f"Test set: X_test shape {X_test.shape}, y_test shape {y_test.shape}")
    
    # Check for NaN values that might cause issues
    if X_train.isnull().any().any():
        logging.warning("NaN values found in X_train. Consider imputation.")
    if X_val.isnull().any().any():
        logging.warning("NaN values found in X_val. Consider imputation.")
    if X_test.isnull().any().any():
        logging.warning("NaN values found in X_test. Consider imputation.")
        
    return X_train, y_train, X_val, y_val, X_test, y_test

def scale_features(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Scales numerical features using StandardScaler.

    Args:
        X_train, X_val, X_test: Feature sets.

    Returns:
        Scaled feature sets and the fitted scaler.
    """
    logging.info("Scaling features...")
    scaler = StandardScaler()
    
    # Identify numerical columns (exclude boolean/object if any were kept)
    numerical_cols_train = X_train.select_dtypes(include=np.number).columns
    
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[numerical_cols_train] = scaler.fit_transform(X_train[numerical_cols_train])
    
    # Ensure val and test sets have the same numerical columns before transforming
    numerical_cols_val = [col for col in numerical_cols_train if col in X_val.columns]
    numerical_cols_test = [col for col in numerical_cols_train if col in X_test.columns]

    if numerical_cols_val:
        X_val_scaled[numerical_cols_val] = scaler.transform(X_val[numerical_cols_val])
    if numerical_cols_test:
        X_test_scaled[numerical_cols_test] = scaler.transform(X_test[numerical_cols_test])
        
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_name: str,
    model_configs: Dict[str, Dict[str, Any]] = MODEL_CONFIGS
) -> Any:
    """
    Initializes, trains, and returns a model.

    Args:
        X_train, y_train: Training data.
        X_val, y_val: Validation data.
        model_name (str): Name of the model to train (key in model_configs).
        model_configs (Dict): Dictionary containing model classes and parameters.

    Returns:
        Any: Trained model object.
    """
    logging.info(f"Training model: {model_name}...")
    if model_name not in model_configs:
        raise ValueError(f"Model '{model_name}' not found in MODEL_CONFIGS.")

    config = model_configs[model_name]
    model = config["model_class"](**config["params"])

    fit_params = config["fit_params"].copy() # Use a copy to modify
    
    # For models that support evaluation sets (like XGBoost)
    if "early_stopping_rounds" in fit_params and X_val is not None and y_val is not None:
        fit_params["eval_set"] = [(X_val, y_val)]
        if model_name == "xgboost" and "verbose" not in config["params"]: # XGB specific verbose
             # fit_params["verbose"] = True # or False, or an int
             pass


    try:
        model.fit(X_train, y_train, **fit_params)
    except Exception as e:
        logging.error(f"Error during model training for {model_name}: {e}")
        logging.error(f"X_train columns: {X_train.columns.tolist()}")
        logging.error(f"X_val columns (if used): {X_val.columns.tolist() if X_val is not None else 'N/A'}")
        # Log dtypes
        logging.error(f"X_train dtypes:\n{X_train.dtypes}")
        if X_val is not None:
             logging.error(f"X_val dtypes:\n{X_val.dtypes}")
        raise
        
    logging.info(f"{model_name} training complete.")
    return model


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str
) -> Dict[str, float]:
    """
    Evaluates the model on the test set and prints metrics.

    Args:
        model (Any): Trained model.
        X_test, y_test: Test data.
        model_name (str): Name of the model for logging.

    Returns:
        Dict[str, float]: Dictionary of evaluation metrics.
    """
    logging.info(f"Evaluating {model_name} on the test set...")
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
    logging.info(f"MAPE: {mape:.2f}%")

    logging.info(f"--- {model_name} Test Set Metrics ---")
    logging.info(f"MAE: {mae:.4f}")
    logging.info(f"RMSE: {rmse:.4f}")
    logging.info(f"R²: {r2:.4f}")
    
    metrics = {"MAE": mae, "RMSE": rmse, "R2": r2}
    
    # Plot actual vs. predicted
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.values, label='Actual Load', alpha=0.7)
    plt.plot(predictions, label='Predicted Load', linestyle='--', alpha=0.7)
    plt.title(f'{model_name} - Actual vs. Predicted Load on Test Set')
    plt.xlabel('Time Steps (Test Set)')
    plt.ylabel('Load (kW)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/{model_name}_actual_vs_predicted.png")
    logging.info(f"Saved actual vs. predicted plot to {model_name}_actual_vs_predicted.png")
    # plt.show() # Uncomment if running interactively and want to display plots

    return metrics

def plot_feature_importance(model: Any, feature_names: List[str], model_name: str, top_n: int = 20):
    """
    Plots feature importances for tree-based models.

    Args:
        model (Any): Trained tree-based model (e.g., XGBoost, RandomForest).
        feature_names (List[str]): List of feature names.
        model_name (str): Name of the model for plot title.
        top_n (int): Number of top features to display.
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        sorted_indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, max(6, top_n // 2))) # Adjust height based on top_n
        sns.barplot(x=importances[sorted_indices][:top_n], y=np.array(feature_names)[sorted_indices][:top_n])
        plt.title(f'{model_name} - Top {top_n} Feature Importances')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig(f"{model_name}_feature_importance.png")
        logging.info(f"Saved feature importance plot to {model_name}_feature_importance.png")
        # plt.show() # Uncomment for interactive display
    else:
        logging.info(f"Model {model_name} does not have 'feature_importances_' attribute. Skipping plot.")


def main(model_to_run: str, data_path: str, scale_data: bool = False):
    """
    Main function to run the load forecasting pipeline.
    """
    logging.info(f"Starting pipeline with model: {model_to_run}")

    # 1. Load and Prepare Data
    df_prepared = load_and_prepare_data(data_path, ZONE_ID_TO_FILTER)
    if df_prepared.empty:
        return

    # 2. Feature Engineering
    df_featured = engineer_features(df_prepared, time_col=TIME_COLUMN)

    # 3. Split Data
    X_train, y_train, X_val, y_val, X_test, y_test = split_data_sequential(
        df_featured,
        target_col=TARGET_COLUMN,
        feature_cols_to_drop=DROP_COLUMNS_PRE_SPLIT
    )
    
    # Store feature names for later use (e.g., plotting importance)
    feature_names = X_train.columns.tolist()

    # 4. (Optional) Scale Features
    if scale_data:
        X_train, X_val, X_test, _ = scale_features(X_train, X_val, X_test)
        logging.info("Data scaling applied.")


    # 5. Train Model
    # Ensure X_val and y_val are not empty for models using early stopping
    eval_X, eval_y = (X_val, y_val) if not X_val.empty else (None, None)
    
    trained_model = train_model(X_train, y_train, eval_X, eval_y, model_to_run, MODEL_CONFIGS)

    # 6. Evaluate Model
    evaluate_model(trained_model, X_test, y_test, model_to_run)

    # 7. Plot Feature Importance (if applicable)
    plot_feature_importance(trained_model, feature_names, model_to_run)
    
    logging.info(f"Pipeline finished for model: {model_to_run}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Load Forecasting Model Training Pipeline.")
    parser.add_argument(
        "--model",
        type=str,
        default="xgboost", # Default model to run
        choices=MODEL_CONFIGS.keys(),
        help=f"Model to train. Choices: {list(MODEL_CONFIGS.keys())}"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=DATA_FILE_PATH,
        help="Path to the input CSV data file."
    )
    parser.add_argument(
        "--scale",
        action="store_true", # Use 'store_true' for boolean flags
        help="Apply StandardScaler to features if specified."
    )
    
    args = parser.parse_args()

    main(model_to_run=args.model, data_path=args.data_path, scale_data=args.scale)

    # --- Example of running other models (can be triggered via CLI argument) ---
    # logging.info("\n--- Running RandomForest Example (if configured) ---")
    # if "random_forest" in MODEL_CONFIGS:
    #     main(model_to_run="random_forest", data_path=args.data_path, scale_data=args.scale)