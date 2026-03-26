"""
Utility Functions for Distillation Soft-Sensor
==============================================

Helper functions for data preprocessing, scaling, and predictions.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from typing import Dict, Tuple, List
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION & PATHS
# ============================================================================

class Config:
    """Configuration for the application"""
    
    PROJECT_ROOT = Path(__file__).parent
    DATA_DIR = PROJECT_ROOT / "data"
    MODEL_DIR = PROJECT_ROOT / "models"
    
    # File paths
    SCALER_PATH = MODEL_DIR / "scaler.pkl"
    MODEL_PATH = MODEL_DIR / "xgb_model.pkl"
    FEATURES_PATH = DATA_DIR / "X_ml_features.csv"
    TARGET_PATH = DATA_DIR / "y_ml_target.csv"
    
    # Model performance metrics (from the test set)
    TEST_R2 = 0.9998
    TEST_RMSE = 0.0010
    TEST_MAE = 0.0008
    
    # Process variable ranges (from EDA)

    VARIABLE_RANGES = {'T1': {'min': 350.76, 'max': 352.32, 'default': 350.91},
                        'T5_lag5': {'min': 350.8, 'max': 369.06, 'default': 351.62},
                        'T5_lag30': {'min': 350.8, 'max': 369.06, 'default': 351.6},
                        'T5_lag60': {'min': 350.8, 'max': 369.06, 'default': 351.6},
                        'L': {'min': 75.0, 'max': 1950.0, 'default': 780.0},
                        'F': {'min': 350.0, 'max': 650.0, 'default': 600.0},
                        'T7': {'min': 350.86, 'max': 372.97, 'default': 352.64},
                        'T1_lag10': {'min': 350.76, 'max': 352.32, 'default': 350.91},
                        'hour_of_day_cos': {'min': -1.0,
                        'max': 1.0,
                        'default': 6.123233995736766e-17},
                        'T4_lag5': {'min': 350.79, 'max': 368.6, 'default': 351.32},
                        'T5_lag1': {'min': 350.8, 'max': 369.06, 'default': 351.62},
                        'T4_lag1': {'min': 350.79, 'max': 368.6, 'default': 351.32},
                        'B': {'min': 90.0, 'max': 450.0, 'default': 300.0},
                        'T13': {'min': 353.15, 'max': 373.06, 'default': 370.85},
                        'T1_lag5': {'min': 350.76, 'max': 352.32, 'default': 350.91},
                        'T4_lag60': {'min': 350.79, 'max': 368.6, 'default': 351.3},
                        'hour_of_day_sin': {'min': -1.0, 'max': 1.0, 'default': 0.0261769483078791},
                        'T4_lag10': {'min': 350.79, 'max': 368.6, 'default': 351.32},
                        'T1_lag240': {'min': 350.76, 'max': 352.32, 'default': 350.9},
                        'T5_lag10': {'min': 350.8, 'max': 369.06, 'default': 351.61},
                        'D': {'min': 150.0, 'max': 350.0, 'default': 260.0},
                        'T4_lag30': {'min': 350.79, 'max': 368.6, 'default': 351.31},
                        'T6': {'min': 350.82, 'max': 372.57, 'default': 352.05},
                        'T5': {'min': 350.8, 'max': 369.06, 'default': 351.62},
                        'T5_lag240': {'min': 350.8, 'max': 369.06, 'default': 351.51},
                        'T4_lag240': {'min': 350.79, 'max': 368.6, 'default': 351.27},
                        'T4': {'min': 350.79, 'max': 368.6, 'default': 351.32},
                        'T14': {'min': 354.52, 'max': 373.07, 'default': 372.83},
                        'T1_lag30': {'min': 350.76, 'max': 352.32, 'default': 350.91},
                        'T1_lag60': {'min': 350.76, 'max': 352.32, 'default': 350.91}}
    
    # Target variable settings
    TARGET_NAME = "Ethanol Concentration"  
    TARGET_COLUMN = "Ethanol concentration"  
    PURITY_THRESHOLDS = {
        'good': 0.85,      # Good purity
        'acceptable': 0.75  # Acceptable purity (below this is poor)
    }


# ============================================================================
# MODEL LOADING FUNCTIONS
# ============================================================================

def load_model(model_path: Path = Config.MODEL_PATH):
    """
    Load trained XGBoost model
    
    Parameters:
    -----------
    model_path : Path, optional
        Path to saved model file. If None, uses Config.MODEL_PATH
    
    Returns:
    --------
    model : XGBRegressor
        Trained XGBoost model
    
    Raises:
    -------
    FileNotFoundError
        If model file not found
    """

    try:
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
        return model
        
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")
    


def load_scaler(scaler_path: Path = Config.SCALER_PATH):
    """
    Load fitted StandardScaler
    
    Parameters:
    -----------
    scaler_path : Path, optional
        Path to saved scaler file. If None, uses Config.SCALER_PATH
    
    Returns:
    --------
    scaler : StandardScaler
        Fitted scaler for feature normalization
    
    Raises:
    -------
    FileNotFoundError
        If scaler file not found
    """
    try:
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found at {scaler_path}")
        
        scaler = joblib.load(scaler_path)
        print(f"Scaler loaded from {scaler_path}")
        return scaler
        
    except Exception as e:
        raise RuntimeError(f"Failed to load scaler: {str(e)}")


def load_feature_names(features_path: Path = Config.FEATURES_PATH) -> List[str]:
    """
    Load feature names from training data
    
    Parameters:
    -----------
    features_path : Path, optional
        Path to X_ml_features.csv. If None, uses Config.FEATURES_PATH
    
    Returns:
    --------
    feature_names : List[str]
        List of feature column names in correct order
    """

    try:
        if not features_path.exists():
            raise FileNotFoundError(f"Features file not found at {features_path}")
        
        df = pd.read_csv(features_path)
        feature_names = df.columns.tolist()
        print(f"Loaded {len(feature_names)} features")
        return feature_names
        
    except Exception as e:
        raise RuntimeError(f"Failed to load feature names: {str(e)}")

# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================

def validate_inputs(inputs: Dict[str, float], feature_names: List[str]) -> Tuple[bool, str]:
    """
    Validate user inputs
    
    Parameters:
    -----------
    inputs : Dict[str, float]
        Dictionary of user inputs {feature_name: value}
    feature_names : List[str]
        Expected feature names from training
    
    Returns:
    --------
    is_valid : bool
        True if inputs are valid
    message : str
        Validation message or error description
    """
    # TODO:
    # 1. Check all required features are present
    # 2. Check values are within expected ranges
    # 3. Check for NaN or infinite values
    # 4. Return (True, "OK") if valid
    # 5. Return (False, error_message) if not valid
    
    input_features = list(inputs.keys())
    
    if set(input_features) != set(feature_names):
        return False, "Error: NOT all required features are present"
    
    for feature in feature_names:
        if inputs[feature] < Config.VARIABLE_RANGES[feature]['min'] or inputs[feature] > Config.VARIABLE_RANGES[feature]['max']:
            return False, "Error: values are NOT within expected ranges"
    
    for v in inputs.values():
        if pd.isnull(v) or np.isinf(v):
            return False, "Error: there are NaN of infinite in the inputs"
        
    return True, 'OK'
    
    
 
    
    
    


def create_input_dataframe(inputs: Dict[str, float], feature_names: List[str]) -> pd.DataFrame:
    """
    Create DataFrame from user inputs in correct feature order
    
    Parameters:
    -----------
    inputs : Dict[str, float]
        User inputs {feature_name: value}
    feature_names : List[str]
        Feature names in correct order
    
    Returns:
    --------
    input_df : pd.DataFrame
        Single row DataFrame with features in correct order
    """
    
    inputs_reordered = {key:inputs[key] for key in feature_names}
    
    inputs_df = pd.DataFrame([inputs_reordered])
    
    assert inputs_df.shape == (1, len(feature_names)), \
        f"Expected shape (1, {len(feature_names)}), got {inputs_df.shape}"
    
    return inputs_df


def scale_inputs(input_df: pd.DataFrame, scaler) -> np.ndarray:
    """
    Scale input features using fitted scaler
    
    Parameters:
    -----------
    input_df : pd.DataFrame
        Input features (1, n_features)
    scaler : StandardScaler
        Fitted scaler from training
    
    Returns:
    --------
    scaled_inputs : np.ndarray
        Scaled input array (1, n_features)
    """
    
    try:
        input_scaled = scaler.transform(input_df)
        input_scaled = np.array(input_scaled)
        return input_scaled

    except Exception as e:
        raise RuntimeError(f"Failed to scale inputs: {str(e)}")    


# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def predict_purity(scaled_inputs: np.ndarray, model) -> float:
    """
    Make purity prediction
    
    Parameters:
    -----------
    scaled_inputs : np.ndarray
        Scaled input features (1, n_features)
    model : XGBRegressor
        Trained XGBoost model
    
    Returns:
    --------
    prediction : float
        Predicted purity value
    """

    purity = float(model.predict(scaled_inputs)[0])
    
    return purity


def get_prediction_status(purity: float) -> Tuple[str, str, str]:
    """
    Determine purity status and color coding
    
    Parameters:
    -----------
    purity : float
        Predicted purity value [0, 1]
    
    Returns:
    --------
    status : str
        Status: 'Good', 'Acceptable', or 'Poor'
    color : str
        Color for UI: 'green', 'orange', or 'red'
    emoji : str
        Status emoji
    """
    if purity > 0.85: 
        return ('Good', 'green', '✅')
    elif purity > 0.75: 
        return ('Acceptable', 'orange', '⚠️')
    else:
        return ('Poor', 'red', '❌')
    


def get_feature_importance(model, feature_names: List[str], top_n: int = 15) -> pd.DataFrame:
    """
    Get feature importance from trained model
    
    Parameters:
    -----------
    model : XGBRegressor
        Trained XGBoost model
    feature_names : List[str]
        Feature names corresponding to model
    top_n : int
        Number of top features to return
    
    Returns:
    --------
    importance_df : pd.DataFrame
        DataFrame with columns ['Feature', 'Importance']
        Sorted by importance descending
    """
   
    importance_df = pd.DataFrame({
    'Feature': model.feature_names_in_,
    'Importance': model.feature_importances_
    })

    importance_df = importance_df.sort_values(by='Importance', ascending=False).head(top_n)
    
    return importance_df


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def get_model_performance() -> Dict[str, float]:
    """
    Get model performance metrics
    
    Returns:
    --------
    metrics : Dict[str, float]
        Dictionary with keys: 'r2', 'rmse', 'mae'
    """
    
    return {'r2':Config.TEST_R2, 'rmse':Config.TEST_RMSE, 'mae':Config.TEST_MAE}


def format_purity_display(purity: float, target: float = 0.90) -> Dict:
    """
    Format purity for display
    
    Parameters:
    -----------
    purity : float
        Predicted purity
    target : float
        Target/setpoint purity
    
    Returns:
    --------
    display_dict : Dict
        Contains: 'purity', 'target', 'difference', 'difference_pct'
    """
        
    display_dict = {'purity':round(purity,4), 'target':round(target,4), 'difference':round((purity-target), 4), 'difference_pct':round(100*((purity-target)/target), 2)}
    
    return display_dict


# ============================================================================
# ERROR HANDLING & LOGGING
# ============================================================================

def create_error_message(error: Exception, context: str) -> str:
    """
    Create user-friendly error message
    
    Parameters:
    -----------
    error : Exception
        The exception that occurred
    context : str
        What was being done when error occurred
    
    Returns:
    --------
    message : str
        User-friendly error message
    """
    
    error_type = type(error).__name__
    error_msg = str(error)
    
    # Simple message
    message = (
        f"❌ Error while {context}\n\n"
        f"**Problem:** {error_type}\n"
        f"**Details:** {error_msg}\n\n"
        f"**What to do:**\n"
        f"- Check that all files are in correct folders\n"
        f"- Verify your inputs are valid\n"
        f"- Try again or restart the app"
    )
    
    return message