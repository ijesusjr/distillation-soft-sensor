"""
Distillation Column Soft-Sensor Predictor
==========================================

Interactive web application for real-time purity prediction 
using machine learning model trained on distillation column data.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

# Import all functions from utils
from utils import (
    Config,
    load_model,
    load_scaler,
    load_feature_names,
    validate_inputs,
    create_input_dataframe,
    scale_inputs,
    predict_purity,
    get_prediction_status,
    get_feature_importance,
    get_model_performance,
    format_purity_display
)

warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION & SETUP
# ============================================================================

st.set_page_config(
    page_title="Distillation Soft-Sensor",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# ============================================================================
# LOAD MODELS AND DATA (with caching)
# ============================================================================

@st.cache_resource
def get_model():
    """Load and cache the model"""
    return load_model()

@st.cache_resource
def get_scaler():
    """Load and cache the scaler"""
    return load_scaler()

@st.cache_data
def get_features():
    """Load and cache feature names"""
    return load_feature_names()

@st.cache_data
def get_importance():
    """Load and cache feature importance"""
    model = get_model()
    features = get_features()
    return get_feature_importance(model, features, top_n=15)

# ============================================================================
# REST OF YOUR APP CODE (sidebar, predictions, etc.)
# ============================================================================

# Load everything at startup
try:
    model = get_model()
    scaler = get_scaler()
    feature_names = get_features()
except Exception as e:
    st.error(f"❌ Failed to load model: {str(e)}")
    st.stop()


# ============================================================================
# HEADER SECTION
# ============================================================================

st.markdown("""
    # 🧪 Distillation Column Soft-Sensor
    ## Real-Time Purity Prediction Using Machine Learning
""")

# Display key metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Model Type", "XGBoost")
with col2:
    st.metric("R² Score", "0.9998")
with col3:
    st.metric("RMSE", "0.0010")
with col4:
    st.metric("Data Type", "Simulated")

st.markdown("""
    ---
    **Note:** This model was trained on simulated distillation column data with synthetic noise.
    Performance on real industrial data may be lower due to unmeasured disturbances and sensor errors.
""")

# ============================================================================
# SIDEBAR - USER INPUTS
# ============================================================================

st.sidebar.header("⚙️ Process Variables Input")
st.sidebar.markdown("Set the current process conditions")

st.sidebar.info("Input controls will be populated based on your feature names")

main_variables = {
    'T1': {'min': 350.76, 'max': 352.32, 'default': 350.91},
    'T4': {'min': 350.79, 'max': 368.6, 'default': 351.32},
    'T5': {'min': 350.8, 'max': 369.06, 'default': 351.62},
    'T6': {'min': 350.82, 'max': 372.57, 'default': 352.05},
    'T7': {'min': 350.86, 'max': 372.97, 'default': 352.64},
    'T13': {'min': 353.15, 'max': 373.06, 'default': 370.85},
    'T14': {'min': 354.52, 'max': 373.07, 'default': 372.83},
    'L': {'min': 75.0, 'max': 1950.0, 'default': 780.0},
    'D': {'min': 150.0, 'max': 350.0, 'default': 260.0},
    'F': {'min': 350.0, 'max': 650.0, 'default': 600.0},
    'B': {'min': 90.0, 'max': 450.0, 'default': 300.0},
}

user_inputs = {}
for var_name, config in main_variables.items():
    value = st.sidebar.slider(
        f'{var_name}',
        min_value=config['min'],
        max_value=config['max'],
        value=config['default']
    )
    user_inputs[var_name] = value
    
# Fill in lagged features (assume constant)
for var_name in main_variables:
    for lag in [1, 5, 10, 30, 60, 240]:
        lagged_name = f'{var_name}_lag{lag}'
        if lagged_name in feature_names:
            user_inputs[lagged_name] = user_inputs[var_name]

# Add time features (current hour)
import datetime
now = datetime.datetime.now()
hour = now.hour / 24.0  # 0-1
user_inputs['hour_of_day_sin'] = np.sin(2 * np.pi * hour)
user_inputs['hour_of_day_cos'] = np.cos(2 * np.pi * hour)


# Predict button
st.sidebar.markdown("---")
predict_button = st.sidebar.button("🔮 Make Prediction", key="predict_btn")

# ============================================================================
# MAIN CONTENT - PREDICTION DISPLAY
# ============================================================================

# ============================================================================
# MAIN CONTENT - PREDICTION DISPLAY
# ============================================================================

if predict_button:
    try:
        # Step 1: Validate inputs
        is_valid, message = validate_inputs(user_inputs, feature_names)
        
        if not is_valid:
            st.error(f"❌ {message}")
        else:
            # Step 2: Create DataFrame and scale
            input_df = create_input_dataframe(user_inputs, feature_names)
            scaled_inputs = scale_inputs(input_df, scaler)
            
            # Step 3: Make prediction
            purity = predict_purity(scaled_inputs, model)
            
            # Step 4: Get status
            status, color, emoji = get_prediction_status(purity)
            
            # Step 5: Format for display
            display = format_purity_display(purity, target=0.90)
            
            # Step 6: Display results
            st.markdown("## 📊 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Predicted Purity", f"{display['purity']:.4f}", emoji)
            
            with col2:
                st.metric("Target Purity", f"{display['target']:.4f}")
            
            with col3:
                st.metric("Difference", 
                         f"{display['difference']:.4f}",
                         delta=f"{display['difference_pct']:.2f}%",
                         delta_color="inverse")
            
            # Status indicator
            st.markdown(f"### Status: {emoji} {status}")
            
    except Exception as e:
        st.error(f"❌ Error making prediction: {str(e)}")

else:
    st.info("👈 Set process variables in the sidebar and click 'Make Prediction' to get started")

# ============================================================================
# FEATURE IMPORTANCE SECTION
# ============================================================================


with st.expander("📈 Model Explanation - Feature Importance"):
    st.markdown("### Top Features Affecting Purity Prediction")
    
    try:
        importance_df = get_importance()
        
        # Create chart
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(importance_df['Feature'], importance_df['Importance'], color='steelblue')
        ax.set_xlabel('Importance Score')
        ax.set_title('Top 15 Most Important Features')
        ax.invert_yaxis()
        
        st.pyplot(fig)
        
        st.markdown("""
            **Interpretation:**
            - Longer bars = more important features
            - Features at the top have the most influence on purity prediction
        """)
        
    except Exception as e:
        st.error(f"Failed to load feature importance: {str(e)}")

# ============================================================================
# MODEL INFO & ABOUT
# ============================================================================

with st.expander("ℹ️ Model Information"):
    st.markdown("### Model Performance (Test Set)")
    
    metrics = get_model_performance()
    
    perf_col1, perf_col2, perf_col3 = st.columns(3)
    
    with perf_col1:
        st.metric("R² Score", f"{metrics['r2']:.4f}")
    with perf_col2:
        st.metric("RMSE", f"{metrics['rmse']:.4f}")
    with perf_col3:
        st.metric("MAE", f"{metrics['mae']:.4f}")
    
    st.markdown("""
        ### Model Details
        - **Algorithm:** XGBoost Regressor
        - **Training Data:** Simulated distillation column (4,408 samples)
        - **Features:** 30 engineered features (lagged, rolling, cyclic)
        - **Training Split:** 70% train, 30% test
        
        ### Limitations
        - Trained on simulated data with synthetic noise
        - Real plant performance may be 10-15% lower
        - Best accuracy for purity values 0.75-0.95
        - Assumes steady-state operation
    """)

with st.expander("❓ Help & FAQ"):
    st.markdown("""
        ### How to Use
        1. Set process variables in the sidebar
        2. Click "Make Prediction"
        3. View predicted purity and status
        4. Check feature importance to understand why
        
        ### What Do the Colors Mean?
        - 🟢 **Green (>0.85):** Good purity
        - 🟡 **Orange (0.75-0.85):** Acceptable purity
        - 🔴 **Red (<0.75):** Poor purity
    """)


# ============================================================================
# ABOUT & LIMITATIONS SECTION
# ============================================================================

with st.expander("ℹ️ About This Application"):
    
    st.markdown("""
        ### What is This App?
        This is a soft-sensor application that predicts ethanol concentration (purity) in a distillation column
        using a machine learning model trained on simulated process data.
        
        ### How Does It Work?
        1. You input current process variables (temperatures, reflux ratio, feed rate, etc.)
        2. The model processes these inputs and makes a prediction
        3. The app displays the predicted purity and explains which variables most influence the prediction
        
        ### Key Limitations
        - **Trained on simulated data:** Real plant performance may be 10-15% lower
        - **Assumes steady-state operation:** Not designed for transient conditions
        - **Feature dependencies:** Requires all input variables to be provided
        - **Accuracy range:** Best accuracy for purity values 0.75-0.95
        
        ### When to Trust This Model
        ✅ When process conditions are within training range
        ✅ For decision support (not autonomous control)
        ✅ When combined with operator judgment
        
        ### When NOT to Trust This Model
        ❌ Outside training data ranges
        ❌ During process upsets or disturbances
        ❌ If sensor values are obviously wrong
        ❌ For critical safety decisions without verification
    """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Distillation Column Soft-Sensor v1.0</p>
        <p>Built with Streamlit | Machine Learning Model: XGBoost</p>
        <p><em>For demonstration purposes. Not for production use without validation.</em></p>
    </div>
""", unsafe_allow_html=True)
