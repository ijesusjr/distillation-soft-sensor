# Distillation Column Soft-Sensor: ML-Based Real-Time Purity Prediction

![Distillation](https://img.shields.io/badge/Domain-Chemical%20Engineering-blue)
![ML](https://img.shields.io/badge/ML%20Framework-XGBoost-green)
![Deployment](https://img.shields.io/badge/Deployment-Streamlit-red)
![Python](https://img.shields.io/badge/Python-3.8+-yellow)

---

## 📋 Overview

### **Problem Statement**

Distillation columns are critical unit operations in the chemical and petroleum industries for separating mixtures based on boiling points. Real-time monitoring of **product purity (ethanol concentration)** is essential for:
- Process control and optimization
- Product quality assurance  
- Cost reduction through efficient operation
- Safe operation within specifications

Traditional measurement methods (lab analysis) are:
- **Time-consuming** (results take hours)
- **Expensive** (require frequent sampling)
- **Delayed** (information arrives too late for real-time control)

**Solution:** Build a **soft-sensor** (virtual sensor) using machine learning to predict purity in real-time from readily available process variables.

---

### **Stakeholders**

- **Plant Operators:** Need real-time purity feedback for process control
- **Process Engineers:** Want to optimize column operation and efficiency
- **Quality Assurance:** Requires continuous purity verification
- **Production Management:** Seeks cost reduction and operational efficiency

---

### **Data Source & Credits**

**Dataset:** [Distillation Column Dataset on Kaggle](https://www.kaggle.com/datasets/jorgecote/distillation-column)

**Creator:** Jorge Cote

**Type:** Simulated distillation column with synthetic noise added

---

### **Analysis Conducted**

#### **Phase 1: Exploratory Data Analysis (EDA)** → `01_eda.ipynb`

**Data Overview:**
- **Total samples:** 4,408 timesteps
- **Sampling interval:** 0.1 hours (6 minutes)
- **Raw features:** 21 process variables
- **Target variable:** Ethanol concentration (purity)
- **Target range:** 0.60 - 1.00 (60% - 100%)

**Key Analyses:**

1. **Time-Series Decomposition**
   - Identified **deterministic trend:** Gradual downward drift in purity (~0.85 → 0.70)
   - Detected **24-hour seasonality:** Clear daily cycle pattern (amplitude: ±0.01-0.015)
   - Analyzed **residuals:** Random noise with occasional spikes indicating process upsets

2. **Autocorrelation (ACF/PACF) Analysis**
   - **PACF:** Significant up to lag 4 (~24 minutes) → direct process response time
   - **ACF:** Significant up to lag 60 (~6 hours) → full process memory window
   - **Lag recommendation:** Use lags 1, 5, 10, 30, 60 for feature engineering
   - **Noise zone:** Lags 60-200 show no predictive power
   - **Daily cycle:** ACF returns outside confidence interval after lag 200 (24 hours)

3. **Stationarity Testing (ADF Test)**
   - Result: Deterministic trend (not unit root)
   - Interpretation: Series has drifting mean but stable pattern
   - Decision: Use series as-is; XGBoost learns trends naturally

4. **Feature Correlation Analysis**
   - Identified high correlations with target
   - Removed multicollinear features (r > 0.95)
   - Retained diverse feature set for model

#### **Phase 2: Feature Engineering**

**Engineered Features (50 total → 30 final):**

1. **Lagged Features** (6 lags for each temperature/flow variable)
   - Lags: 1, 5, 10, 30, 60, 240 timesteps
   - Rationale: Captured from ACF analysis (lags up to 60 are significant)
   - Example: T1_lag1, T1_lag5, T1_lag10, T1_lag30, T1_lag60, T1_lag240

2. **Rolling Statistics** (mean, std over time windows)
   - 5-sample window (0.5 hours)
   - 30-sample window (3 hours)
   - Captures trends and volatility in local windows

3. **Domain-Specific Features**
   - Temperature ratios (T1/T5, T4/T6)
   - Reflux/Feed ratio (L/F) - thermodynamically meaningful
   - Energy balance features

4. **Cyclic Time Encoding**
   - hour_of_day_sin = sin(2π × hour/24)
   - hour_of_day_cos = cos(2π × hour/24)
   - Captures 24-hour seasonality identified in decomposition

**Feature Selection:**
- Started with ~50 engineered features
- Dropped highly correlated features (r > 0.95)
- **Final feature set: 30 features**
- Final features: T4_lag60, T4_lag1, T5_lag240, T4_lag10, T5_lag1, T5, T1_lag5, T4_lag30, T5_lag30, T7, T4_lag5, T1_lag10, T13, T5_lag10, T1_lag60, T14, T6, T5_lag60, T4_lag240, T1, T4, T1_lag240, T5_lag5, T1_lag30, L, F, B, D, hour_of_day_cos, hour_of_day_sin

#### **Phase 3: Model Development** → `02_ml_modeling.ipynb`

**Data Splitting:**
- Train: 70% (3,086 samples)
- Test: 30% (1,322 samples)
- **Chronological split** (preserves time-series structure, prevents leakage)

**Models Trained & Evaluated:**

| Model | Test R² | Test RMSE | Test MAE | Key Characteristics |
|-------|---------|-----------|----------|-------------------|
| **Linear Regression** | 0.9859 | 0.0080 | 0.0062 | Simple, interpretable, assumes linear relationships |
| **XGBoost** ✅ | **0.9998** | **0.0010** | **0.0008** | Best accuracy, captures interactions, gradient boosting |

**Winner: XGBoost**
- **87.5% improvement in RMSE over Linear Regression** (0.008 → 0.001)
- Captures non-linear threshold effects and feature interactions
- Successfully models complex distillation dynamics

---

### **Key Decisions Made**

1. **XGBoost Selected Over Linear Regression**
   - Linear Regression: R² = 0.9859 (good baseline)
   - XGBoost: R² = 0.9998 (near-perfect)
   - Decision: 87.5% RMSE improvement justifies added complexity

2. **No RNN/LSTM Implemented**
   - Simulated data has clean, deterministic patterns
   - XGBoost already achieves near-perfect accuracy
   - RNN would add complexity without meaningful gain
   - Prioritized deployment speed and interpretability

3. **Feature Engineering Strategy**
   - Lagged features capture temporal dependencies
   - Cyclic encoding explicitly models 24-hour pattern
   - Rolling statistics capture local trends
   - Removed correlated features to prevent overfitting

4. **Chronological Train/Test Split**
   - Respects time-series structure
   - Prevents data leakage (train doesn't see future)
   - Validates real-world deployment scenario

---

### **Results Summary**

**XGBoost Test Set Performance:**
- **R² Score:** 0.9998 (explains 99.98% of variance)
- **RMSE:** 0.0010 (purity units)
- **MAE:** 0.0008 (mean absolute error)
- **Training Time:** ~2 seconds
- **Inference Time:** ~1ms per prediction

---

## 🎯 Key Finding: Simulated Data Characteristics

### ⚠️ Important Note on Model Results

This model achieves **R² = 0.9998** with exceptional performance metrics. However, it's crucial to understand why:

**XGBoost Feature Importance Distribution:**

| Feature | Importance | Percentage |
|---------|-----------|-----------|
| T1 (Current Temperature) | 9.909e-01 | 99.09% |
| T5 (Temperature) | 2.791e-03 | 0.28% |
| T6 (Temperature) | 2.124e-03 | 0.21% |
| T4_lag1 (Lagged Temperature) | 1.091e-03 | 0.11% |
| T1_lag5 (Lagged Temperature) | 6.952e-04 | 0.07% |
| Other 25 features | ~4.5e-04 | 0.24% |

### **Why T1 Dominates (99.09% Importance)**

**This is EXPECTED and APPROPRIATE for simulated data:**

1. **Simulated Data Characteristics:**
   - Synthetic data has perfect mathematical relationships
   - Temperature is the primary determinant of purity in the simulation
   - Significantly less noise/uncertainty than real industrial data
   - Deterministic patterns lead to dominant features

2. **Physical Justification:**
   - In actual distillation, temperature directly drives separation
   - Column top temperature (T1) is the strongest control variable
   - This aligns with thermodynamic principles of distillation
   - The simulation accurately reflects physical reality

3. **Expected Behavior in Production:**
   - Real plant data would show more distributed feature importance
   - Unmeasured disturbances would increase value of lagged features
   - Sensor noise would reduce dominance of single features
   - Model R² would realistically be 0.85-0.90 (not 0.9998)

4. **Model Quality Assessment:**
   - ✅ R² = 0.9998 is excellent and APPROPRIATE for simulated data
   - ✅ Single dominant feature is NORMAL for high-quality simulations
   - ✅ Lagged features provide incremental value (~0.07% each)
   - ✅ Model successfully identifies true driving variable

### **Implications for Deployment**

- ✅ Model works perfectly on simulated data (as expected)
- ⚠️ Real plant validation is critical before production use
- 📌 On real industrial data, expect:
  - More distributed feature importance (5-15% top features)
  - Lower R² (~0.85-0.90 considered excellent)
  - Greater value from lagged/rolling features (10-20% combined)
  - Need for periodic model retraining with real data

**This is NOT a problem** — it demonstrates that the model correctly identifies that temperature is the primary driver of purity, exactly as physical theory predicts!

---

## 🎯 App Demo

### **Live Application**

[Streamlit Cloud URL - Deploy Instructions Below]

### **Features:**

- **Real-time Predictions:** Set process variables and get instant purity prediction
- **Interactive Sliders:** Control 11 main process variables
  - Temperatures: T1, T4, T5, T6, T7, T13, T14
  - Flow rates: L (Reflux), D (Distillate), F (Feed), B (Bottom product)
- **Automatic Lag Handling:** App fills lagged features assuming constant history
- **Status Indicators:** Color-coded purity status
  - 🟢 **Green (>0.85):** Good
  - 🟡 **Orange (0.75-0.85):** Acceptable
  - 🔴 **Red (<0.75):** Poor
- **Feature Importance:** Visualize which variables most influence predictions
- **Model Explainability:** Understand decision-making process

### **Quick Start (Local):**

```bash
streamlit run app.py
```

Then:
1. Adjust sliders in sidebar
2. Click "🔮 Make Prediction"
3. View results and insights

---

## 📊 Dataset

### **Source & Credits**

**Original Dataset:** [Distillation Column on Kaggle](https://www.kaggle.com/datasets/jorgecote/distillation-column)

**Creator Credit:** Jorge Cote

**Type:** Simulated distillation column with synthetic noise

### **Dataset Specifications**

| Property | Value |
|----------|-------|
| **Samples** | 4,408 timesteps |
| **Interval** | 0.1 hours (6 minutes) |
| **Duration** | ~18.3 days |
| **Raw Variables** | 21 |
| **Target Variable** | Ethanol concentration |
| **Target Range** | 0.60 - 1.00 |
| **Missing Values** | None |

### **Variables (21 Raw Features)**

**Temperatures (7):**
- T1, T4, T5, T6, T7, T13, T14 (in °C)

**Flow Rates (4):**
- L: Reflux rate
- D: Distillate (top product) rate
- F: Feed rate
- B: Bottom product rate

**Compositions/Other (10):**
- Pressure, composition measurements, etc.

### **Target Variable: Ethanol Concentration**

```
Statistics:
  Mean: 0.9162 (91.62%)
  Std Dev: 0.0783
  Min: 0.6007 (60.07%)
  Max: 0.9999 (99.99%)
  
Physical Meaning:
  - 0.60 to 0.75: Poor separation
  - 0.75 to 0.85: Acceptable purity
  - >0.85: Good/high purity
```

### **Key Time-Series Properties**

1. **Trend:** Gradual downward drift (non-stationary)
2. **Seasonality:** Strong 24-hour cycle (240 timesteps = 24 hours)
3. **Autocorrelation:** Decays significantly after lag 60 (~6 hours)
4. **Noise:** Synthetic Gaussian (~0.01 std)

---

## 🏗️ Model Architectures

### **1. Linear Regression (Baseline)**

```
Model: purity = β₀ + Σ(βᵢ × featureᵢ)

Characteristics:
  ✓ Fully interpretable
  ✓ Fast training/inference
  ✓ Shows direct effect sizes

Limitations:
  ✗ Assumes linear relationships
  ✗ Cannot model interactions
  ✗ Higher prediction error

Test Performance:
  R² = 0.9859
  RMSE = 0.0080
  MAE = 0.0062
```

### **2. XGBoost (Primary Model)** ✅ SELECTED

```
Model: Gradient Boosting with Regularization

Algorithm:
  - Sequentially builds decision trees
  - Each tree corrects previous errors (gradient boosting)
  - Regularization prevents overfitting
  - 100 boosting rounds

Strengths:
  ✓ Best accuracy (R² = 0.9998)
  ✓ Handles non-linearity
  ✓ Captures feature interactions
  ✓ Fast inference (~1ms/prediction)
  ✓ Built-in regularization (L1 & L2)
  ✓ Feature importance available

Test Performance:
  R² = 0.9998 (explains 99.98% of variance)
  RMSE = 0.0010 (87.5% better than Linear)
  MAE = 0.0008
  Training: ~2 seconds
  Inference: ~1ms/prediction
```

---

## 🔄 Pipeline

### **Complete Data Flow**

```
┌─────────────────────────────────────────────────────────────┐
│ RAW DATA: dataset_distill.csv (4,408 samples × 21 features) │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 1: EXPLORATORY DATA ANALYSIS (01_eda.ipynb)           │
├─────────────────────────────────────────────────────────────┤
│ • Time-series decomposition (trend, seasonality, residuals) │
│ • ACF/PACF analysis → lag selection (1,5,10,30,60)         │
│ • Stationarity testing (ADF)                                │
│ • Correlation analysis & outlier detection                  │
│ • Feature engineering (50 → 30 features after filtering)   │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ ENGINEERED DATA: X_ml_features.csv (30 features)            │
│ TARGET DATA: y_ml_target.csv (Ethanol concentration)        │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 2: MODEL DEVELOPMENT (02_ml_modeling.ipynb)           │
├─────────────────────────────────────────────────────────────┤
│ • Train/test split: 70/30 (chronological)                  │
│ • Feature scaling: StandardScaler                           │
│ • Baseline models: Linear Regression                        │
│ • Primary model: XGBoost                                    │
│ • Evaluation: R², RMSE, MAE, feature importance             │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ TRAINED MODELS                                              │
│ • xgb_model.pkl (XGBoost Regressor)                        │
│ • scaler.pkl (StandardScaler)                              │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 3: DEPLOYMENT (app.py + utils.py)                     │
├─────────────────────────────────────────────────────────────┤
│ • Load pre-trained models                                   │
│ • User input → feature engineering                          │
│ • Real-time predictions                                     │
│ • Visualization & explanations                              │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ STREAMLIT WEB APPLICATION (LIVE)                            │
│ Interactive soft-sensor dashboard                           │
└─────────────────────────────────────────────────────────────┘
```

### **Feature Engineering Workflow**

```
INPUT: 21 raw variables
    ↓
[1] LAGGED FEATURES
    For each variable: lag 1, 5, 10, 30, 60, 240
    (Based on ACF decay at lag 60)
    → 126 lagged features
    ↓
[2] ROLLING STATISTICS
    Mean & std over 5 & 30 sample windows
    → 84 rolling features
    ↓
[3] DOMAIN FEATURES
    Temperature ratios, reflux/feed ratio
    → 10 domain features
    ↓
[4] TIME FEATURES
    hour_of_day_sin, hour_of_day_cos
    (Captures 24-hour seasonality)
    → 2 cyclic features
    ↓
[5] CORRELATION FILTERING
    Drop features with |r| > 0.95
    → Remove multicollinearity
    ↓
OUTPUT: 30 engineered features
```

---

## 💻 Installation

### **Prerequisites**

- Python 3.8+
- pip or conda
- Git

### **Step 1: Clone Repository**

```bash
git clone https://github.com/YOUR_USERNAME/distillation-soft-sensor.git
cd distillation-soft-sensor
```

### **Step 2: Create Virtual Environment**

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Or conda
conda create -n distillation python=3.9
conda activate distillation
```

### **Step 3: Install Dependencies**

```bash
pip install -r requirements.txt
```

### **Step 4: Verify Setup**

```bash
python -c "import streamlit, xgboost, sklearn; print('✓ Ready!')"
```

---

## 🚀 Usage

### **Local Deployment**

```bash
streamlit run app.py
```

App opens at: `http://localhost:8501`

### **Using the Application**

**Step 1: Set Process Variables**
- Adjust 11 sliders in left sidebar
- Values: T1, T4, T5, T6, T7, T13, T14, L, D, F, B
- Default values provided based on training data ranges

**Step 2: Make Prediction**
- Click "🔮 Make Prediction" button
- Wait ~1 second for calculation

**Step 3: View Results**
- **Predicted Purity:** Main prediction (0-1 scale)
- **Target Purity:** Setpoint (default 0.90)
- **Difference:** Deviation from target
- **Status:** Color-coded indicator
  - 🟢 Green (>0.85): Good
  - 🟡 Orange (0.75-0.85): Acceptable
  - 🔴 Red (<0.75): Poor

**Step 4: Explore Insights**
- **Feature Importance:** Which variables matter most
- **Model Info:** Performance metrics & details
- **Help:** Usage instructions & FAQ

### **Cloud Deployment (Streamlit Cloud)**

1. Push to GitHub:
```bash
git add .
git commit -m "Deploy soft-sensor"
git push origin main
```

2. Go to [share.streamlit.io](https://share.streamlit.io/)

3. Deploy:
   - Repo: `YOUR_USERNAME/distillation-soft-sensor`
   - File: `app.py`
   - Click "Deploy"

4. Live at: `https://YOUR_USERNAME-distillation-soft-sensor.streamlit.app`

---

## 📈 Results

### **Model Performance Summary**

```
TEST SET RESULTS (30% of data, 1,322 samples):

┌──────────────────┬─────────────┬──────────────┐
│ Metric           │ Linear Reg  │ XGBoost ✅   │
├──────────────────┼─────────────┼──────────────┤
│ R² Score         │ 0.9859      │ 0.9998       │
│ RMSE             │ 0.0080      │ 0.0010       │
│ MAE              │ 0.0062      │ 0.0008       │
│ Improvement      │ —           │ 87.5% better │
└──────────────────┴─────────────┴──────────────┘
```

### **Top Features by Importance**

```
Rank | Feature      | Importance | Percentage | Physical Role
─────┼──────────────┼────────────┼────────────┼──────────────────────
  1  | T1           | 9.909e-01  | 99.09%     | Column top temperature
  2  | T5           | 2.791e-03  | 0.28%      | Mid-column temperature
  3  | T6           | 2.124e-03  | 0.21%      | Mid-column temperature
  4  | T4_lag1      | 1.091e-03  | 0.11%      | Temp at 6-min lag
  5  | T1_lag5      | 6.952e-04  | 0.07%      | Temp at 30-min lag
  6  | T5_lag1      | 4.933e-04  | 0.05%      | Temp at 6-min lag
  7  | L            | 4.837e-04  | 0.05%      | Reflux rate
  8  | T4           | 3.861e-04  | 0.04%      | Mid-column temperature
  9  | T13          | 1.368e-04  | 0.01%      | Column temperature
 10  | B            | 9.800e-05  | 0.01%      | Bottom product rate
```

**Key Insight:** T1 (column top temperature) is the dominant predictor at 99.09% importance. This is expected for simulated data where the simulator's mathematical model makes temperature the primary driver of purity. All other features combined contribute only 0.91%.

---

## 📁 Project Structure

```
distillation-soft-sensor/
│
├── 📓 NOTEBOOKS (Analysis & Development)
│   ├── 01_eda.ipynb
│   │   ├── Part 1: Setup & Data Loading
│   │   ├── Part 2: EDA (distributions, correlations)
│   │   ├── Part 3: Time-Series Analysis (ACF/PACF, decomposition)
│   │   ├── Part 4: Data Cleaning & Outliers
│   │   ├── Part 5: Feature Engineering (50 → 30 features)
│   │   └── Part 6: Summary & Export
│   │
│   └── 02_ml_modeling.ipynb
│       ├── Part 1: Setup & Data Loading
│       ├── Part 2: Train/Test Split (70/30, chronological)
│       ├── Part 3: Feature Scaling (StandardScaler)
│       ├── Part 4: Baseline Model (Linear Regression)
│       ├── Part 5: XGBoost Training & Evaluation
│       ├── Part 6: Feature Importance Analysis
│       ├── Part 7: Model Comparison & Results
│       └── Part 8: Save Models & Scaler
│
├── 🚀 APPLICATION (Streamlit)
│   ├── app.py                          # Main Streamlit app
│   └── utils.py                        # Helper functions
│       ├── Config (paths, thresholds)
│       ├── load_model(), load_scaler(), load_feature_names()
│       ├── validate_inputs()
│       ├── create_input_dataframe()
│       ├── scale_inputs()
│       ├── predict_purity()
│       ├── get_prediction_status()
│       ├── get_feature_importance()
│       ├── get_model_performance()
│       └── format_purity_display()
│
├── 📦 DATA
│   ├── X_ml_features.csv               # 30 engineered features (reference)
│   ├── y_ml_target.csv                 # Target variable (purity)
│   └── dataset_distill.csv             # Original raw data
│
├── 🤖 MODELS (Pre-trained)
│   ├── xgb_model.pkl                   # XGBoost regressor
│   └── scaler.pkl                      # StandardScaler
│
├── 📚 DOCUMENTATION
│   ├── README.md                       # This file
│   ├── DEPLOYMENT_GUIDE.md             # Detailed setup & deployment
│   ├── QUICK_START.md                  # Quick reference
│   └── IMPLEMENTATION_CHECKLIST.md     # Development tasks
│
└── 📋 CONFIGURATION
    ├── requirements.txt                 # Python dependencies
    ├── .gitignore                      # Git rules
    └── examples.py                     # Example implementations
```

---

## 👥 Contributors

**Project Lead:** Ildebrando de Jesus Rodrigues (ijesusjr)

**Responsibilities:**
- Complete data analysis & EDA (01_eda.ipynb)
- Feature engineering & selection
- Model development & evaluation (02_ml_modeling.ipynb)
- Streamlit application development
- Deployment & documentation

**Technologies:**
- **Python:** 3.8+
- **ML/Data:** XGBoost, scikit-learn, pandas, numpy
- **Visualization:** matplotlib, seaborn, plotly
- **Deployment:** Streamlit
- **Time-Series:** statsmodels

---

## 📚 References

### **Dataset & Credits**

- **Source:** [Kaggle Distillation Column Dataset](https://www.kaggle.com/datasets/jorgecote/distillation-column)
- **Creator:** Jorge Cote
- **License:** Kaggle Dataset License
- **Citation:** Please credit Jorge Cote when using this dataset

### **Machine Learning Papers**

- **XGBoost:** Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD '16
  - [arXiv](https://arxiv.org/abs/1603.02754)

- **Gradient Boosting:** Friedman, J. H. (2001). Greedy Function Approximation: A Gradient Boosting Machine
  - [JSTOR](https://www.jstor.org/stable/2699986)

- **Feature Engineering:** Guyon, I., & Elisseeff, A. (2003). An Introduction to Variable and Feature Selection
  - [JMLR](https://www.jmlr.org/papers/v3/guyon03a.html)

### **Time-Series Analysis**

- **Box, Jenkins, Reinsel** (2015). Time Series Analysis: Forecasting and Control
  - Authoritative reference on ACF/PACF & stationarity
  - [Wiley](https://www.wiley.com/en-us/Time+Series+Analysis:+Forecasting+and+Control,+5th+Edition-p-9781118675778)

- **ADF Test:** Said, S. E., & Dickey, D. A. (1984). Testing for Unit Roots
  - [EconPapers](https://econpapers.repec.org/article/ecmemetrp/v_3a52_3ay_3a1984_3ai_3a3_3ap_3a1057-72.htm)

### **Soft-Sensors in Chemical Engineering**

- **Fortuna et al.** (2006). Soft Sensors for Product Quality Monitoring and Control
  - [Springer](https://www.springer.com/gp/book/9781848003438)

- **Kadlec, Gabrys, Strandt** (2009). Data-Driven Soft Sensors in the Process Industry
  - [Computers & Chemical Engineering](https://www.sciencedirect.com/science/article/abs/pii/S002526880900025X)

### **Tools & Libraries**

- **Streamlit:** [Docs](https://docs.streamlit.io/)
- **XGBoost:** [Docs](https://xgboost.readthedocs.io/)
- **scikit-learn:** [Docs](https://scikit-learn.org/)
- **Pandas:** [Docs](https://pandas.pydata.org/docs/)
- **NumPy:** [Docs](https://numpy.org/doc/)
- **statsmodels:** [Docs](https://www.statsmodels.org/)

---

## 📊 Key Findings Summary

1. **Strong Temperature Dependence:** T1 (column top temperature) accounts for 99.09% of model predictions, reflecting that temperature is the primary driver of purity in distillation.

2. **Simulated Data Characteristics:** The extremely high importance of a single feature and near-perfect R² (0.9998) are expected for simulated data with deterministic relationships.

3. **Seasonality Confirmed:** ACF analysis revealed strong 24-hour cycle patterns (240 timesteps), which XGBoost leverages through cyclic time encoding.

4. **Appropriate Feature Engineering:** Lagged features (1, 5, 10, 30, 60 timesteps) capture temporal dependencies identified through ACF analysis, though their individual impact is minimal in simulated data.

5. **Production Readiness Caveat:** Model architecture is production-ready, but real plant deployment would require:
   - Validation on industrial data
   - Expectation of lower R² (~0.85-0.90)
   - More distributed feature importance
   - Periodic retraining cycles

---

## 🔗 Links

- **GitHub:** [your-repo-link]
- **Streamlit App:** [your-app-link - Coming Soon]
- **Dataset Source:** https://www.kaggle.com/datasets/jorgecote/distillation-column
- **LinkedIn:** [optional]

---

## 📄 Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | December 2024 | Initial release with XGBoost model, Streamlit app, complete documentation |
| 1.1 | TBD | Real plant data validation, model refinement |
| 2.0 | TBD | Production deployment, monitoring system |

---

**Last Updated:** December 2024  
**Status:** ✅ Production Ready  
**Model:** XGBoost with R² = 0.9998 (simulated data)

---

*Built with ❤️ using machine learning for chemical engineering*

*Dataset credits: Jorge Cote on Kaggle*

*Project developed as part of Le Wagon Data Science & AI Bootcamp*
