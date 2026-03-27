# Distillation Column Soft-Sensor: ML-Based Purity Prediction

---

## 📋 Overview

### **Problem Statement**

Distillation columns are critical unit operations in the chemical and petroleum industries for separating mixtures based on boiling points. Real-time monitoring of **product purity** is essential for:
- Process control and optimization
- Product quality assurance  
- Cost reduction through efficient operation

Traditional measurement methods (lab analysis) are:
- **Time-consuming** (results may take hours)
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

**Type:** Synthetic distillation column data generated from a mathematical model. Industrial considerations atypical data features like noise, outliers, and missing data have been added, in order to simulate industrial conditions in the dataset.

---

### **Analysis Conducted**

#### **Phase 1: Exploratory Data Analysis (EDA)** → `01_eda.ipynb`
**Key Steps:**
1. Time-Series Decomposition
2. Autocorrelation (ACF/PACF) Analysis
3. Stationarity Testing (ADF Test)
4. Feature Correlation Analysis

#### **Phase 2: Feature Engineering** → `01_eda.ipynb`
**Key Steps:**
1. Lagged Features
2. Rolling Statistics
3. Cyclic Time Encoding
4. Feature Selection:
    - Started with ~50 engineered features
    - Dropped highly correlated features (r > 0.95)

#### **Phase 3: Model Development** → `02_ml_modeling.ipynb`

**Data Splitting:**
- Train: 80% / Test: 20% 
- Chronological split, preserving time-series structure and preventing leakage

**Models Trained & Evaluated:**

| Model | Test R² | Test RMSE | Test MAE |
|-------|---------|-----------|----------|
| **Linear Regression** | 0.9859 | 0.0080 | 0.0062 |
| **XGBoost** ✅ | **0.9998** | **0.0010** | **0.0008** |

**Winner: XGBoost**
- **87.5% improvement in RMSE over Linear Regression** (0.008 → 0.001)
- Captures non-linear threshold effects and feature interactions and models complex distillation dynamics

---

### **Key Decisions Made**

1. **XGBoost Selected Over Linear Regression**: both models with excellent R2, however the 87.5% RMSE improvement in XGBoost justifies added complexity.

2. **No RNN/LSTM Implemented**: once simulated data has clean and deterministic patterns, XGBoost already achieves near-perfect accuracy and RNN would add complexity without meaningful gain.

3. **Feature Engineering Strategy**:
   - Lagged features capture temporal dependencies and can better represent what happens in reality
   - Cyclic encoding explicitly models 24-hour pattern
   - Removed correlated features to prevent overfitting

4. **Chronological Train/Test Split**: respects time-series structure and emulates real-world deployment scenario

---

### **Results Summary**

**XGBoost Test Set Performance:**
- **R² Score:** 0.9998 (explains 99.98% of variance)
- **RMSE:** 0.0010 
- **MAE:** 0.0008 
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

1. **Simulated Data Characteristics:** synthetic data shows perfect mathematical relationships where temperature is the primary determinant of purity with minimal noise, resulting in deterministic patterns that lead to feature dominance.

2. **Physical Justification:** in real distillation, temperature directly drives separation as the strongest control variable, which aligns with thermodynamic principles and is accurately reflected in the simulation.

3. **Expected Behavior in Production:** real plant data would show more distributed feature importance due to unmeasured disturbances and sensor noise, resulting in realistic R² values of 0.80-0.90 rather than 0.9998.

4. **Model Quality Assessment:** R² = 0.9998 is excellent and appropriate for simulated data, where single dominant features are normal and lagged features provide incremental value while the model successfully identifies the true driving variable.

### **Implications for Deployment**

- Model works perfectly on simulated data (as expected)
- Real plant validation is critical before production use
- On real industrial data, expect:
  - More distributed feature importance 
  - Lower R² 
  - Greater value from lagged/rolling features
  - Need for periodic model retraining with real data


---

## 🎯 App Demo

### **Live Application**

[\[Streamlit Cloud URL - Deploy Instructions Below\] ](https://ijesusjr-distillation-soft-sensor.streamlit.app/)

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

- Pressure of column 
- Temperature at each tray 
- Liquid flowrate 
- Vapor flowrate 
- Distillate flowrate
- Bottoms flowrate 
- Feed flowrate
- Molar concentration of ethanol 

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

---

## 📈 Results

### **Model Performance Summary**

```
TEST SET RESULTS (20% of data):

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
│   └── 02_ml_modeling.ipynb
│
├── 🚀 APPLICATION (Streamlit)
│   ├── app.py                          # Main Streamlit app
│   └── utils.py                        # Helper functions
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
├── README.md                       # This file
│
└── requirements.txt                 # Python dependencies
```

---

## 👥 Contributors

**Project Lead:** Ildebrando de Jesus Junior (ijesusjr)

**Responsibilities:**
- Complete data analysis & EDA (01_eda.ipynb)
- Feature engineering & selection (01_eda.ipynb)
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

---

## 📊 Key Findings Summary

1. **Strong Temperature Dependence:** T1 (column top temperature) accounts for 99.09% of model predictions, reflecting that temperature is the primary driver of purity in distillation.

2. **Simulated Data Characteristics:** The extremely high importance of a single feature and near-perfect R² (0.9998) are expected for simulated data with deterministic relationships.

3. **Seasonality Confirmed:** ACF analysis revealed strong 24-hour cycle patterns (240 timesteps), which XGBoost leverages through cyclic time encoding.

4. **Appropriate Feature Engineering:** Lagged features (1, 5, 10, 30, 60 timesteps) capture temporal dependencies identified through ACF analysis, though their individual impact is minimal in simulated data.

5. **Production Readiness Caveat:** Model architecture is production-ready, but real plant deployment would require:
   - Validation on industrial data
   - Expectation of lower R² (~0.80-0.90)
   - More distributed feature importance
   - Periodic retraining cycles

---

## 🔗 Links

- **GitHub:** https://github.com/ijesusjr/distillation-soft-sensor
- **Streamlit App:** https://ijesusjr-distillation-soft-sensor.streamlit.app/
- **Dataset Source:** https://www.kaggle.com/datasets/jorgecote/distillation-column
- **LinkedIn:** https://www.linkedin.com/in/ijesus/

---


**Last Updated:** March 2026  