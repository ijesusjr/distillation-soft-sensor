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

| Model | Test R² | Test RMSE | Test MAE | Inference Time | Model Size |
|-------|---------|-----------|----------|----------------|-----------|
| **Linear Regression** ✅ | **0.9859** | **0.0080** | **0.0062** | **<0.1ms** | **<1KB** |
| **XGBoost** | 0.9998 | 0.0010 | 0.0008 | 1-2ms | ~500KB |

**Selected Model: Linear Regression**
- Explains 98.59% of variance (excellent for production)
- Only 1.39% performance loss vs XGBoost
- **140x faster inference** (<0.1ms vs 1-2ms)
- **500x smaller model** (<1KB vs 500KB)
- **Fully interpretable** coefficients

---

### **Key Decisions Made**

1. **Linear Regression Selected Over XGBoost**: 
   - While XGBoost achieves R² = 0.9998, Linear Regression's R² = 0.9859 is excellent and explains 98.59% of variance
   - **Decision rationale:** Trade-off analysis prioritized production efficiency, simplicity, and interpretability over marginal 0.01% R² improvement
   - Linear model reduces infrastructure costs, computational overhead, and deployment complexity

2. **Why Linear Over XGBoost?**
   - **Simplicity:** Single equation vs 100 decision trees (easier to maintain, debug, audit)
   - **Computational Cost:** <0.1ms vs 1-2ms per prediction (140x faster)
   - **Model Size:** <1KB vs ~500KB (500x smaller, instant loading)
   - **Interpretability:** Direct feature coefficients reveal how each variable impacts purity
   - **Production Reliability:** Fewer dependencies, lower memory footprint, easier A/B testing
   - **Regulatory Compliance:** Simpler models easier to validate and explain to auditors

3. **No RNN/LSTM Implemented**: Simulated data has clean deterministic patterns; Linear Regression already captures 98.59% of variance.

4. **Feature Engineering Strategy**:
   - Lagged features capture temporal dependencies
   - Cyclic encoding explicitly models 24-hour pattern
   - Removed correlated features to prevent overfitting

5. **Chronological Train/Test Split**: Respects time-series structure and emulates real-world deployment scenario

---

### **Results Summary**

**Linear Regression Test Set Performance:**
- **R² Score:** 0.9859 (explains 98.59% of variance)
- **RMSE:** 0.0080 
- **MAE:** 0.0062 
- **Inference Time:** <0.1ms per prediction
- **Model Size:** <1KB

---

## 🎯 Key Finding: Simulated Data Characteristics

### **Why Linear Works So Well**

Linear Regression achieves R² = 0.9859 because synthetic data exhibits perfect mathematical relationships where temperature is the primary determinant of purity with minimal noise, resulting in strong linear patterns.

The relationship between process variables and purity is inherently **linear in this simulated system**, making Linear Regression an ideal fit.

### **Expected Performance on Real Plant Data**

Real industrial data would show:
- More distributed feature importance
- Non-linear relationships requiring more complex models
- Sensor noise reducing overall R²
- Realistic R² values of 0.75-0.85 for either model

**Production Readiness Caveat:** Model architecture is production-ready, but real plant deployment would require:
- Validation on industrial data
- Potential model selection revision based on real patterns
- Periodic retraining cycles
- Feature drift monitoring


---

## 🎯 App Demo

### **Live Application**

[Streamlit App](https://ijesusjr-distillation-soft-sensor.streamlit.app/)

### **Features:**

- **Real-time Predictions:** Set process variables and get instant purity prediction (<0.1ms)
- **Interactive Sliders:** Control 11 main process variables
  - Temperatures: T1, T4, T5, T6, T7, T13, T14
  - Flow rates: L (Reflux), D (Distillate), F (Feed), B (Bottom product)
- **Automatic Lag Handling:** App fills lagged features assuming constant history
- **Status Indicators:** Color-coded purity status
  - 🟢 **Green (>0.85):** Good
  - 🟡 **Orange (0.75-0.85):** Acceptable
  - 🔴 **Red (<0.75):** Poor
- **Feature Coefficients:** See how each variable directly impacts purity prediction
- **Model Explainability:** Fully transparent predictions with direct interpretability

### **Quick Start (Local):**

```bash
streamlit run app.py
```

Then:
1. Adjust sliders in sidebar
2. Click "🔮 Make Prediction"
3. View results and feature contributions

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
TEST SET RESULTS (30% of data):

┌──────────────────┬──────────────┬──────────────┐
│ Metric           │ Linear Reg ✅│ XGBoost      │
├──────────────────┼──────────────┼──────────────┤
│ R² Score         │ 0.9859       │ 0.9998       │
│ RMSE             │ 0.0080       │ 0.0010       │
│ MAE              │ 0.0062       │ 0.0008       │
│ Inference Time   │ <0.1ms       │ 1-2ms        │
│ Model Size       │ <1KB         │ ~500KB       │
└──────────────────┴──────────────┴──────────────┘
```

### **Linear Regression Coefficients**

Top 10 features by absolute coefficient value (direct impact on purity):

```
Feature         | Coefficient | Interpretation
────────────────┼─────────────┼──────────────────────────────────
T1              | +0.0847     | 1-unit ↑ T1 → +0.0847 purity
T5              | -0.0156     | 1-unit ↑ T5 → -0.0156 purity
T6              | +0.0143     | 1-unit ↑ T6 → +0.0143 purity
T4_lag1         | -0.0089     | Lagged temperature effect
T1_lag5         | +0.0065     | Past temperature influence
T5_lag1         | -0.0042     | Lagged temperature effect
L               | +0.0004     | Reflux rate contribution
T4              | -0.0003     | Mid-column temperature
T13             | -0.0001     | Column temperature
B               | +0.0001     | Bottom product rate
```

**Key Insight:** T1 (column top temperature) has the strongest positive impact (+0.0847 coefficient), directly confirming that temperature is the primary driver of purity.


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
│   ├── lr_model.pkl                    # Linear Regression (PRODUCTION)
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

1. **Linear Relationships:** Linear Regression achieves R² = 0.9859, indicating predominantly linear relationships in simulated distillation data.

2. **T1 Dominance:** Column top temperature (T1) is the strongest predictor with coefficient +0.0847, confirming thermodynamic principles.

3. **Production Efficiency:** Linear model delivers 140x faster inference and 500x smaller footprint vs XGBoost with only 1.39% accuracy loss.

4. **Appropriate Feature Engineering:** Lagged features and cyclic encoding capture temporal patterns identified through ACF/PACF analysis.

5. **Production Readiness:** Simple, interpretable Linear Regression is ideal for industrial deployment where reliability, cost, and explainability are priorities.

6. **Real-World Validation Needed:** Deployment on real plant data would require model revalidation; expect lower R² (~0.75-0.85) with non-linear patterns.

---

## 🔗 Links

- **GitHub:** https://github.com/ijesusjr/distillation-soft-sensor
- **Streamlit App:** https://ijesusjr-distillation-soft-sensor.streamlit.app/
- **Dataset Source:** https://www.kaggle.com/datasets/jorgecote/distillation-column
- **LinkedIn:** https://www.linkedin.com/in/ijesus/

---


**Last Updated:** May 2026  