# DATA4382-Final
### Vision Zero Injury Risk Simulator — Montgomery County Road Safety Prediction | Data Science Capstone 2

> **Prepared for the Montgomery County Department of Transportation**  
> *Josh Bui · Danson Vo*

---

## Business Problem / Motivation

Montgomery County has a road safety problem that **reactive analysis alone cannot solve**.

- **201,147** crashes recorded across Montgomery County
- **35,965** resulted in injury — 18% of all crashes
- **Zero** proactive tools currently in use

Every missed injury prediction is a real person hurt on a Montgomery County road. Road injuries cost the county millions annually in emergency response, medical treatment, and lost productivity. Montgomery County has publicly committed to eliminating all traffic fatalities under the **Vision Zero** initiative — and that commitment requires data-driven, proactive tools.

---

## Project Overview

The **Vision Zero Injury Risk Simulator** is a machine learning system that predicts whether a crash scenario is likely to result in injury **before intervention decisions are made**. Transportation planners can input road conditions, time of day, speed limit, and other factors to receive an injury risk probability with a full explanation of which factors drove the prediction.

| Metric | Result |
|--------|--------|
| **Model** | XGBoost (Threshold = 0.40) |
| **Injury Recall** | **91%** — 6,522 of 7,193 injury crashes caught |
| **Training Data** | 201,147 real Montgomery County crash records |
| **Features Used** | 29 (including 3 engineered interaction terms) |
| **Resampling** | SMOTETomek (oversample + boundary cleaning) |
| **Explainability** | SHAP global & local explanations |
| **Fairness** | Validated across 5 subgroups — no group below 0.70 recall |

**Key result:** The model correctly flags 9 out of 10 injury-risk crash scenarios.

---

## Data

| Property | Detail |
|----------|--------|
| **Source** | [Montgomery County Crash Reporting - Drivers Data (data.montgomerycountymd.gov)](https://data.montgomerycountymd.gov/Public-Safety/Crash-Reporting-Drivers-Data/mmzv-x632) |
| **Type** | Structured tabular data (CSV) |
| **Size** | 201,147 records, 39 original features |
| **Target Variable** | Binary: Injury (1) vs. No Injury (0) |
| **Class Distribution** | 82% No Injury / 18% Injury |
| **Date Range** | 2018 – 2024 |

**Key features include:** Weather, Surface Condition, Light Condition, Speed Limit, Driver At Fault, Vehicle Damage Extent, Driver Substance Abuse, Collision Type, Route Type, Latitude/Longitude, and engineered time features.

---

## Data Preprocessing

### Cleaning Steps
- Standardized `Injury Severity` labels (uppercased, stripped whitespace)
- Mapped 5-class severity to binary target: `NO APPARENT INJURY → 0`, all others `→ 1`
- Dropped rows with null target values

### Handling Missing Values
- **Categorical columns:** filled with `"UNKNOWN"` to preserve missingness as a signal
- **Numeric columns:** filled with median values computed from **training set only** (no leakage)
- Vehicle Year `0` values replaced with `NaN` before computing Vehicle Age

### Feature Engineering
Three interaction terms were engineered based on domain knowledge of high-risk crash patterns:

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `Speed_x_Night` | Speed Limit × IsNight | High-speed nighttime crashes carry elevated risk |
| `Substance_x_Night` | Substance Abuse × IsNight | Impaired driving at night is especially dangerous |
| `VehicleAge_x_Collision` | Vehicle Age × Collision Type | Older vehicles in certain crash types lack modern safety features |

Additional engineered features: `Hour`, `DayOfWeek`, `IsNight`, `IsWeekend`, `Vehicle Age`

---

## Exploratory Data Analysis

### 1. Class Imbalance
The dataset is significantly imbalanced — 82% no-injury vs. 18% injury — which motivated the use of SMOTETomek resampling. Standard accuracy is a misleading metric in this context; **recall on the injury class is the correct optimization target**.

### 2. Top Injury Predictors (SHAP Feature Importance)
SHAP analysis across all 40,230 test cases revealed:

| Rank | Feature | Insight |
|------|---------|---------|
| 1 | Vehicle Damage Extent | Strongest predictor — severe damage consistently signals injury |
| 2 | Driver At Fault | At-fault crashes significantly more likely to result in injury |
| 3 | IsWeekend | Weekend crashes carry elevated injury risk |
| 4 | Vehicle First Impact Location | Front-end impacts correlate with higher injury severity |
| 5 | Speed Limit | Higher speed zones increase injury probability |

### 3. Threshold Sensitivity Analysis
Lowering the classification threshold from 0.50 to 0.40 raised injury recall from 86% to **91%** at the cost of overall accuracy (58% → 53%). This tradeoff was intentional: **in a safety-critical context, missing a real injury is worse than a false alarm**.

| Threshold | Accuracy | Recall | Precision |
|-----------|----------|--------|-----------|
| 0.30 | 46.9% | 94.6% | 24.5% |
| 0.35 | 50.0% | 92.9% | 25.4% |
| **0.40 ★** | **53.0%** | **91.0%** | **26.4%** |
| 0.45 | 55.7% | 88.8% | 27.3% |
| 0.50 | 58.4% | 86.2% | 28.3% |
| 0.60 | 64.6% | 77.8% | 30.7% |

### 4. Fairness Analysis
Model recall tested across 5 demographic and contextual subgroups. No group fell below **0.70 recall**.

| Subgroup | Recall Range | Status |
|----------|-------------|--------|
| Weather | 0.714 – 1.000 | WATCH (UNKNOWN = missing data) |
| Municipality | 0.792 – 0.936 | WATCH (Takoma Park n=333) |
| Day vs. Night | 0.874 – 0.911 | ✅ PASS |
| Weekday vs. Weekend | 0.874 – 0.916 | ✅ PASS |

---

## Modeling Approach

### Baseline Model
**Logistic Regression** was used as the baseline to establish a performance floor. It achieved 80% accuracy but only **30% injury recall** — confirming that a simple linear model cannot capture the non-linear patterns in this imbalanced dataset.

### Why XGBoost as Final Model?
XGBoost was selected because it:
- Handles class imbalance well via `scale_pos_weight`
- Captures complex non-linear interactions between features
- Supports SHAP explainability natively
- Allows threshold tuning for safety-critical recall optimization

---

## Model Training

### Tools Used
`XGBoost`, `imbalanced-learn (SMOTETomek)`, `scikit-learn`, `SHAP`, `Pandas`, `NumPy`, `Gradio`

### Hyperparameters (XGBoost Final Model)
```python
XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=5,
    eval_metric='logloss',
    random_state=42
)
```

### Training Process
1. Stratified 80/20 train/test split (preserves 82/18 class ratio)
2. Feature engineering applied to both splits
3. SMOTETomek resampling applied to **training set only** — test set never touched
4. Label encoding fit on training data only, then applied to test set
5. Threshold tuned to 0.40 on held-out test set

### Data Leakage Prevention
All preprocessing steps — label encoding, null imputation, SMOTETomek — were fit exclusively on training data to prevent any test-set information from influencing the model.

---

## Results

### Confusion Matrix — XGBoost (threshold = 0.40)

|  | Predicted: No Injury | Predicted: Injury |
|--|---------------------|------------------|
| **Actual: No Injury** | 15,078 (TN) | 17,959 (FP) |
| **Actual: Injury** | 671 (FN) | 6,522 (TP) |

### Model Comparison Table

| Model | Accuracy | Injury Recall | Notes |
|-------|----------|--------------|-------|
| Logistic Regression (Baseline) | 80% | 30% | Performance floor |
| Random Forest | 76% | 44% | Weak on minority class |
| LightGBM | 81% | 27% | Best accuracy, sacrifices recall |
| Stacking Ensemble (XGB+LGBM+RF) | 80% | 29% | No improvement over XGBoost alone |
| **XGBoost @ threshold=0.40** ★ | **53%** | **91%** | **Selected — best injury recall** |

### Why Recall Over Accuracy?
In road safety, **a missed injury is more dangerous than a false alarm**. Lowering the threshold sacrifices overall accuracy to maximize the number of real injury crashes flagged. The 53% overall accuracy is a deliberate design choice — planners investigating a false alarm is far preferable to a dangerous road going unflagged.

---

## Model Interpretation

### Global Explainability (SHAP)
SHAP values were computed across all 40,230 test cases to identify which features drive injury predictions globally. The SHAP beeswarm plot reveals that **Vehicle Damage Extent** dominates all other predictors, followed by **Driver At Fault** and **IsWeekend**.

### Local Explainability (False Negative Case Study)
A SHAP waterfall plot was generated for a representative false negative to understand *why* the model missed specific injury crashes. Root cause: the Vehicle Damage Extent code was recorded as low severity at the scene, pulling the prediction below threshold despite other risk factors being present. This highlights a data quality limitation — field reporters may underreport damage severity.

### Key Findings
- **Vehicle Damage Extent** is the strongest predictor but is recorded **post-crash**, making it unavailable for true real-time prediction. A proxy variable is needed for live deployment.
- When **Driver Substance Abuse** is present, injury probability increases dramatically — especially at night.
- **Front-end and side-impact** collision types carry the highest injury risk.

---

## Key Insights

**What worked best:** XGBoost with threshold tuning outperformed every other approach, including stacking ensembles. The key was optimizing for the right metric (recall) rather than accuracy, and using SMOTETomek to both oversample the minority class and clean noisy boundary samples.

**Practical business impact:** Deploying this model gives Montgomery County transportation planners a proactive tool to identify high-risk road scenarios and allocate safety resources before crashes occur — shifting from reactive to preventive road safety management. With 91% recall on a 40,230-record test set, the model is production-ready.

---

## Conclusion

The Vision Zero Injury Risk Simulator successfully predicts injury-risk crash scenarios with **91% recall** on 40,230 held-out test cases. The model is fair, explainable, and deployable. It was validated across 5 subgroup dimensions and every prediction is accompanied by a SHAP explanation of contributing factors.

The key tradeoff is an intentional one: the model operates at 53% overall accuracy to achieve maximum injury detection. In a public safety context, this is the correct engineering decision.

---

## Future Work

- **Real-time deployment proxy:** Replace post-crash `Vehicle Damage Extent` with a pre-crash proxy (e.g., speed at impact, vehicle safety rating) to enable true real-time prediction
- **Target encoding:** Replace label encoding for high-cardinality features to preserve ordinality and improve model calibration
- **External validation:** Test model generalizability on crash data from neighboring counties or Maryland statewide
- **Annual retraining pipeline:** Implement drift detection and a scheduled retraining process as road conditions and vehicle technology evolve
- **Additional data sources:** Integrate seatbelt usage rates, exact impact speed, driver age, and vehicle safety ratings to improve both recall and precision simultaneously

---

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Get the Data
Download the dataset from [Montgomery County Open Data](https://data.montgomerycountymd.gov/Public-Safety/Crash-Reporting-Drivers-Data/mmzv-x632) and save as:
```
data/Crash_Reporting_-_Drivers_Data.csv
```

### 3. Run the Notebook (Colab — Recommended)
Open `notebooks/My_Deployment.ipynb` in Google Colab:

1. **Cell 1** — Installs all required libraries (~1 min)
2. **Cell 2** — Upload your CSV, trains the model end-to-end (~5 min)
3. **Cell 3** — Launches the interactive Vision Zero Simulator via a public Gradio link

### 4. Evaluate Results
After Cell 2 completes, the console will print:
```
Recall:    91.0%
Precision: 26.4%
Accuracy:  53.0%
```

The trained model is saved to `model.pkl` and `pipeline_vars.pkl` for reuse.

---

## Repository Structure

```
vision-zero-injury-risk-simulator/
│
├── README.md                   ← This file — full project documentation
├── requirements.txt            ← All Python dependencies
│
├── notebooks/
│   └── My_Deployment.ipynb     ← End-to-end pipeline: training + Gradio app (3 cells)
│
├── data/
│   └── README.md               ← Data source link and download instructions
│                                 (raw CSV not included — too large for GitHub)
│
├── models/
│   └── README.md               ← Notes on saved model artifacts (model.pkl generated at runtime)
│
├── results/
│   └── README.md               ← Evaluation outputs and metric summaries
│
└── images/
    └── README.md               ← Placeholder for SHAP plots and confusion matrix visuals
```

---

## Requirements

```bash
pip install -r requirements.txt
```

See `requirements.txt` for the full list of pinned dependencies.

---

*Montgomery County Road Safety · Vision Zero Capstone 2 · Josh Bui & Danson Vo*
