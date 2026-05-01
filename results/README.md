# Results

## Final Model Performance — XGBoost (threshold = 0.40)

### Confusion Matrix

|  | Predicted: No Injury | Predicted: Injury |
|--|---------------------|------------------|
| **Actual: No Injury** | 15,078 ✅ | 17,959 ⚠️ |
| **Actual: Injury** | 671 ❌ | 6,522 ✅ |

### Core Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Injury Recall** | **91%** | Catches 9 in 10 real injury crashes |
| Overall Accuracy | 53% | Intentional tradeoff for safety |
| Precision | 26% | Acceptable false alarm rate |
| False Negatives | 671 | Missed injury crashes (minimize) |
| False Positives | 17,959 | Non-injury crashes flagged (acceptable) |

## Model Comparison

| Model | Accuracy | Injury Recall | Notes |
|-------|----------|--------------|-------|
| Logistic Regression (Baseline) | 80% | 30% | Performance floor |
| Random Forest | 76% | 44% | Weak on minority class |
| LightGBM | 81% | 27% | Best accuracy, sacrifices recall |
| Stacking Ensemble (XGB+LGBM+RF) | 80% | 29% | No improvement |
| **XGBoost @ 0.40 ★** | **53%** | **91%** | **Final model** |

## Threshold Analysis

| Threshold | Accuracy | Recall | Precision |
|-----------|----------|--------|-----------|
| 0.30 | 46.9% | 94.6% | 24.5% |
| 0.35 | 50.0% | 92.9% | 25.4% |
| **0.40 ★** | **53.0%** | **91.0%** | **26.4%** |
| 0.45 | 55.7% | 88.8% | 27.3% |
| 0.50 | 58.4% | 86.2% | 28.3% |
| 0.60 | 64.6% | 77.8% | 30.7% |

## Fairness Results

| Subgroup | Recall Range | Status |
|----------|-------------|--------|
| Weather | 0.714 – 1.000 | WATCH |
| Municipality | 0.792 – 0.936 | WATCH |
| Day vs. Night | 0.874 – 0.911 | ✅ PASS |
| Weekday vs. Weekend | 0.874 – 0.916 | ✅ PASS |

No group fell below 0.70 recall threshold.

## Top SHAP Features (Global)

1. Vehicle Damage Extent
2. Driver At Fault
3. IsWeekend
4. Vehicle First Impact Location
5. Vehicle Going Direction
6. Speed Limit
7. DayOfWeek
8. Vehicle Body Type
9. Collision Type
10. Hour
