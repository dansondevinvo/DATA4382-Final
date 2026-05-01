# Images

This folder contains visualizations generated during model development.

## Contents

| File | Description |
|------|-------------|
| `shap_global_importance.png` | SHAP bar chart — mean absolute SHAP values for all 29 features |
| `shap_beeswarm.png` | SHAP beeswarm plot — feature value direction and magnitude across all test cases |
| `shap_local_fn.png` | SHAP waterfall plot — false negative case study explaining a missed injury |
| `confusion_matrix.png` | Confusion matrix — XGBoost at threshold 0.40 |
| `threshold_tradeoff.png` | Accuracy vs. recall across thresholds 0.30 – 0.60 |
| `fairness_municipality.png` | Injury recall by municipality |
| `fairness_light.png` | Injury recall by light condition |

## How to Regenerate

All plots are generated automatically when running `notebooks/My_Deployment.ipynb`.  
SHAP plots require the trained model (`model.pkl`) and test set data to be in memory.
