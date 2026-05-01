# Models

After running **Cell 2** of `notebooks/My_Deployment.ipynb`, the following files are generated:

| File | Description |
|------|-------------|
| `model.pkl` | Trained XGBoost model (serialized with joblib) |
| `pipeline_vars.pkl` | Label encoder mappings, feature names, and decision threshold (0.40) |

These files are required by **Cell 3** to launch the Gradio app.

## Model Summary

**Final Model:** XGBoost Classifier  
**Decision Threshold:** 0.40 (tuned for maximum injury recall)

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

## Performance

| Metric | Value |
|--------|-------|
| Injury Recall | **91%** |
| Overall Accuracy | 53% |
| Precision | 26% |
| Test Set Size | 40,230 records |
| True Positives (injuries caught) | 6,522 |
| False Negatives (injuries missed) | 671 |
