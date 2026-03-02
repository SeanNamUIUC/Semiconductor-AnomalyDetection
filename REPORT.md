# Semiconductor Anomaly Detection – EDA Report
Developed as a structured ML engineering project with reproducible analysis and model-oriented reasoning.

---

# 1. Dataset Overview

## 1.1 Structure

The dataset consists of **1567 samples** and **593 columns**.

| Category | Count |
|----------|-------|
| Sensor Features | 590 |
| Metadata Columns | 2 (`sample_id`, `timestamp`) |
| Target Label | 1 |

All sensor features are numeric (`float64`).  
The label is categorical with values {-1 (Normal), 1 (Defective)}.

Time range:
**2008-07-19 11:55:00 ~ 2008-10-17 06:07:00**

---

# 2. Role-Based Feature Classification

| Type | Columns | Used for Modeling? |
|------|---------|-------------------|
| Sensor Features | `sensor_001` ~ `sensor_590` | ✅ Yes |
| Label | `label` | ❌ Target only |
| Metadata | `sample_id`, `timestamp` | ⚠️ Not directly |

- `sample_id` is an identifier and excluded from modeling.
- `timestamp` may be used for temporal feature engineering but is not directly included.

---

# 3. Class Imbalance Analysis

Label distribution:

| Class | Count | Percentage |
|-------|-------|------------|
| Normal (-1) | 1463 | 93.36% |
| Defective (1) | 104 | 6.64% |

This dataset is **highly imbalanced**.

## 3.1 Baseline Accuracy

If all samples are predicted as normal:

Baseline Accuracy ≈ **93.4%**

> Accuracy alone is misleading due to severe class imbalance.

Therefore, evaluation metrics such as **Recall, F1-score, and Precision-Recall tradeoff** will be prioritized.

---

# 4. High-Dimensional Risk Assessment

- Number of samples: 1567  
- Number of input features: 590  
- Samples per feature: **2.66**

Although the number of features is smaller than the number of samples (p < n),
the low samples-per-feature ratio indicates a **moderate-to-high risk of overfitting**.

## 4.1 Risk Factors

- High number of sensor features
- Low samples per feature
- Class imbalance
- Potential noisy sensors

---

# 5. Modeling Implications

Based on the EDA findings:

1. Accuracy will not be used as the primary evaluation metric.
2. Recall and F1-score will be emphasized.
3. Regularization techniques will be applied.
4. Feature reduction strategies will be explored.
5. Cross-validation will be implemented to mitigate overfitting risk.

---

# 6. Next Steps

- Missing value strategy
- Low variance feature removal
- Correlation analysis
- Baseline modeling (Logistic Regression)
- Tree-based models comparison

---

# 7. Limitations

- Dataset spans only ~3 months
- Severe class imbalance
- Potential sensor redundancy

Further experimentation is required for robust deployment readiness.
