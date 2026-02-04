
# HACCP / Food Safety — Noncompliance Risk Modeling (Logistic Regression)

**Timeframe:** Late 2023 – Late 2025  
**Context:** Wafer & cream-filled line (flow-pack + carton + boxing)

---

## Background (Why this matters)
HACCP is designed to **prevent food safety hazards before they reach consumers**. In real plants, QA teams often collect monitoring logs (area, shift, environmental conditions, follow-up swabs). A logistic regression model can help **quantify the probability of noncompliance**, prioritize high-risk zones, and support **targeted sanitation / verification plans** to reduce microbial contamination risk.

---

## Key results (from the fitted model)
- **Model type:** Logistic regression (binary outcome: `Noncompliance = 1`)
- **AUC (ROC):** **0.50** (train/test split, 70/30)
- **Top risk drivers (directional):**
  - Stronger positive association with risk (examples): **Sampling_Reason_Post-cleaning verification**, **Area_Filling Nozzles**
  - Stronger protective association (examples): **Sampling_Reason_Pre-shift check**, **Area_Cooling Tunnel**
- **Highest-risk areas (avg predicted):** typically **Filling Nozzles** and **Packaging Room**  
- **Environmental effect:** **Higher humidity** tends to increase predicted risk (continuous predictor)

> Note: This is a learning-focused anonymized dataset structured like a QA monitoring log.

---

## Outputs (plots)
### ROC curve
![ROC Curve](haccp_roc_curve.png)

### Average risk by area
![Risk by Area](haccp_risk_by_area.png)

---

## Model details (variables)
- **Continuous predictors:** `Humidity_pct`, `Temp_C`
- **Categorical predictors:** `Area`, `Shift`, `Sampling_Reason`, `Product`
- **Interpretation:** odds ratios (>1 increases risk, <1 decreases risk) help identify where controls should be strengthened.

---

## How to run
```bash
Rscript HACCP_Compliance_LogisticRegression_Niusha.R
```

---

## Files
- `haccp_compliance_regression_wafercream_2023_2024.csv` — regression-ready dataset  
- `HACCP_Compliance_LogisticRegression_Niusha.R` — R script (model + ROC + plots)  
- `haccp_roc_curve.png` — ROC output  
- `haccp_risk_by_area.png` — risk-by-area output  

---

## Tags
`r` · `logistic-regression` · `food-safety` · `haccp` · `data-analysis` · `predictive-modeling`


