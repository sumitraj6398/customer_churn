# 🔄 Customer Churn Predictor + Monitoring Dashboard

Predicts telecom customer churn using XGBoost, explains predictions with SHAP, 
and monitors model health over time.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-red)](https://khushirajput123-customer-churn-predictor-app-7gg8da.streamlit.app/)
[![GitHub](https://img.shields.io/badge/GitHub-Code-black)](https://github.com/Khushirajput123/customer-churn-predictor)

---

## Tech Stack
Python, XGBoost, SHAP, Scikit-learn, Pandas, Streamlit, SQLite

## Dataset
IBM Telco Customer Churn — 7,043 customers, 21 features, 26.5% churn rate

## Model Results

| Model | Churn Recall | ROC-AUC |
|---|---|---|
| Logistic Regression | 0.70 | 0.824 |
| **XGBoost (tuned)** | **0.81** | **0.846** |

Optimized for **recall and ROC-AUC** over accuracy — missing a churner costs 
more than a false alarm.

## Key Findings
- Month-to-month contracts churn at 42% vs 3% for two-year contracts
- Churn risk highest in first 12 months of tenure
- Customers without tech support churn significantly more

## Features
- Churn probability prediction with risk level + business recommendation
- SHAP waterfall plot explaining every individual prediction
- Model monitoring dashboard with KS-test drift detection
- Prediction logging to SQLite

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Author
**Khushi** | [GitHub](https://github.com/Khushirajput123) | [LinkedIn](YOUR_LINKEDIN)
