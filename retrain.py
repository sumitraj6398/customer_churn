import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib
import os

# Load data
df = pd.read_csv('data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Clean
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
df = df.drop('customerID', axis=1)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0}).astype(int)

# Feature engineering
service_cols = ['PhoneService','OnlineSecurity','OnlineBackup',
                'DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
df['total_services'] = (df[service_cols] == 'Yes').sum(axis=1)
df['charge_per_tenure'] = df['TotalCharges'] / (df['tenure'] + 1)
df['tenure_group'] = pd.cut(df['tenure'], bins=[0,12,24,48,72],
                             labels=['0-1yr','1-2yr','2-4yr','4-6yr'])
df['household_type'] = df['Partner'] + '_' + df['Dependents']

# Encode
X = pd.get_dummies(df.drop('Churn', axis=1), drop_first=True).astype(int)
y = df['Churn']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# Scale
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])

# Train
scale_pos_weight = (y_train==0).sum() / (y_train==1).sum()
model = XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    eval_metric='auc',
    random_state=42,
    subsample=0.7,
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    colsample_bytree=0.8
)
model.fit(X_train, y_train)

# Save
os.makedirs('models', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)
joblib.dump(model, 'models/churn_model_v1.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(list(X_train.columns), 'models/feature_columns.pkl')
X_train.sample(1000, random_state=42).to_csv(
    'data/processed/reference_data.csv', index=False)

print("All saved successfully with sklearn", )
from sklearn import __version__
print("sklearn version:", __version__)