# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import shap
# import matplotlib.pyplot as plt
# import sqlite3
# from datetime import datetime
#
# # ── Load saved artifacts ──────────────────────────────
# model = joblib.load('models/churn_model_v1.pkl')
# scaler = joblib.load('models/scaler.pkl')
# feature_columns = joblib.load('models/feature_columns.pkl')
#
# # ── Page config ───────────────────────────────────────
# st.set_page_config(page_title="Churn Predictor", layout="wide")
# st.title("Customer Churn Prediction + Monitoring")
#
# # ── Tabs ──────────────────────────────────────────────
# tab1, tab2 = st.tabs(["Predict", "Monitoring Dashboard"])
#
# # ════════════════════════════════════════════════════
# # TAB 1: PREDICTION
# # ════════════════════════════════════════════════════
# with tab1:
#     st.header("Predict Customer Churn")
#     st.sidebar.header("Enter Customer Details")
#
#     tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
#     monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18, 120, 65)
#     total_charges = monthly_charges * tenure
#
#     contract = st.sidebar.selectbox("Contract",
#         ["Month-to-month", "One year", "Two year"])
#     internet = st.sidebar.selectbox("Internet Service",
#         ["DSL", "Fiber optic", "No"])
#     tech_support = st.sidebar.selectbox("Tech Support",
#         ["Yes", "No", "No internet service"])
#     payment = st.sidebar.selectbox("Payment Method",
#         ["Electronic check", "Mailed check",
#          "Bank transfer (automatic)", "Credit card (automatic)"])
#     partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
#     dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
#     senior = st.sidebar.selectbox("Senior Citizen", [0, 1])
#     paperless = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
#
#     if st.sidebar.button("Predict"):
#
#         # Build raw input
#         input_dict = {
#             'gender': 'Male',
#             'SeniorCitizen': senior,
#             'Partner': partner,
#             'Dependents': dependents,
#             'tenure': tenure,
#             'PhoneService': 'Yes',
#             'MultipleLines': 'No',
#             'InternetService': internet,
#             'OnlineSecurity': 'No',
#             'OnlineBackup': 'No',
#             'DeviceProtection': 'No',
#             'TechSupport': tech_support,
#             'StreamingTV': 'No',
#             'StreamingMovies': 'No',
#             'Contract': contract,
#             'PaperlessBilling': paperless,
#             'PaymentMethod': payment,
#             'MonthlyCharges': monthly_charges,
#             'TotalCharges': total_charges
#         }
#
#         input_df = pd.DataFrame([input_dict])
#
#         # Feature engineering (same as training)
#         service_cols = ['PhoneService','OnlineSecurity','OnlineBackup',
#                         'DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
#         input_df['total_services'] = (input_df[service_cols]=='Yes').sum(axis=1)
#         input_df['charge_per_tenure'] = input_df['TotalCharges'] / (input_df['tenure']+1)
#         input_df['tenure_group'] = pd.cut(input_df['tenure'],
#                                            bins=[0,12,24,48,72],
#                                            labels=['0-1yr','1-2yr','2-4yr','4-6yr'])
#         input_df['household_type'] = input_df['Partner']+'_'+input_df['Dependents']
#
#         # Encode + align columns
#         input_encoded = pd.get_dummies(input_df).reindex(
#             columns=feature_columns, fill_value=0).astype(int)
#
#         # Scale
#         numeric_cols = ['tenure','MonthlyCharges','TotalCharges',
#                         'total_services','charge_per_tenure']
#         input_encoded[numeric_cols] = scaler.transform(
#             input_encoded[numeric_cols])
#
#         # Predict
#         prob = model.predict_proba(input_encoded)[0][1]
#
#         # Display
#         col1, col2 = st.columns(2)
#         with col1:
#             st.metric("Churn Probability", f"{prob:.1%}")
#         with col2:
#             if prob > 0.5:
#                 st.error("⚠️ High Risk - Likely to Churn")
#             else:
#                 st.success("✅ Low Risk - Likely to Stay")
#
#         # Risk level
#         if prob > 0.7:
#             st.warning("Recommended Action: Offer immediate retention discount")
#         elif prob > 0.5:
#             st.warning("Recommended Action: Schedule customer success call")
#         else:
#             st.info("Recommended Action: Standard engagement")
#
#         # SHAP explanation
#         st.subheader("Why this prediction?")
#         explainer = shap.TreeExplainer(model)
#         shap_vals = explainer.shap_values(input_encoded)
#
#         fig, ax = plt.subplots(figsize=(10,5))
#         shap.waterfall_plot(shap.Explanation(
#             values=shap_vals[0],
#             base_values=explainer.expected_value,
#             data=input_encoded.iloc[0],
#             feature_names=feature_columns
#         ), show=False)
#         st.pyplot(fig)
#         plt.close()
#
#         # Log to SQLite
#         conn = sqlite3.connect('data/predictions.db')
#         conn.execute('''CREATE TABLE IF NOT EXISTS predictions_log
#                        (timestamp TEXT, tenure REAL, monthly_charges REAL,
#                         total_charges REAL, contract TEXT, prediction REAL)''')
#         conn.execute("INSERT INTO predictions_log VALUES (?,?,?,?,?,?)",
#                     (str(datetime.now()), tenure, monthly_charges,
#                      total_charges, contract, prob))
#         conn.commit()
#         conn.close()
#
# # ════════════════════════════════════════════════════
# # TAB 2: MONITORING
# # ════════════════════════════════════════════════════
# with tab2:
#     st.header("Model Monitoring Dashboard")
#
#     try:
#         ref = pd.read_csv('data/processed/reference_data.csv')
#         conn = sqlite3.connect('data/predictions.db')
#         logs = pd.read_sql('SELECT * FROM predictions_log', conn)
#         conn.close()
#
#         if len(logs) >= 5:
#             from scipy import stats
#
#             # Drift detection
#             st.subheader("Data Drift Detection")
#             results = []
#             for col in ['tenure','monthly_charges','total_charges']:
#                 ref_col = col if col in ref.columns else col
#                 if ref_col in ref.columns and col in logs.columns:
#                     ks, p = stats.ks_2samp(ref[ref_col], logs[col])
#                     results.append({
#                         'Feature': col,
#                         'KS Statistic': round(ks, 3),
#                         'P-value': round(p, 3),
#                         'Status': "⚠️ Drift" if p < 0.05 else "✅ Stable"
#                     })
#             st.dataframe(pd.DataFrame(results))
#
#             # Prediction trend
#             st.subheader("Churn Risk Trend Over Time")
#             logs['timestamp'] = pd.to_datetime(logs['timestamp'])
#             st.line_chart(logs.set_index('timestamp')['prediction'])
#
#             # Distribution
#             st.subheader("Prediction Distribution")
#             fig2, ax2 = plt.subplots()
#             logs['prediction'].hist(bins=20, ax=ax2, color='coral')
#             ax2.set_xlabel("Churn Probability")
#             ax2.set_ylabel("Count")
#             st.pyplot(fig2)
#             plt.close()
#
#         else:
#             st.info("Make at least 5 predictions to see monitoring data")
#
#     except Exception as e:
#         st.info("No predictions logged yet — go to Predict tab first")






import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import sqlite3
from datetime import datetime

# ── Load saved artifacts ──────────────────────────────
model = joblib.load('models/churn_model_v1.pkl')
scaler = joblib.load('models/scaler.pkl')
feature_columns = joblib.load('models/feature_columns.pkl')

# ── Page config ───────────────────────────────────────
st.set_page_config(page_title="Churn Predictor", layout="wide")
st.title("Customer Churn Prediction + Monitoring")

# ── Tabs ──────────────────────────────────────────────
tab1, tab2 = st.tabs(["Predict", "Monitoring Dashboard"])

# ════════════════════════════════════════════════════
# TAB 1: PREDICTION
# ════════════════════════════════════════════════════
with tab1:
    st.header("Predict Customer Churn")
    st.sidebar.header("Enter Customer Details")

    tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18, 120, 65)
    total_charges = monthly_charges * tenure

    contract = st.sidebar.selectbox("Contract",
        ["Month-to-month", "One year", "Two year"])
    internet = st.sidebar.selectbox("Internet Service",
        ["DSL", "Fiber optic", "No"])
    tech_support = st.sidebar.selectbox("Tech Support",
        ["Yes", "No", "No internet service"])
    payment = st.sidebar.selectbox("Payment Method",
        ["Electronic check", "Mailed check",
         "Bank transfer (automatic)", "Credit card (automatic)"])
    partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
    dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
    senior = st.sidebar.selectbox("Senior Citizen", [0, 1])
    paperless = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
    phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.sidebar.selectbox("Multiple Lines",
        ["Yes", "No", "No phone service"])
    online_security = st.sidebar.selectbox("Online Security",
        ["Yes", "No", "No internet service"])
    online_backup = st.sidebar.selectbox("Online Backup",
        ["Yes", "No", "No internet service"])
    device_protection = st.sidebar.selectbox("Device Protection",
        ["Yes", "No", "No internet service"])
    streaming_tv = st.sidebar.selectbox("Streaming TV",
        ["Yes", "No", "No internet service"])
    streaming_movies = st.sidebar.selectbox("Streaming Movies",
        ["Yes", "No", "No internet service"])
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

    if st.sidebar.button("Predict"):

        # Build raw input — all original columns
        input_dict = {
            'gender': gender,
            'SeniorCitizen': senior,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone_service,
            'MultipleLines': multiple_lines,
            'InternetService': internet,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless,
            'PaymentMethod': payment,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }

        input_df = pd.DataFrame([input_dict])

        # Feature engineering — must match training exactly
        service_cols = ['PhoneService','OnlineSecurity','OnlineBackup',
                        'DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
        input_df['total_services'] = (input_df[service_cols] == 'Yes').sum(axis=1)
        input_df['charge_per_tenure'] = input_df['TotalCharges'] / (input_df['tenure'] + 1)
        input_df['tenure_group'] = pd.cut(
            input_df['tenure'],
            bins=[0, 12, 24, 48, 72],
            labels=['0-1yr', '1-2yr', '2-4yr', '4-6yr']
        )
        input_df['household_type'] = input_df['Partner'] + '_' + input_df['Dependents']

        # Encode + align to training columns
        input_encoded = pd.get_dummies(input_df).reindex(
            columns=feature_columns, fill_value=0).astype(int)

        # Scale — ONLY the 3 columns scaler was fitted on
        numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        input_encoded[numeric_cols] = scaler.transform(
            input_encoded[numeric_cols])

        # Predict
        prob = model.predict_proba(input_encoded)[0][1]

        # ── Display results ──
        st.subheader("Prediction Result")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Churn Probability", f"{prob:.1%}")
        with col2:
            st.metric("Tenure", f"{tenure} months")
        with col3:
            st.metric("Monthly Charges", f"${monthly_charges}")

        if prob > 0.5:
            st.error("⚠️ HIGH RISK — This customer is likely to churn")
        else:
            st.success("✅ LOW RISK — This customer is likely to stay")

        # Risk-based recommendation
        st.subheader("Recommended Action")
        if prob > 0.7:
            st.warning("🔴 Offer immediate retention discount or upgrade")
        elif prob > 0.5:
            st.warning("🟡 Schedule a customer success call within 7 days")
        elif prob > 0.3:
            st.info("🟢 Send engagement email / loyalty reward")
        else:
            st.info("🟢 Standard engagement — customer appears satisfied")

        # ── SHAP explanation ──
        st.subheader("Why this prediction?")
        try:
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(input_encoded)

            fig, ax = plt.subplots(figsize=(10, 5))
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_vals[0],
                    base_values=explainer.expected_value,
                    data=input_encoded.iloc[0],
                    feature_names=feature_columns
                ), show=False
            )
            st.pyplot(fig)
            plt.close()
        except Exception as e:
            st.warning(f"SHAP plot could not be generated: {e}")

        # ── Log to SQLite ──
        try:
            os.makedirs('data', exist_ok=True)
            conn = sqlite3.connect('data/predictions.db')
            conn.execute('''CREATE TABLE IF NOT EXISTS predictions_log
                           (timestamp TEXT, tenure REAL, monthly_charges REAL,
                            total_charges REAL, contract TEXT, prediction REAL)''')
            conn.execute("INSERT INTO predictions_log VALUES (?,?,?,?,?,?)",
                        (str(datetime.now()), tenure, monthly_charges,
                         total_charges, contract, prob))
            conn.commit()
            conn.close()
            st.caption("✓ Prediction logged for monitoring")
        except Exception as e:
            st.caption(f"Logging skipped: {e}")

# ════════════════════════════════════════════════════
# TAB 2: MONITORING DASHBOARD
# ════════════════════════════════════════════════════
with tab2:
    st.header("Model Monitoring Dashboard")
    st.markdown("Tracks prediction trends and detects data drift over time.")

    try:
        # Load reference training data
        ref = pd.read_csv('data/processed/reference_data.csv')

        # Load prediction logs
        conn = sqlite3.connect('data/predictions.db')
        logs = pd.read_sql('SELECT * FROM predictions_log', conn)
        conn.close()

        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Predictions Made", len(logs))
        with col2:
            if len(logs) > 0:
                high_risk = (logs['prediction'] > 0.5).sum()
                st.metric("High Risk Predictions", high_risk)
        with col3:
            if len(logs) > 0:
                avg_prob = logs['prediction'].mean()
                st.metric("Average Churn Probability", f"{avg_prob:.1%}")

        if len(logs) >= 5:
            from scipy import stats

            # ── Drift Detection ──
            st.subheader("Data Drift Detection (KS Test)")
            st.markdown("Compares incoming prediction inputs vs training distribution. "
                       "P-value < 0.05 = drift detected.")

            drift_results = []
            # Map log column names to reference column names
            col_mapping = {
                'tenure': 'tenure',
                'monthly_charges': 'MonthlyCharges',
                'total_charges': 'TotalCharges'
            }

            for log_col, ref_col in col_mapping.items():
                if ref_col in ref.columns and log_col in logs.columns:
                    ks_stat, p_val = stats.ks_2samp(
                        ref[ref_col].dropna(),
                        logs[log_col].dropna()
                    )
                    drift_results.append({
                        'Feature': ref_col,
                        'KS Statistic': round(ks_stat, 3),
                        'P-value': round(p_val, 3),
                        'Status': "⚠️ Drift Detected" if p_val < 0.05 else "✅ Stable"
                    })

            if drift_results:
                st.dataframe(pd.DataFrame(drift_results), use_container_width=True)
            else:
                st.info("Column names don't match for drift detection")

            # ── Prediction trend ──
            st.subheader("Churn Risk Trend Over Time")
            logs['timestamp'] = pd.to_datetime(logs['timestamp'])
            logs_sorted = logs.sort_values('timestamp')
            st.line_chart(logs_sorted.set_index('timestamp')['prediction'])

            # ── Prediction distribution ──
            st.subheader("Prediction Distribution")
            fig2, ax2 = plt.subplots(figsize=(8, 3))
            logs['prediction'].hist(bins=20, ax=ax2, color='coral', edgecolor='white')
            ax2.axvline(0.5, color='red', linestyle='--', label='Decision threshold')
            ax2.set_xlabel("Churn Probability")
            ax2.set_ylabel("Count")
            ax2.legend()
            st.pyplot(fig2)
            plt.close()

            # ── Recent predictions table ──
            st.subheader("Recent Predictions Log")
            st.dataframe(
                logs.sort_values('timestamp', ascending=False).head(10),
                use_container_width=True
            )

        else:
            remaining = 5 - len(logs)
            st.info(f"Make {remaining} more prediction(s) in the Predict tab "
                   f"to unlock the monitoring dashboard.")

    except FileNotFoundError:
        st.info("Reference data not found. Make sure "
               "'data/processed/reference_data.csv' exists.")
    except Exception as e:
        st.info(f"No predictions logged yet — go to Predict tab first. ({e})")