import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

st.set_page_config(page_title="Credit Risk Predictor", layout="wide")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("german_credit_data.csv")

df = load_data()

st.title("💳 Credit Risk Prediction App")
st.markdown("Predict whether a loan applicant is a **good** or **bad** credit risk using machine learning.")

# Display raw data
st.subheader("Dataset Preview")
st.dataframe(df.head())

# Check for columns
st.write(f"Columns in dataset: {', '.join(df.columns)}")

# Check if Risk column exists
if 'Risk' not in df.columns:
    # Create a synthetic target variable (for demonstration only)
    st.info("Creating a synthetic Risk column based on Credit amount and Duration")
    df['Risk'] = np.where(
        (df['Credit amount'] > df['Credit amount'].median()) & 
        (df['Duration'] > df['Duration'].median()), 
        0,  # Higher risk
        1   # Lower risk
    )
    target_col = 'Risk'
    st.write("Synthetic Risk column created (1 = Good, 0 = Bad)")
else:
    target_col = 'Risk'

# Handle NA values
for col in df.columns:
    if df[col].isna().any():
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

# Split features/target
target = df[target_col]
features = df.drop(target_col, axis=1)

# One-hot encode features
features_encoded = pd.get_dummies(features, drop_first=True)
X = features_encoded
y = target

st.write(f"Target distribution: {y.value_counts().to_dict()}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate model
y_pred = model.predict(X_test_scaled)
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])

st.sidebar.header("Model Evaluation")
st.sidebar.markdown(f"**ROC-AUC:** {roc_auc:.2f}")
st.sidebar.text(classification_report(y_test, y_pred))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test_scaled)[:, 1])
st.subheader("📈 ROC Curve")
fig, ax = plt.subplots()
ax.plot(fpr, tpr, label='AUC = {:.2f}'.format(roc_auc))
ax.plot([0,1], [0,1], 'k--')
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend()
st.pyplot(fig)

# Feature Importance (without SHAP)
st.subheader("🔍 Feature Importance")
plt.figure(figsize=(10, 8))
importances = model.feature_importances_
indices = np.argsort(importances)[-15:]  # Top 15 features
plt.barh(range(len(indices)), importances[indices])
plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
plt.xlabel('Feature Importance')
plt.title('Top Features (Random Forest)')
st.pyplot(plt)

# User input section
st.subheader("📊 Predict New Applicant")
user_input = {}

# Create two columns for input form
col1, col2 = st.columns(2)

# Split fields between columns
fields = list(X.columns)
half = len(fields) // 2

with col1:
    for col in fields[:half]:
        if any(keyword in col for keyword in ['Age', 'Duration', 'amount']):
            user_input[col] = st.number_input(col, value=float(X[col].median()))
        else:
            unique_vals = sorted(X[col].unique())
            user_input[col] = st.selectbox(col, unique_vals)

with col2:
    for col in fields[half:]:
        if any(keyword in col for keyword in ['Age', 'Duration', 'amount']):
            user_input[col] = st.number_input(col, value=float(X[col].median()))
        else:
            unique_vals = sorted(X[col].unique())
            user_input[col] = st.selectbox(col, unique_vals)

input_df = pd.DataFrame([user_input])
input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)[0]
prob = model.predict_proba(input_scaled)[0][1]

st.markdown("---")
st.subheader("🔮 Prediction Result")
col1, col2 = st.columns(2)
with col1:
    if prediction == 1:
        st.success("Credit Risk: Good")
    else:
        st.error("Credit Risk: Bad")
        
with col2:
    st.metric("Probability of Good Credit", f"{prob:.2%}")

# Save model and scaler
joblib.dump(model, "credit_risk_model.pkl")
joblib.dump(scaler, "scaler.pkl")
