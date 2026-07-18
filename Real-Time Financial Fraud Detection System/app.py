import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -------- LOAD --------
@st.cache_resource
def load():
    model = pickle.load(open("model/fraud_model.pkl", "rb"))
    encoders = pickle.load(open("model/encoders.pkl", "rb"))
    columns = pickle.load(open("model/columns.pkl", "rb"))
    return model, encoders, columns

model, encoders, columns = load()

# -------- PREPROCESS --------
def preprocess(df):
    df = df.copy()

    if 'trans_date_trans_time' in df.columns:
        df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], errors='coerce')
        df['hour'] = df['trans_date_trans_time'].dt.hour
        df['day'] = df['trans_date_trans_time'].dt.day
        df = df.drop(columns=['trans_date_trans_time'])

    if 'amt' in df.columns:
        df['amt_log'] = np.log1p(df['amt'])

    for col, encoder in encoders.items():
        if col in df.columns:
            try:
                df[col] = encoder.transform(df[col].astype(str))
            except:
                df[col] = 0

    for col in columns:
        if col not in df.columns:
            df[col] = 0

    return df[columns].fillna(0)

# -------- UI --------
st.title("💳 Fraud Detection System")

# -------- SINGLE INPUT --------
st.sidebar.header("Single Transaction")

amount = st.sidebar.number_input("Amount", 0.0)
predict = st.sidebar.button("Predict")

if predict:
    df = pd.DataFrame([{"amt": amount}])
    df_model = preprocess(df)

    prob = model.predict_proba(df_model)[0][1]
    risk = int(prob * 100)

    if risk > 60:
        st.error(f"Fraud 🚨 ({risk}%)")
    elif risk > 30:
        st.warning(f"Suspicious ⚠️ ({risk}%)")
    else:
        st.success(f"Normal ✅ ({risk}%)")

# -------- CSV --------
st.subheader("Upload CSV")

file = st.file_uploader("Upload file", type=["csv"])

if file:
    df = pd.read_csv(file)

    df_model = preprocess(df)
    probs = model.predict_proba(df_model)[:, 1]

    df['Risk'] = (probs * 100).astype(int)
    df['Prediction'] = df['Risk'].apply(
        lambda x: "Fraud" if x > 60 else "Suspicious" if x > 30 else "Normal"
    )

    st.write(df.head())
    st.bar_chart(df['Prediction'].value_counts())