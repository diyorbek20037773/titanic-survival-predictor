import streamlit as st
import pandas as pd
import joblib

# ------------ Modellar va scaler ------------
model  = joblib.load("model.joblib")
scaler = joblib.load("scaler.joblib")        # scaler.feature_names_in_ xotirada bor

st.title("🚢 Titanic yo‘lovchi omon qolish bashorati")

# ------------ Foydalanuvchi kiritadigan ma'lumotlar ------------
pclass = st.selectbox("Pclass (1-yuqori, 3-past)", [1, 2, 3])
age    = st.slider("Yosh", 0, 80, 25)
sibsp  = st.number_input("SibSp (aka-uka / opa-singil soni)", 0, 10, 0)
parch  = st.number_input("Parch (ota-ona / bolalar soni)", 0, 10, 0)
fare   = st.number_input("Fare (chipta narxi)", 0.0, 600.0, 30.0)

# Embarked portlari – model fit paytida drop_first=True bo‘lgani uchun
embarked_S = st.checkbox("S portidan (Southampton) chiqqanmi?")
embarked_Q = st.checkbox("Q portidan (Queenstown) chiqqanmi?")

sex_male = st.radio("Jins", ["Ayol", "Erkak"]) == "Erkak"   # bool

# ------------ One-hot’ga mos DF (nomlar model kutilganidek) ------------
row = pd.DataFrame({
    "Pclass":     [pclass],
    "Age":        [age],
    "SibSp":      [sibsp],
    "Parch":      [parch],
    "Fare":       [fare],
    "Embarked_Q": [1 if embarked_Q else 0],
    "Embarked_S": [1 if embarked_S else 0],
    "Sex_male":   [1 if sex_male   else 0],
})

# Tartibni scaler/model kutayotgan aniq tartibga keltiramiz
row = row[scaler.feature_names_in_]          # muammo tugadi

# ------------ Bashorat ------------
if st.button("🚀 Bashoratni ko‘rish"):
    row_scaled = scaler.transform(row)
    pred = model.predict(row_scaled)[0]
    if pred == 1:
        st.success("✅ Alloh xohlasa, yo‘lovchi OMOn QOLADI!")
    else:
        st.error("❌ Afsus, omon qolish ehtimoli past.")
