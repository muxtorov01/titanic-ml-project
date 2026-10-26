import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load("titanic_model.joblib")

model = load_model()

st.title("🚢 Titanic: Tirik qolish bashorati")
st.caption("Yo'lovchi ma'lumotlarini kiriting, model tirik qolish ehtimolini hisoblaydi.")

with st.form("passenger_form"):
    col1, col2 = st.columns(2)
    with col1:
        sex = st.radio("Jinsi", ["female", "male"],
                       format_func=lambda x: "Ayol" if x == "female" else "Erkak",
                       horizontal=True)
        age = st.slider("Yoshi", 0, 80, 30)
        pclass = st.selectbox("Sinf", [1, 2, 3], format_func=lambda x: f"{x}-sinf")
        title = st.selectbox("Murojaat (Title)", ["Mr", "Mrs", "Miss", "Master", "Rare"])
    with col2:
        sibsp = st.number_input("Aka-uka/er-xotin soni (SibSp)", 0, 8, 0)
        parch = st.number_input("Ota-ona/bolalar soni (Parch)", 0, 6, 0)
        fare = st.number_input("Chipta narxi (Fare)", 0.0, 520.0, 32.0, step=1.0)
        embarked = st.selectbox("Port", ["S", "C", "Q"],
                                format_func=lambda x: {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}[x])
    submitted = st.form_submit_button("Bashorat qilish", use_container_width=True)

if submitted:
    X = pd.DataFrame([{
        "Pclass": pclass, "Sex": sex, "Age": float(age),
        "SibSp": sibsp, "Parch": parch, "Fare": fare,
        "Embarked": embarked, "FamilySize": sibsp + parch + 1, "Title": title,
    }])
    prob = model.predict_proba(X)[0, 1]
    st.divider()
    st.subheader("Natija")
    st.metric("Tirik qolish ehtimoli", f"{prob:.1%}")
    st.progress(float(prob))
    if prob >= 0.5:
        st.success("✅ Tirik qolishi **ehtimoli yuqori**.")
    else:
        st.error("⚠️ Tirik qolishi **ehtimoli past**.")

st.divider()
st.caption("Model: Logistic Regression (Accuracy ≈ 0.84, ROC-AUC ≈ 0.88). Ta'lim maqsadida yaratilgan.")
