import streamlit as st
import joblib
import pandas as pd

# Trained model load karo
model = joblib.load("titanic_model.pkl")

st.title("Titanic Survival Predictor")
st.write("Passenger ki details bharo aur dekho wo bachta ya nahi.")

# User se input lo
pclass = st.selectbox("Ticket Class (1 = sabse mehnga)", [1, 2, 3])
sex = st.selectbox("Gender", ["male", "female"])
age = st.slider("Age", 1, 80, 25)
sibsp = st.number_input("Siblings / Spouse saath me", 0, 8, 0)
parch = st.number_input("Parents / Children saath me", 0, 6, 0)
fare = st.number_input("Ticket Fare", 0.0, 500.0, 30.0)
embarked = st.selectbox("Kaha se chadha (C, Q, S)", ["C", "Q", "S"])

# Input ko wahi format do jo training me tha
row = pd.DataFrame(
    [[pclass, age, sibsp, parch, fare,
      sex == "male", embarked == "Q", embarked == "S"]],
    columns=["pclass", "age", "sibsp", "parch", "fare",
             "sex_male", "embarked_Q", "embarked_S"],
)

if st.button("Predict"):
    result = model.predict(row)[0]
    if result == 1:
        st.success("Ye passenger bach jayega!")
    else:
        st.error("Ye passenger nahi bachega.")