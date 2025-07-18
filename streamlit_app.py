import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('salary_model.pkl')

st.set_page_config(page_title="Employee Salary Predictor", layout="centered")

st.title("💼 Employee Salary Prediction App")

# Form inputs
with st.form("salary_form"):
    st.subheader("Enter Employee Details:")

    age = st.number_input("Age", min_value=18, max_value=65, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    department = st.selectbox("Department", ["Engineering", "HR", "Sales", "Finance", "Marketing", "Support"])
    job_title = st.selectbox("Job Title", ["Engineer", "Analyst", "Manager", "Executive", "Intern"])
    experience = st.slider("Years of Experience", 0, 40, 5)
    education = st.selectbox("Education Level", ["Bachelor", "Master", "PhD"])
    location = st.selectbox("Location", ["New York", "Austin", "Seattle", "San Francisco", "Chicago"])

    submit = st.form_submit_button("Predict Salary")

# Predict
if submit:
    input_df = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Department': [department],
        'Job_Title': [job_title],
        'Experience_Years': [experience],
        'Education_Level': [education],
        'Location': [location]
    })

    prediction = model.predict(input_df)[0]
    st.success(f"💰 Predicted Salary: ${prediction:,.2f}")
