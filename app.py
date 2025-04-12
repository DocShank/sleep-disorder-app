import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the model
try:
    model = joblib.load('sleep_disorder_model.joblib')
    st.success("Model loaded successfully!")
except FileNotFoundError:
    st.error("Model file 'sleep_disorder_model.joblib' not found. Please upload...")
    st.stop()

# Define feature columns (45 features)
feature_columns = [
    'Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level',
    'Stress Level', 'Heart Rate', 'Daily Steps', 'Gender_Male',
    'Occupation_Doctor', 'Occupation_Engineer', 'Occupation_Lawyer',
    'Occupation_Manager', 'Occupation_Nurse', 'Occupation_Sales Representative',
    'Occupation_Salesperson', 'Occupation_Scientist', 'Occupation_Software Engineer',
    'Occupation_Teacher', 'BMI Category_Normal Weight', 'BMI Category_Obese',
    'BMI Category_Overweight', 'Blood Pressure_115/78', 'Blood Pressure_117/76',
    'Blood Pressure_118/75', 'Blood Pressure_118/76', 'Blood Pressure_119/77',
    'Blood Pressure_120/80', 'Blood Pressure_121/79', 'Blood Pressure_122/80',
    'Blood Pressure_125/80', 'Blood Pressure_125/82', 'Blood Pressure_126/83',
    'Blood Pressure_128/84', 'Blood Pressure_128/85', 'Blood Pressure_129/84',
    'Blood Pressure_130/85', 'Blood Pressure_130/86', 'Blood Pressure_131/86',
    'Blood Pressure_132/87', 'Blood Pressure_135/88', 'Blood Pressure_135/90',
    'Blood Pressure_139/91', 'Blood Pressure_140/90', 'Blood Pressure_140/95',
    'Blood Pressure_142/92'
]

# Function to preprocess input data
def preprocess_example(example):
    example_df = pd.DataFrame([example])
    example_encoded = pd.get_dummies(example_df, columns=['Gender', 'Occupation', 'BMI Category', 'Blood Pressure'])
    for col in feature_columns:
        if col not in example_encoded.columns:
            example_encoded[col] = 0
    return example_encoded[feature_columns]

# Streamlit app
st.title("Sleep Disorder Prediction App")
st.write("Enter the patient details below to predict the likelihood of a sleep disorder.")

# Input fields
age = st.number_input("Age", min_value=18, max_value=100, value=30, help="Enter age in years")
gender = st.selectbox("Gender", ["Male", "Female"], help="Select gender")
occupation = st.selectbox("Occupation", [
    "Doctor", "Engineer", "Lawyer", "Manager", "Nurse",
    "Sales Representative", "Salesperson", "Scientist", "Software Engineer", "Teacher"
], help="Select occupation")
sleep_duration = st.number_input("Sleep Duration (hours)", min_value=0.0, max_value=24.0, value=7.0, step=0.1, help="Enter sleep duration per night")
quality_of_sleep = st.slider("Quality of Sleep", min_value=1, max_value=10, value=6, help="Rate sleep quality (1=poor, 10=excellent)")
physical_activity = st.number_input("Physical Activity Level (minutes/day)", min_value=0, max_value=300, value=30, help="Enter daily physical activity in minutes")
stress_level = st.slider("Stress Level", min_value=1, max_value=10, value=5, help="Rate stress level (1=low, 10=high)")
bmi_category = st.selectbox("BMI Category", ["Normal Weight", "Overweight", "Obese"], help="Select BMI category")
blood_pressure = st.selectbox("Blood Pressure (mmHg)", [
    "115/78", "117/76", "118/75", "118/76", "119/77", "120/80",
    "121/79", "122/80", "125/80", "125/82", "126/83", "128/84",
    "128/85", "129/84", "130/85", "130/86", "131/86", "132/87",
    "135/88", "135/90", "139/91", "140/90", "140/95", "142/92"
], help="Select blood pressure")
heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=150, value=70, help="Enter heart rate in beats per minute")
daily_steps = st.number_input("Daily Steps", min_value=0, max_value=30000, value=5000, help="Enter average daily steps")

# Prediction
if st.button("Predict"):
    try:
        # Create input dictionary
        example = {
            'Age': age,
            'Sleep Duration': sleep_duration,
            'Quality of Sleep': quality_of_sleep,
            'Physical Activity Level': physical_activity,
            'Stress Level': stress_level,
            'Heart Rate': heart_rate,
            'Daily Steps': daily_steps,
            'Gender': gender,
            'Occupation': occupation,
            'BMI Category': bmi_category,
            'Blood Pressure': blood_pressure
        }

        # Preprocess and predict
        example_processed = preprocess_example(example)
        prediction = model.predict(example_processed)[0]
        probability = model.predict_proba(example_processed)[0][1]

        # Display results
        if prediction == 1:
            st.error(f"Prediction: Sleep Disorder")
            st.write(f"Probability of Sleep Disorder: {probability:.2f}")
            st.error("This individual is likely to have a sleep disorder. Consider consulting a specialist.")
        else:
            st.success(f"Prediction: No Sleep Disorder")
            st.write(f"Probability of Sleep Disorder: {probability:.2f}")
            st.success("This individual is unlikely to have a sleep disorder.")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
