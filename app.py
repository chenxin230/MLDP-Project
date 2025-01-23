import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Obesity level mapping
obesity_mapping = {
    0: 'Insufficient_Weight',
    1: 'Normal_Weight', 
    2: 'Overweight_Level_I',
    3: 'Overweight_Level_II', 
    4: 'Obesity_Type_I',
    5: 'Obesity_Type_II',
    6: 'Obesity_Type_III'
}

st.title('Obesity Classification Predictor')

loaded_data = joblib.load('obesity_decision_tree_retrained_model.pkl')
MODEL_FEATURE_NAMES = loaded_data.feature_names_in_

st.sidebar.header('Enter Your Details')

def user_input_features():
    age = st.sidebar.slider('Age', 15, 70, 30)
    gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
    height = st.sidebar.slider('Height (m)', 1.4, 2.0, 1.7)
    weight = st.sidebar.slider('Weight (kg)', 40, 150, 70)
    
    family_history = st.sidebar.selectbox('Family History of Obesity', ['Yes', 'No'])
    high_calorie_food = st.sidebar.selectbox('Frequent High Calorie Food Consumption', ['Yes', 'No'])
    
    vegetable_meals = st.sidebar.slider('Vegetables in Meals (Frequency)', 1, 3, 2)
    main_meals = st.sidebar.slider('Number of Main Meals Daily', 1, 4, 3)
    water_intake = st.sidebar.slider('Water Intake (Glasses per Day)', 1, 3, 2)
    physical_activity = st.sidebar.slider('Physical Activity Frequency', 0, 3, 1)
    
    # Prepare data dictionary
    data = {
        'Age': age,
        'Gender': 1 if gender == 'Male' else 0,
        'family_history': 1 if family_history == 'Yes' else 0,
        'FAVC': 1 if high_calorie_food == 'Yes' else 0,
        'FCVC': vegetable_meals,
        'NCP': main_meals,
        'CH2O': water_intake,
        'FAF': physical_activity,
        'BMI': weight / (height ** 2),
        'CAEC_Frequently': 0,
        'CAEC_Sometimes': 1,
        'CALC_Sometimes': 1,
        'CALC_no': 0,
        'MTRANS_Automobile': 1,
        'MTRANS_Public_Transportation': 0
    }

    input_df = pd.DataFrame([{feature: data.get(feature, 0) for feature in MODEL_FEATURE_NAMES}])
    
    return input_df

input_df = user_input_features()

st.subheader('Your Health Profile')
st.write(input_df)

# Prediction button
if st.button('Predict Obesity Classification'):
    # Make prediction
    prediction = loaded_data.predict(input_df)
    prediction_proba = loaded_data.predict_proba(input_df)
    
    # Display results
    st.subheader('Prediction Results')
    st.write(f"**Predicted Obesity Level:** {obesity_mapping[prediction[0]]}")
    
    st.subheader('Classification Probabilities')
    proba_df = pd.DataFrame({
        'Obesity Level': list(obesity_mapping.values()),
        'Probability (%)': prediction_proba[0] * 100
    })
    st.dataframe(proba_df)