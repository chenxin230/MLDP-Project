# app.py
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

# Sidebar for user inputs
st.sidebar.header('Enter Your Details')

def user_input_features():
    # Collect user inputs
    age = st.sidebar.slider('Age', 15, 70, 30)
    gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
    height = st.sidebar.slider('Height (m)', 1.4, 2.0, 1.7)
    weight = st.sidebar.slider('Weight (kg)', 40, 150, 70)
    
    family_history = st.sidebar.selectbox('Family History of Obesity', ['Yes', 'No'])
    high_calorie_food = st.sidebar.selectbox('Frequent High Calorie Food Consumption', ['Yes', 'No'])
    
    # Add smoking selectbox
    smoking = st.sidebar.selectbox('Do You Smoke?', ['No', 'Yes'])
    
    vegetable_meals = st.sidebar.slider('Vegetables in Meals (Frequency)', 1, 3, 2)
    main_meals = st.sidebar.slider('Number of Main Meals Daily', 1, 4, 3)
    water_intake = st.sidebar.slider('Water Intake (Glasses per Day)', 1, 3, 2)
    physical_activity = st.sidebar.slider('Physical Activity Frequency', 0, 3, 1)
    
    transport_mode = st.sidebar.selectbox('Main Transportation', 
        ['Public Transportation', 'Automobile', 'Motorbike', 'Bike', 'Walking'])

    # Prepare data dictionary
    data = {
        'Age': [age],
        'Gender': [1 if gender == 'Male' else 0],
        'family_history': [1 if family_history == 'Yes' else 0],
        'FAVC': [1 if high_calorie_food == 'Yes' else 0],
        'FCVC': [vegetable_meals],
        'NCP': [main_meals],
        'CH2O': [water_intake],
        'FAF': [physical_activity],
        'BMI': [weight / (height ** 2)],
        'SMOKE': [1 if smoking == 'Yes' else 0],  # Updated to map smoking status
        'CAEC_no': [1], 'CAEC_sometimes': [0], 'CAEC_frequently': [0], 'CAEC_always': [0],
        'CALC_no': [1], 'CALC_sometimes': [0], 'CALC_frequently': [0], 'CALC_always': [0]
    }

    # One-hot encoding for transport mode
    transport_modes = ['Public_Transportation', 'Automobile', 'Motorbike', 'Bike', 'Walking']
    for mode in transport_modes:
        data[f'MTRANS_{mode}'] = [1 if transport_mode.replace(' ', '_') == mode else 0]

    return pd.DataFrame(data)

# Get user input
input_df = user_input_features()

# Display input parameters
st.subheader('Your Health Profile')
st.write(input_df)

# Prediction button
if st.button('Predict Obesity Classification'):
    # Load your trained model
    loaded_data = joblib.load('obesity_decision_tree_model.pkl')
    model = loaded_data['model']
    expected_columns = loaded_data['columns']
    
    # Reorder and ensure all columns are present
    input_df = input_df.reindex(columns=expected_columns, fill_value=0)
    
    # Make prediction
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)
    
    # Display results
    st.subheader('Prediction Results')
    st.write(f"**Predicted Obesity Level:** {obesity_mapping[prediction[0]]}")
    
    # Probability breakdown
    st.subheader('Classification Probabilities')
    proba_df = pd.DataFrame({
        'Obesity Level': list(obesity_mapping.values()),
        'Probability (%)': prediction_proba[0] * 100
    })
    st.dataframe(proba_df)