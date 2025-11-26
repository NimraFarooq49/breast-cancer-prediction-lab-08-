import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from time import sleep

# --- File Names ---
MODEL_FILENAME = 'svm_cancer_model.pkl'
SCALER_FILENAME = 'scaler_cancer.pkl'

# --- 30 Features used in the Breast Cancer Dataset ---
# These must exactly match the feature order used during training.
FEATURE_NAMES = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
    'smoothness_mean', 'compactness_mean', 'concavity_mean',
    'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
    'fractal_dimension_se', 'radius_worst', 'texture_worst',
    'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave points_worst',
    'symmetry_worst', 'fractal_dimension_worst'
]

# --- Helper Function to Load Pickled Objects ---
@st.cache_resource
def load_assets(model_path, scaler_path):
    """Loads the trained model and scaler."""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError:
        st.error(f"Error: Required files ({model_path} and/or {scaler_path}) not found.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

# --- Load Assets ---
model, scaler = load_assets(MODEL_FILENAME, SCALER_FILENAME)

# --- Input Widget Creation ---
def user_input_features():
    """Creates a sidebar with input widgets for all 30 features."""
    st.sidebar.header('Patient Data Input')
    
    # Define feature groups for better organization
    groups = {
        'Mean Values': FEATURE_NAMES[0:10],
        'Standard Error (SE) Values': FEATURE_NAMES[10:20],
        'Worst (Largest) Values': FEATURE_NAMES[20:30]
    }
    
    data = {}
    
    # Iterate through groups and create expanding sections
    for group_name, features in groups.items():
        with st.sidebar.expander(group_name, expanded=True):
            for feature in features:
                # Use a simple slider/number input based on feature name.
                # Since the data is scaled, we rely on standard numerical inputs.
                # Using mean/median/std of the dataset for initial values would be better,
                # but we'll use placeholder defaults here for simplicity.
                default_value = 10.0 if 'mean' in feature else (1.0 if 'worst' in feature else 0.1)
                
                # Using st.number_input is generally better than st.slider for precise medical data
                data[feature] = st.number_input(
                    label=feature.replace('_', ' ').title(),
                    min_value=0.0,
                    max_value=100.0, # Setting a high max value for all features
                    value=default_value,
                    step=0.01,
                    format="%.3f",
                    key=feature
                )
    
    # Convert dictionary to DataFrame (required structure for scaling)
    features_df = pd.DataFrame(data, index=[0])
    return features_df

# --- Streamlit Main Page Layout ---
st.title('Breast Cancer Malignancy Prediction')
st.markdown("""
This app predicts whether a tumor is **Benign (0)** or **Malignant (1)** using a trained Support Vector Machine (SVM) model.
---
""")

# Get user input
input_df = user_input_features()

st.subheader('Input Features')
st.dataframe(input_df.T.rename(columns={0: 'Value'}))

# --- Prediction Logic ---
if st.button('Analyze Patient Data'):
    
    with st.spinner('Running prediction...'):
        sleep(1.5) # Simulate processing time
        
        # 1. Scaling the input data is CRITICAL
        input_scaled = scaler.transform(input_df)
        
        # 2. Make prediction
        prediction = model.predict(input_scaled)
        
        # 3. Get probability (requires SVC(probability=True))
        prediction_proba = model.predict_proba(input_scaled)
        
        st.subheader('Prediction Result')
        
        if prediction[0] == 1:
            st.error(
                f"""
                ### 🚨 Result: Malignant (Cancerous)
                **Probability:** {prediction_proba[0][1]*100:.2f}%
                """
            )
            st.balloons()
        else:
            st.success(
                f"""
                ### ✅ Result: Benign (Non-Cancerous)
                **Probability:** {prediction_proba[0][0]*100:.2f}%
                """
            )
            
    st.markdown("""
    ---
    *Disclaimer: This is a machine learning prediction and should not be used for actual medical diagnosis.*
    """)