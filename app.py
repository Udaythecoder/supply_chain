
import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the LightGBM model
# Ensure the 'models' directory and 'lightgbm_model.pkl' exist
try:
    model = pickle.load(open("models/lightgbm_model.pkl", "rb"))
except FileNotFoundError:
    st.error("Model file 'lightgbm_model.pkl' not found in 'models/' directory.")
    st.stop()

st.title("Supply Chain Demand Prediction")

# Input fields for prediction
st.header("Enter features for prediction:")

#this is an assumption to get the app working; actual feature mapping would need clarification.

# Get the feature names the model expects (excluding 'Number of products sold')
feature_columns = [col for col in X.columns]

# Initialize a dictionary for the input features with dummy values
input_data = {col: [0] for col in feature_columns}

# Create input widgets for all original features in X (minus 'Number of products sold')
input_values = {}
for col in X.columns:
    if X[col].dtype == 'object':
        
        input_values[col] = st.number_input(f"Enter encoded value for {col} (categorical)", value=0, key=f'input_{col}')
    else:
        # For numerical features
        input_values[col] = st.number_input(f"Enter value for {col} ({X[col].dtype})", value=float(X[col].mean()), key=f'input_{col}')


if st.button("Predict Demand"):
    # Create a DataFrame from the input_values dictionary
    # Ensure the column order matches the training data (X.columns)
    input_df = pd.DataFrame([input_values], columns=X.columns)

    for col in input_df.select_dtypes('object').columns:
        temp_encoder = LabelEncoder()
        input_df[col] = temp_encoder.fit_transform(input_df[col])

    if 'Inventory_Turnover' in input_df.columns:
        
        revenue_gen = input_values.get('Revenue generated', 1) # Default to 1 to avoid div by zero
        stock_lev = input_values.get('Stock levels', 1) # Default to 1 to avoid div by zero
        if stock_lev == 0: stock_lev = 1 # Prevent division by zero

        input_df['Inventory_Turnover'] = revenue_gen / stock_lev
    
    input_df = input_df[feature_columns]

    # Predict using the loaded model
    prediction = model.predict(input_df)

    st.success(f"Predicted Demand: {prediction[0]:.2f}")
