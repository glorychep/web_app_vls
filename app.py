
import streamlit as st
import pandas as pd
import joblib
import numpy as np # Import numpy for isnan check

# Load trained model
try:
    # Load the model trained in cell 7ohF3S-NBrXi
    model = joblib.load("vl_model_pipeline.pkl")
except FileNotFoundError:
    st.error("Model file not found. Please train the model first using cell 7ohF3S-NBrXi.")
    st.stop()

st.title("Viral Load Suppression Prediction")

st.write("Upload patient data (CSV) or enter manually to predict suppression.")

# --- Option A: CSV Upload ---
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    try:
        # Define the expected feature names based on the model trained in cell 7ohF3S-NBrXi
        original_feature_names = ['age', 'Weight', 'Height', 'Months of Prescription', 'Current Regimen', 'Last VL Result']

        # Ensure the uploaded data has the necessary columns
        data_for_prediction = data[original_feature_names].copy()

        # Handle potential missing values in uploaded data if necessary (model pipeline handles some, but direct input might need checks)
        # For simplicity, assuming the model pipeline handles imputation.

        preds = model.predict(data_for_prediction)
        data["Prediction"] = ["Suppressed" if p == 1 else "Not Suppressed" for p in preds]
        st.write("### Predictions")
        st.dataframe(data)
    except KeyError as e:
        st.error(f"Missing columns in uploaded CSV: {e}. Please ensure the CSV contains columns: {original_feature_names}")
    except Exception as e:
        st.error(f"Error making predictions: {e}")


# --- Option B: Manual Input ---
st.write("### Manual Input")

# Use a dictionary to store user inputs
user_input = {}

user_input["Age at reporting"] = st.number_input("Enter Age at Reporting", min_value=0, max_value=120, value=30)
user_input["Sex"] = st.selectbox("Select Sex", ["Male", "Female", "Other"])
regimen_options = ["TDF+3TC+DTG", "AZT+3TC+NVP", "TDF+3TC+EFV", "Other"] # Example options
user_input["Current Regimen"] = st.selectbox("Select ART Regimen", regimen_options)
user_input["Last VL Result Clean"] = st.number_input("Enter Last Viral Load Result", min_value=0.0, max_value=10000000.0, value=50.0)

# Add inputs for other features required by the model (based on cell 7ohF3S-NBrXi)
user_input["Weight"] = st.number_input("Enter Weight (kg)", min_value=0.0, max_value=200.0, value=65.0)
user_input["Height"] = st.number_input("Enter Height (cm)", min_value=0.0, max_value=250.0, value=170.0)
user_input["Months of Prescription"] = st.number_input("Enter Months of Prescription", min_value=0, max_value=36, value=6)


# Create DataFrame for prediction, mapping user_input keys to model's expected column names
# Note: The model trained in cell 7ohF3S-NBrXi did NOT include 'Sex'.
# We include the input field here as requested, but it's not used for prediction with the current model.
input_df = pd.DataFrame({
    "age": [user_input["Age at reporting"]], # Map 'Age at reporting' input to 'age' column
    "Weight": [user_input["Weight"]],
    "Height": [user_input["Height"]],
    "Months of Prescription": [user_input["Months of Prescription"]],
    "Current Regimen": [user_input["Current Regimen"]],
    "Last VL Result": [user_input["Last VL Result Clean"]] # Map 'Last VL Result Clean' input to 'Last VL Result' column
})

if st.button("Predict"):
    # Check for missing values in manual input (Streamlit inputs handle basic types, but check for NaN from number_input)
    if input_df.isnull().values.any():
         st.warning("Please ensure all input fields are filled.")
    else:
        try:
            pred = model.predict(input_df)[0]
            # If the model has predict_proba, show confidence
            if hasattr(model, 'predict_proba'):
                 prob = model.predict_proba(input_df)[0][pred]
                 label = "Suppressed" if pred == 1 else "Not Suppressed"
                 st.success(f"Prediction: {label} (Confidence: {prob:.2%})")
            else:
                 label = "Suppressed" if pred == 1 else "Not Suppressed"
                 st.success(f"Prediction: {label}")

        except Exception as e:
            st.error(f"Error making prediction: {e}")
            st.write("Please check the input values and ensure the model is trained correctly.")

