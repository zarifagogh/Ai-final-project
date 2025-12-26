import streamlit as st
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler # Required for type hinting/loading

# --- Configuration ---
st.set_page_config(
    page_title="ML Final Project Deployment",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Define Paths ---
# NOTE: The Linear Regression file path is updated to include the scaler
LINEAR_MODEL_PATH = 'MyLinearRegression_theta_and_scaler.pkl' 
LOGISTIC_MODEL_PATH = 'MyLogisticRegression_theta.pkl'


# --- 1. Load Custom Models and Scalers (Use st.cache_data for performance) ---

@st.cache_resource
def load_linear_regression_data(path):
    """Loads the Linear Regression theta parameters AND the fitted StandardScaler."""
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data['theta'], data['scaler'] # Load both theta and scaler
    except FileNotFoundError:
        st.error(f"Error: Linear Regression model file not found at {path}. Make sure you ran the updated Phase I saving code.")
        return None, None

@st.cache_resource
def load_logistic_regression_data(path):
    """Loads the Logistic Regression theta parameters and the fitted StandardScaler."""
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data['theta'], data['scaler']
    except FileNotFoundError:
        st.error(f"Error: Logistic Regression model file not found at {path}. Run Phase II script first.")
        return None, None

# Load all resources at startup
LINEAR_THETA, LINEAR_SCALER = load_linear_regression_data(LINEAR_MODEL_PATH)
LOGISTIC_THETA, LOGISTIC_SCALER = load_logistic_regression_data(LOGISTIC_MODEL_PATH)


# --- 2. Custom Prediction Functions (Matches your NumPy logic) ---

def predict_linear(X_input_scaled, theta):
    """Performs Linear Regression prediction (h = X * theta)."""
    # Add intercept (column of ones)
    X_b = np.insert(X_input_scaled, 0, 1, axis=1).reshape(1, -1)
    # Prediction: Dot product
    return X_b.dot(theta)[0, 0]

def predict_logistic(X_input_scaled, theta):
    """Performs Logistic Regression prediction (Sigmoid(X * theta))."""
    X_b = np.insert(X_input_scaled, 0, 1, axis=1).reshape(1, -1)
    
    Z = X_b.dot(theta)
    h = 1 / (1 + np.exp(-np.clip(Z, -500, 500)))
    
    prediction = (h >= 0.5).astype(int)[0, 0]
    probability = h[0, 0]
    
    return prediction, probability

# --- 3. Streamlit Application Structure ---

st.title("💡 Final Project ML Pipeline Deployment")
st.markdown("### From Scratch Model Demo: Linear & Logistic Regression")

# Sidebar for Model Selection
st.sidebar.title("Project Track Selection")
model_choice = st.sidebar.selectbox(
    "Select Model to Deploy:",
    ("Real Estate Valuation (Regression)", "Breast Cancer Diagnosis (Classification)")
)
st.sidebar.markdown("---")

# --- Linear Regression UI ---
if model_choice == "Real Estate Valuation (Regression)" and LINEAR_THETA is not None:
    st.header("🏠 Real Estate Valuation (Linear Regression)")
    st.markdown("Predict the **Price per Unit Area** of a house based on its attributes.")
    
    features = ['HouseAge', 'DistanceToMRT', 'NumConvenienceStores', 'Latitude', 'Longitude']
    
    with st.form("linear_form"):
        # Set default values based on the scaler's training data mean for a sensible starting point
        # Fallback values are used if the scaler means are not available (e.g., if simulation data was too simple)
        try:
            default_values = [
                LINEAR_SCALER.mean_[0], LINEAR_SCALER.mean_[1], int(LINEAR_SCALER.mean_[2]), 
                LINEAR_SCALER.mean_[3], LINEAR_SCALER.mean_[4]
            ]
        except:
            default_values = [20.0, 800.0, 5, 24.985, 121.54]
            
        # Create input columns for features
        col1, col2, col3 = st.columns(3)
        
        with col1:
            house_age = st.number_input(f"{features[0]} (Years)", min_value=1.0, max_value=100.0, value=default_values[0], step=0.1)
            distance = st.number_input(f"{features[1]} (Meters to MRT)", min_value=10.0, max_value=5000.0, value=default_values[1], step=10.0)

        with col2:
            stores = st.number_input(f"{features[2]} (Count)", min_value=0, max_value=15, value=default_values[2], step=1)
            latitude = st.number_input(f"{features[3]} (Lat.)", format="%.5f", value=default_values[3], step=0.0001)
            
        with col3:
            longitude = st.number_input(f"{features[4]} (Lon.)", format="%.5f", value=default_values[4], step=0.0001)
            
        submitted = st.form_submit_button("Predict House Price")
        
        if submitted:
            # 1. Gather input data (must be a 2D array [1, n_features])
            input_data = np.array([[house_age, distance, stores, latitude, longitude]])
            
            # 2. SCALING (THE CRITICAL FIX)
            # Use the loaded scaler to transform the raw user input
            X_input_scaled = LINEAR_SCALER.transform(input_data)
            
            # 3. Predict
            prediction = predict_linear(X_input_scaled, LINEAR_THETA)
            
            st.success("✅ **Prediction Complete!**")
            
            if prediction < 0:
                 st.warning("⚠️ **NOTE:** The predicted price is negative. This happens because Linear Regression can extrapolate results far outside the trained range, especially if the feature combination is unusual.")
                 prediction_display = f"${prediction:,.2f}"
            else:
                prediction_display = f"${prediction:,.2f}"

            st.metric(
                label="Predicted Price per Unit Area (Target Y)",
                value=prediction_display
            )
            st.markdown("---")
            st.info(f"Input Data was successfully scaled before prediction using the saved training parameters.")


# --- Logistic Regression UI ---
elif model_choice == "Breast Cancer Diagnosis (Classification)" and LOGISTIC_THETA is not None:
    st.header("🦠 Breast Cancer Diagnosis (Logistic Regression)")
    st.markdown("Predicts the classification of a cell cluster (1 = Benign, 0 = Malignant).")
    
    # Using the first 10 features of the Breast Cancer dataset (for UI clarity)
    feature_names = [
        "mean radius", "mean texture", "mean perimeter", "mean area", 
        "mean smoothness", "mean compactness", "mean concavity", 
        "mean concave points", "mean symmetry", "mean fractal dimension"
    ]
    
    with st.expander("🔬 Input Mean Feature Values (First 10 of 30 Features)", expanded=True):
        input_values = {}
        cols = st.columns(5)
        
        for i, feature in enumerate(feature_names):
            input_values[feature] = cols[i % 5].number_input(
                feature.title(), 
                value=LOGISTIC_SCALER.mean_[i], 
                format="%.3f"
            )
            
    # Pad the remaining 20 features with zeros (or the mean from the training data)
    # Using the mean from the scaler for consistency with the training data distribution
    input_list = list(input_values.values()) + list(LOGISTIC_SCALER.mean_[10:])
    
    predict_log_button = st.button("Run Diagnosis", type="primary")
    
    if predict_log_button:
        # 1. Gather and format input data
        input_data = np.array([input_list])
        
        # 2. SCALING (CRITICAL STEP)
        X_input_scaled = LOGISTIC_SCALER.transform(input_data)
        
        # 3. Predict
        prediction, probability = predict_logistic(X_input_scaled, LOGISTIC_THETA)
        
        st.success("✅ **Diagnosis Complete!**")
        
        col_pred, col_prob = st.columns(2)
        
        if prediction == 1:
            col_pred.metric(
                label="Diagnosis Result",
                value="Benign (Non-Cancerous)",
                delta="Low Risk"
            )
            col_prob.metric(
                label="Probability (P=1)",
                value=f"{probability:.2%}"
            )
            st.balloons()
        else:
            col_pred.metric(
                label="Diagnosis Result",
                value="Malignant (Cancerous)",
                delta="High Risk",
                delta_color="inverse"
            )
            col_prob.metric(
                label="Probability (P=0)",
                value=f"{1 - probability:.2%}"
            )
            st.error("🚨 Please consult a medical professional for confirmation.")