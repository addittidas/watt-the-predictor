import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set page configuration
st.markdown("""
    <h1 style='text-align:center;'>Watt the Predictor<br>
    Energy Consumption Predictor</h1>
""", unsafe_allow_html=True)

st.markdown("<p style='text-align:center; font-size:16px;'>Compare and test different machine learning models to predict energy usage in buildings.</p>", unsafe_allow_html=True)
st.markdown("---")

# Custom CSS for better aesthetics
st.markdown("""
    <style>
    .main {
        padding: 10px;
    }
    h1, h2, h3 {
        color: #0e4d92;
    }
    .stButton > button {
        background-color: #0073e6;
        color: white;
        font-weight: bold;
    }
    .stDownloadButton > button {
        background-color: #28a745;
        color: white;
        font-weight: bold;
    }
    .stRadio > div {
        padding: 10px 0px;
    }
    </style>
""", unsafe_allow_html=True)

# Load available models
MODEL_DIR = "models"
model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")]

# Manually maps filenames to display names
model_display_map = {
    "linear_regression_model.pkl": "Linear Regression",
    "ridge_model.pkl": "Ridge Regression",
    "lasso_model.pkl": "Lasso Regression",
    "elastic_net_model.pkl": "ElasticNet Regression",
    "decision_tree_model.pkl": "Decision Tree (GridSearch)",
    "sgd_regressor_model.pkl": "SGD Regressor"
}

# Filter model files that exist and are mapped
available_model_files = [f for f in model_files if f in model_display_map]
display_names = [model_display_map[f] for f in available_model_files]

# Dropdown with friendly names
selected_display_name = st.selectbox("Choose a model", display_names)

# Get corresponding filename
selected_model_filename = [f for f, name in model_display_map.items() if name == selected_display_name][0]
selected_model_path = os.path.join(MODEL_DIR, selected_model_filename)

# Load the model
model = joblib.load(selected_model_path)

model_descriptions = {
    "Linear Regression": "Fits a straight line to minimize the squared difference between predictions and actual values.",
    "Ridge Regression": "Linear regression with L2 regularization to prevent overfitting.",
    "Lasso Regression": "Linear regression with L1 regularization, which can shrink coefficients to zero.",
    "ElasticNet Regression": "Combines L1 and L2 regularization for balanced performance.",
    "Decision Tree (GridSearch)": "Creates a tree structure of decisions based on features. Tuned using GridSearchCV.",
    "SGD Regressor": "Uses stochastic gradient descent to minimize loss. Efficient for large datasets."
}

model_metrics = {
    "Linear Regression": {
        "RMSLE": 1.3812, "RMSE": 1.3812, "MAE": 1.0202, "R²": 0.3923
    },
    "ElasticNet Regression": {
        "RMSLE": 1.4732, "RMSE": 1.4732, "MAE": 1.1241, "R²": 0.3087
    },
    "Ridge Regression": {
        "RMSLE": 1.3828, "RMSE": 1.3828, "MAE": 1.0225, "R²": 0.3910
    },
    "Lasso Regression": {
        "RMSLE": 1.3812, "RMSE": 1.3812, "MAE": 1.0203, "R²": 0.3923
    },
    "Decision Tree (GridSearch)": {
        "RMSLE": 0.6099, "RMSE": 0.6099, "MAE": 0.3875, "R²": 0.8815
    },
    "SGD Regressor": {
        "RMSLE": 1.3812, "RMSE": 1.3812, "MAE": 1.0202, "R²": 0.3923
    }
}

st.info(f"**Model Description:** {model_descriptions.get(selected_display_name)}")
st.markdown("<br>", unsafe_allow_html=True)

# Show evaluation metrics in card-style layout
metrics = model_metrics.get(selected_display_name)
if metrics:
    st.subheader("📊 Evaluation Metrics")

    card_style = """
        <div style="background-color:#f8f9fa; padding:16px; border-radius:10px; 
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1); margin-bottom:10px; text-align:center;">
            <h4 style="margin:0; color:#4a4a4a;">{}</h4>
            <p style="font-size:24px; font-weight:bold; color:#0073e6; margin:0;">{}</p>
        </div>
    """

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(card_style.format("RMSLE", f"{metrics['RMSLE']:.4f}"), unsafe_allow_html=True)
    with col2:
        st.markdown(card_style.format("RMSE", f"{metrics['RMSE']:.4f}"), unsafe_allow_html=True)
    with col3:
        st.markdown(card_style.format("MAE", f"{metrics['MAE']:.4f}"), unsafe_allow_html=True)
    with col4:
        r2_value = f"{metrics['R²']:.4f}" if metrics['R²'] is not None else "N/A"
        st.markdown(card_style.format("R²", r2_value), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

if hasattr(model, "feature_importances_"):
    st.subheader("🔍 Feature Importance")
    importances = model.feature_importances_
    features = ['building_id', 'square_feet', 'primary_use', 'meter',
                'air_temperature', 'dayofyear', 'hour', 'isDayTime', 'dayofweek']
    imp_df = pd.DataFrame({"Feature": features, "Importance": importances})
    imp_df = imp_df.sort_values(by="Importance", ascending=False)

    st.bar_chart(imp_df.set_index("Feature"))

st.markdown("---")

# Custom styled heading with icon and new title
st.markdown("""
    <div style='display: flex; align-items: center; gap: 10px; margin-bottom: 10px;'>
        <h1 style='color: #0e4d92; margin: 0;'>Energy Usage Forecast</h1>
    </div>
    <p style='margin-top: -10px; font-size: 16px; color: #444;'>Predict building energy usage using different trained machine learning models.</p>
""", unsafe_allow_html=True)

# Input method selection (side by side)
col1, col2 = st.columns([1, 3])
with col1:
    st.write("**Select input method**")
with col2:
    input_method = st.radio("", ["Manual Input", "Upload CSV"], horizontal=True)

# Define feature input
def get_manual_input():
    # Styled header for the form
    st.markdown("""
        <div style='background-color:#f0f4f8; padding:20px 25px; border-radius:8px; 
                    border-left: 5px solid #0e4d92; margin-bottom:20px;'>
            <h3 style='color:#0e4d92; margin-bottom:8px;'>📝 Manual Input Form</h3>
            <p style='color:#333;'>Please enter the details below to predict the energy usage for a building.</p>
        </div>
    """, unsafe_allow_html=True)

    building_id = st.number_input("Building ID", min_value=0, step=1)
    square_feet = st.number_input("Square Feet", min_value=0.0)
    primary_use = st.selectbox("Primary Use", ['Education', 'Office', 'Lodging/residential', 'Entertainment/public assembly', 'Other'])
    meter = st.selectbox("Meter Type", [0, 1, 2, 3], format_func=lambda x: ['Electricity', 'Chilled Water', 'Steam', 'Hot Water'][x])
    air_temperature = st.number_input("Air Temperature (°C)")
    dayofyear = st.slider("Day of Year", 1, 366)
    hour = st.slider("Hour of Day", 0, 23)
    isDayTime = st.radio("Is Daytime?", [0, 1])
    dayofweek = st.slider("Day of Week", 0, 6)
    primary_use_mapping = {
        'Education': 0,
        'Office': 1,
        'Lodging/residential': 2,
        'Entertainment/public assembly': 3,
        'Other': 4
    }

    primary_use_encoded = primary_use_mapping.get(primary_use, 4)

    input_data = pd.DataFrame([[
        building_id, square_feet, primary_use_encoded, meter,
        air_temperature, dayofyear, hour, isDayTime, dayofweek
    ]], columns=[
        'building_id', 'square_feet', 'primary_use', 'meter',
        'air_temperature', 'dayofyear', 'hour', 'isDayTime', 'dayofweek'
    ])

    return input_data

def get_csv_input():
    st.markdown("""
            <div style='background-color: #f0f4f8; padding: 20px; border-radius: 10px; border-left: 6px solid #0e4d92;'>
                <h3 style='color: #0e4d92;'>📁 Upload CSV File</h3>
                <p>Please upload your CSV file containing building data to get predictions.</p>
            </div>
        """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(" ", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(df.head())
        return df
    return None

# Make prediction
input_df = None

if input_method == "Manual Input":
    input_df = get_manual_input()
elif input_method == "Upload CSV":
    input_df = get_csv_input()

if input_df is not None and st.button("Predict Energy Usage"):
    try:
        predictions = model.predict(input_df)
        st.success("✅ Prediction complete!")

        if len(predictions) == 1:
            st.markdown(f"**Predicted Energy Usage:** `{predictions[0]:.2f}` units")
        else:
            st.subheader("📈 Predictions")
            input_df["Predicted Usage"] = predictions
            st.dataframe(input_df)
            csv = input_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions CSV", data=csv, file_name="predictions.csv", mime="text/csv")

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")