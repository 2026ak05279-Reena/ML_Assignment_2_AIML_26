import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Page Configurations for a professional look
st.set_page_config(
    page_title="Breast Cancer Model Evaluator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to inject professional green and teal styling
st.markdown("""
    <style>
    /* Main background color */
    .stApp {
        background-color: #f0fdf4 !important;
    }
    /* Custom styles for headers */
    h1 {
        color: #042f2e !important;
        font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;
        font-weight: 800;
    }
    h2, h3 {
        color: #0f766e !important;
        font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;
        font-weight: 600;
    }
    /* Metric adjustments */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border-left: 5px solid #0d9488;
        border-radius: 6px;
        padding: 10px 15px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
    }
    div[data-testid="stMetricValue"] {
        font-size: 26px;
        font-weight: 800;
        color: #0f766e !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 14px;
        color: #1e293b !important;
        font-weight: 600;
    }
    /* Sidebar Background adjustment */
    section[data-testid="stSidebar"] {
        background-color: #0f766e !important;
    }
    section[data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    /* Target specific file text styling to make test.csv / test_data.csv stand out */
    .highlight-text {
        color: #fef08a !important; /* Bright high-contrast yellow/gold */
        font-weight: bold;
        background-color: rgba(255, 255, 255, 0.1);
        padding: 2px 6px;
        border-radius: 4px;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar Section for app information and branding
with st.sidebar:
    st.markdown("# Diagnostic Hub")
    st.markdown("This workspace evaluates 5 pre-trained machine learning classifiers trained on the **Breast Cancer Wisconsin (Diagnostic) Dataset** containing 30 cellular features.")
    st.markdown("---")
    st.markdown("### Execution Quickstart")
    st.markdown("1. Upload your <span class='highlight-text'>test_data.csv</span> (or <span class='highlight-text'>test.csv</span>) evaluation dataset.", unsafe_allow_html=True)
    st.write("2. Select any model architecture from the list.")
    st.write("3. Inspect prediction scores, matrices, and metrics dynamically!")

# Header
st.title("Machine Learning Cancer Diagnostic Platform")
st.markdown("Deploy, visualize, and benchmark classification model architectures in real-time.")
st.markdown("---")

# 1. Dataset Upload Option
uploaded_file = st.file_uploader("Upload your evaluation CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Display data preview inside an expander
    with st.expander("Preview Uploaded Dataset Records", expanded=True):
        st.dataframe(df.head(), use_container_width=True)

    # Target split
    if 'target' in df.columns:
        X_test = df.drop(columns=['target'])
        y_test = df['target']

        # 2. Model selection dropdown
        st.subheader("Diagnostics Control Panel")
        model_option = st.selectbox(
            "Choose a trained ML Classifier architecture:",
            ("Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Support Vector Machine")
        )

        # Load the saved model and scaler
        model_filename = f"model/{model_option.lower().replace(' ', '_')}.joblib"
        if os.path.exists(model_filename):
            model = joblib.load(model_filename)

            # Apply Scaling if necessary
            if model_option in [
