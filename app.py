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
    </style>
""", unsafe_allow_html=True)

# Sidebar Section for app information and branding
with st.sidebar:
    st.markdown("# Diagnostic Hub")
    st.markdown("This workspace evaluates 5 pre-trained machine learning classifiers trained on the **Breast Cancer Wisconsin (Diagnostic) Dataset** containing 30 cellular features.")
    st.markdown("---")
    st.markdown("### Execution Quickstart")
    st.write("1. Upload your `test_data.csv` evaluation dataset.")
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
            if model_option in ["Logistic Regression", "KNN", "Support Vector Machine"]:
                scaler = joblib.load("model/scaler.joblib")
                X_test_processed = scaler.transform(X_test)
            else:
                X_test_processed = X_test

            # Predict
            preds = model.predict(X_test_processed)
            probs = model.predict_proba(X_test_processed)[:, 1] if hasattr(model, "predict_proba") else None

            st.markdown("---")
            st.subheader(f"Active Performance Metrics Dashboard: {model_option}")

            # 3. Display evaluation metrics beautifully in structured columns
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            with col1:
                st.metric("Accuracy", f"{accuracy_score(y_test, preds):.4f}")
            with col2:
                if probs is not None:
                    st.metric("AUC Score", f"{roc_auc_score(y_test, probs):.4f}")
                else:
                    st.metric("AUC Score", "N/A")
            with col3:
                st.metric("MCC Score", f"{matthews_corrcoef(y_test, preds):.4f}")
            with col4:
                st.metric("Precision", f"{precision_score(y_test, preds):.4f}")
            with col5:
                st.metric("Recall", f"{recall_score(y_test, preds):.4f}")
            with col6:
                st.metric("F1 Score", f"{f1_score(y_test, preds):.4f}")

            st.markdown("---")

            # Layout for Visualizations
            vis_col1, vis_col2 = st.columns([1, 1])
            
            with vis_col1:
                st.subheader("Confusion Matrix Heatmap")
                cm = confusion_matrix(y_test, preds)
                fig, ax = plt.subplots(figsize=(5, 4))
                # Custom teal/green colormap
                sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", ax=ax, cbar=False,
                            xticklabels=['Malignant', 'Benign'], yticklabels=['Malignant', 'Benign'])
                plt.ylabel('Actual Label')
                plt.xlabel('Predicted Label')
                plt.tight_layout()
                st.pyplot(fig)

            with vis_col2:
                st.subheader("Classification Report Summary")
                st.markdown("Statistical performance indicators categorized by target classes:")
                
                # Generate report as a dictionary to tabulate
                report_dict = classification_report(y_test, preds, target_names=['Malignant', 'Benign'], output_dict=True)
                report_df = pd.DataFrame(report_dict).transpose().round(4)
                st.dataframe(report_df, use_container_width=True)
        else:
            st.error(f"Model file '{model_filename}' not found. Please verify that your trained models are uploaded inside the 'model/' directory.")
    else:
        st.error("The uploaded test CSV must contain a 'target' column.")
