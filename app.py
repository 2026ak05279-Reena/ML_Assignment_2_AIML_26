import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.title("ML Classification Model Evaluator")

# 1. Dataset Upload Option
uploaded_file = st.file_uploader("Upload your test CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:", df.head())
    
    # Target split
    if 'target' in df.columns:
        X_test = df.drop(columns=['target'])
        y_test = df['target']
        
        # 2. Model selection dropdown
        model_option = st.selectbox(
            "Choose a trained model",
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
            
            # 3. Display evaluation metrics
            st.subheader(f"Metrics for {model_option}")
            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", f"{accuracy_score(y_test, preds):.4f}")
            if probs is not None:
                col2.metric("AUC Score", f"{roc_auc_score(y_test, probs):.4f}")
            col3.metric("MCC Score", f"{matthews_corrcoef(y_test, preds):.4f}")
            
            # Extra metric displays
            col4, col5, col6 = st.columns(3)
            col4.metric("Precision", f"{precision_score(y_test, preds):.4f}")
            col5.metric("Recall", f"{recall_score(y_test, preds):.4f}")
            col6.metric("F1 Score", f"{f1_score(y_test, preds):.4f}")
            
            # 4. Confusion matrix and classification report
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, preds)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            st.pyplot(fig)
            
            st.subheader("Classification Report")
            st.text(classification_report(y_test, preds))
        else:
            st.error(f"Model file '{model_filename}' not found. Please make sure to commit your saved models inside the 'model/' directory.")
    else:
        st.error("The uploaded test CSV must contain a 'target' column.")
