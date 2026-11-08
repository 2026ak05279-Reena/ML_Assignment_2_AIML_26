# Breast Cancer Wisconsin (Diagnostic) Classification

## a. Problem Statement
The objective of this project is to predict whether a breast cancer tumor is **Malignant (0)** or **Benign (1)** based on computed cell nuclei features extracted from a digitized image of a fine needle aspirate (FNA) of a breast mass.

---

## b. Dataset Description
* **Name:** Breast Cancer Wisconsin (Diagnostic) Dataset
* **Instances:** 569
* **Features:** 30 numeric features (e.g., radius, texture, perimeter, area, smoothness)
* **Target:** Binary classification (`0: Malignant`, `1: Benign`)

---

## c. Models Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.9825** | **0.9954** | **0.9861** | **0.9861** | **0.9861** | **0.9623** |
| **Decision Tree** | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |
| **KNN** | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| **Naive Bayes** | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| **Support Vector Machine** | **0.9825** | 0.9950 | **0.9861** | **0.9861** | **0.9861** | **0.9623** |

---

## d. Performance Observations

* **Logistic Regression & Support Vector Machine:** 
  These are the top performers with outstanding accuracy (**98.25%**) and MCC score (**0.9623**). Standardizing the features helped both models capture the linear/non-linear decision boundaries effectively.
* **KNN & Naive Bayes:** 
  Showed solid overall performance, but scored slightly lower than Logistic Regression / SVM due to the high dimensionality (30 features) of the dataset.
* **Decision Tree:** 
  Performed the worst among all evaluated algorithms (**91.23%** accuracy), as single decision trees tend to overfit relatively small datasets with high feature counts.
* **Overall Winner:** 
  **Logistic Regression**, as it achieved the highest AUC score (**0.9954**), tied for highest overall accuracy, and offers faster inference speed.
