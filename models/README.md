**Problem Statement**
Predicting whether a breast cancer tumor is Malignant (0) or Benign (1) based on computed nuclei features from a digitized image of a fine needle aspirate (FNA) of a breast mass.

**Dataset Description**
Name: Breast Cancer Wisconsin (Diagnostic) Dataset
Instances: 569
Features: 30 numeric (e.g., radius, texture, perimeter, area, smoothness)
Target: Binary classification (Malignant / Benign)

**Models Comparison Table**
Based on your model execution in this notebook, here is your completed metrics table:

ML Model Name	Accuracy	AUC	Precision	Recall	F1	MCC
Logistic Regression	0.9825	0.9954	0.9861	0.9861	0.9861	0.9623
Decision Tree	0.9123	0.9157	0.9559	0.9028	0.9286	0.8174
KNN	0.9561	0.9788	0.9589	0.9722	0.9655	0.9054
Naive Bayes	0.9386	0.9878	0.9452	0.9583	0.9517	0.8676
Support Vector Machine	0.9825	0.9950	0.9861	0.9861	0.9861	0.9623

**Performance Observations**
Logistic Regression & Support Vector Machine: These are the top performers with outstanding accuracy (98.25%) and MCC score (0.9623). Standardizing the features helped both models capture the linear/non-linear boundaries perfectly.
KNN & Naive Bayes: Good performance but slightly lower than Logistic Regression / SVM due to high dimensionality (30 features).
Decision Tree: Performed the worst (91.23% accuracy) because single decision trees tend to overfit small datasets with high feature counts.
Overall Winner: Logistic Regression (highest AUC score of 0.9954 and fastest inference speed).
