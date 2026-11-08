Credit Default Model Explorer
============================

Problem statement
-----------------

Compare five machine-learning classifiers for predicting whether a credit
card client will default on payment next month. Provide an interactive
Streamlit application for evaluating saved models on labeled test data.

Dataset description
-------------------

Source: UCI Machine Learning Repository — Default of Credit Card Clients

Dataset link:
https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

The dataset contains 30,000 records and 23 input features covering credit
limit, demographic information, repayment history, bill amounts, and
previous payments.

The customer ID is excluded from training.

Target: default
- 0: No default
- 1: Default

Class distribution:
- No default: 23,364 records
- Default: 6,636 records

A stratified 80/20 split produces 24,000 training records and 6,000 test
records, using random_state=42.

X2, X3, and X4 are one-hot encoded. Remaining inputs are standardized.
Preprocessing is fitted only on training data and saved with each model
as a scikit-learn pipeline.

GitHub repository link
---------------------

https://github.com/2026ak05635-cmd/credit-default-classifier

Live Streamlit application
-------------------------

https://credit-default-classifier-ktswaqjj7szekf6n6x4zcu.streamlit.app/

Models used
-----------

- Logistic Regression
- Decision Tree Classifier
- K-Nearest Neighbors Classifier
- Gaussian Naive Bayes
- Support Vector Machine with an RBF kernel

Evaluation results
------------------

All models are evaluated on the same held-out test set.
Precision, recall, and F1 use default (1) as the positive class.
AUC uses positive-class probabilities, except for SVM, which uses
decision scores.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.8088 | 0.7100 | 0.6923 | 0.2442 | 0.3610 | 0.3302 |
| Decision Tree | 0.8143 | 0.7466 | 0.6611 | 0.3293 | 0.4396 | 0.3730 |
| KNN | 0.8087 | 0.7303 | 0.6228 | 0.3421 | 0.4416 | 0.3599 |
| Naive Bayes | 0.2903 | 0.7240 | 0.2340 | 0.9714 | 0.3771 | 0.1034 |
| Support Vector Machine | 0.8163 | 0.7202 | 0.6662 | 0.3399 | 0.4501 | 0.3824 |

Observations on model performance
---------------------------------

Add your own interpretation for each model below. Use the comparison
table and confusion matrices from your notebook.

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | Highest precision, 0.6923, but low recall, 0.2442. Its default predictions are relatively reliable, but it misses many actual defaults. |
| Decision Tree | 	Highest AUC, 0.7466, and accuracy 0.8143. Better default detection than Logistic Regression. |
| KNN | Recall 0.3421 and F1 0.4416, slightly higher than Decision Tree, but lower precision and MCC. |
| Naive Bayes | Highest recall, 0.9714, but 4,220 false positives and accuracy 0.2903. It identifies most defaults while incorrectly flagging many non-default clients. |
| Support Vector Machine | Highest accuracy 0.8163, F1 0.4501, and MCC 0.3824. Recall remains modest at 0.3399. |
| Overall winner for this dataset | SVM, if prioritizing accuracy, F1, and MCC on this test set. Decision Tree leads on AUC. Explain which metrics support your choice. |

Application features
--------------------

- Upload labeled test data in CSV format.
- Select one of five saved classifiers.
- Display Accuracy, AUC, Precision, Recall, F1, and MCC.
- Display a confusion matrix and classification report.
- Compare saved results from the original held-out test set.
- Download test data and model predictions.
- Evaluate uploaded data without retraining.

Uploaded CSV files must include X1 through X23 and the default column.

Project files
-------------

- app.py: Streamlit application.
- requirements.txt: Runtime dependencies.
- credit_default_training.ipynb: Training and evaluation notebook.
- test_data.csv: Held-out test features and labels.
- model_comparison.csv: Evaluation results for all five models.
- model/: Five saved model pipelines.

Running the application
-----------------------

Install dependencies:

    pip install -r requirements.txt

Start the application:

    streamlit run app.py

Limitations
-----------

The dataset is imbalanced. Accuracy alone is insufficient for comparing
default detection. Results are from one held-out split and do not
establish suitability for real-world lending decisions.
