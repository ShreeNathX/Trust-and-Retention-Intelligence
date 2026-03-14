# FinTech Trust and Retention Intelligence

An end-to-end machine learning system for real-time customer churn prediction and credit risk scoring in the financial services domain. The project covers the full data science lifecycle from raw data to a deployed web application.

---

## Live Demo

The application is deployed on Streamlit Community Cloud and available at:

```
https://trust-and-retention-intelligence.streamlit.app
```

---

## Project Overview

Financial institutions lose significant revenue to customer churn. This project builds a production-grade churn prediction system using three real-world datasets, trained with industry-standard techniques, and deployed as an interactive web application.

| Item | Detail |
|---|---|
| Primary target | Customer churn (binary classification) |
| Secondary target | Credit risk scoring |
| Best model | XGBoost |
| Accuracy | 96.8% |
| AUC Score | 0.992 |
| F1 Score | 0.898 |
| Recall | 0.883 |
| Training rows | 10,127 |
| Features used | 24 (19 original + 5 engineered) |

---

## Datasets

| Dataset | Source | Rows | Purpose |
|---|---|---|---|
| BankChurners.csv | Kaggle - Credit Card Customers | 10,127 | Primary churn model |
| Churn_Modelling.csv | Kaggle - Predicting Churn for Bank Customers | 10,000 | Supporting analysis |
| german_credit_data.csv | Kaggle - German Credit | 1,000 | Credit risk analysis |

Dataset sources:

- https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers
- https://www.kaggle.com/datasets/adammaus/predicting-churn-for-bank-customers
- https://www.kaggle.com/datasets/uciml/german-credit

---

## Folder Structure

```
TRUST AND RETENTION INTELLIGENCE/
│
├── app.py                        # Streamlit web application
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
├── Model.ipynb                   # Full ML pipeline notebook
├── Links of data.txt             # Dataset source links
│
├── Data/
│   ├── BankChurners.csv
│   ├── Churn_Modelling.csv
│   └── german_credit_data.csv
│
├── models/
│   ├── churn_model.pkl           # Trained XGBoost model
│   ├── scaler.pkl                # StandardScaler
│   └── feature_names.pkl         # Ordered feature list
│
└── outputs/
    └── class_balance.png         # EDA output chart
```

---

## Pipeline — Model.ipynb

The notebook is structured in 10 sequential blocks:

| Block | Description |
|---|---|
| Block 1 | Load and inspect all three datasets |
| Block 2 | Data cleaning - nulls, duplicates, encoding targets |
| Block 3 | Feature engineering - 5 new business-driven features |
| Block 4 | Exploratory data analysis - 7 visualizations |
| Block 5 | Hypothesis validation - 5 business hypotheses tested |
| Block 6 | Preprocessing - label encoding, train/test split, StandardScaler, SMOTE |
| Block 7 | Model training - Logistic Regression, Random Forest, XGBoost with GridSearchCV |
| Block 8 | Model evaluation - confusion matrix, ROC curves, classification report |
| Block 9 | SHAP explainability - summary plots, beeswarm plots, waterfall charts for all 3 models |
| Block 10 | Business impact report - financial value in INR, actionable recommendations |

---

## Feature Engineering

Five new features were created from raw data to capture customer behaviour signals:

| Feature | Description |
|---|---|
| engagement_score | Weighted score combining transaction count, relationship count, and inactivity months |
| txn_decline_flag | Binary flag - 1 if spending declined more than 30% from Q1 to Q4 |
| high_contact_flag | Binary flag - 1 if customer contacted support 3 or more times in 12 months |
| revolving_ratio | Ratio of revolving balance to credit limit |
| utilization_bucket | Ordinal encoding of credit utilization into 4 bands |

---

## Hypothesis Validation

All five business hypotheses were confirmed before model training:

| Hypothesis | Result |
|---|---|
| Churned customers make fewer transactions | Confirmed |
| High contact frequency predicts churn | Confirmed |
| Longer inactivity periods predict churn | Confirmed |
| Declining spend from Q1 to Q4 predicts churn | Confirmed |
| Lower engagement score predicts churn | Confirmed |

---

## Model Results

| Model | Accuracy | F1 Score | AUC | Precision | Recall |
|---|---|---|---|---|---|
| Logistic Regression | 82.1% | 0.634 | 0.918 | 0.525 | 0.800 |
| Random Forest | 94.3% | 0.853 | 0.984 | 0.857 | 0.849 |
| XGBoost | 96.8% | 0.898 | 0.992 | 0.914 | 0.883 |

XGBoost was selected as the production model based on highest AUC and F1 scores.

---

## Techniques Applied

- SMOTE - oversampling to handle 84/16 class imbalance in training data
- Stratified K-Fold Cross Validation - 5 folds, prevents data leakage
- GridSearchCV - exhaustive hyperparameter tuning for XGBoost
- SHAP (SHapley Additive exPlanations) - model explainability for all three models
- StandardScaler - feature scaling applied consistently across train and test sets

---

## Web Application

The Streamlit app provides two tabs:

**Predict Churn**

Enter customer details across three sections — Identity, Profile, and Relationship — along with financial details including credit, transaction data, and utilization ratio. The app returns:

- Churn probability as a percentage
- Binary prediction - Will Churn or Will Stay
- Confidence score
- Risk level - CRITICAL, HIGH, MEDIUM, or LOW
- Colour-coded probability gauge
- Actionable business recommendation
- Risk signals detected from input values

**Model Performance**

Displays model accuracy metrics, a comparison table of all three models, dataset information, and a summary of all techniques applied.

---

## How to Run Locally

**Step 1 - Clone the repository**

```bash
git clone https://github.com/ShreeNathX/Trust-and-Retention-Intelligence.git
cd fintech-churn-app
```

**Step 2 - Install dependencies**

```bash
pip install streamlit pandas numpy joblib scikit-learn xgboost
```

**Step 3 - Run the pipeline notebook**

Open `Model.ipynb` in Jupyter and run all cells. This generates the three model files inside the `models/` folder.

**Step 4 - Launch the app**

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`

---

## How to Deploy on Streamlit Cloud

1. Push the repository to GitHub
2. Go to share.streamlit.io
3. Sign in with GitHub
4. Click New App
5. Set Repository to `https://github.com/ShreeNathX/Trust-and-Retention-Intelligence`, Branch to `main`, and Main file to `app.py`
6. Click Deploy

---

## Dependencies

```
streamlit
pandas
numpy
joblib
scikit-learn
xgboost
```

For the notebook only:

```
matplotlib
seaborn
imbalanced-learn
shap
```

---

## Business Impact

Based on the model performance on the test set, using an assumed annual customer value of INR 15,000 and a retention cost of INR 2,000 per customer:

- The model identifies 88.3% of actual churners
- Revenue protected per intervention cycle exceeds INR 4,000,000
- Net business value after retention campaign costs exceeds INR 3,500,000

---

## Author
**Shree Nath Mahato (Leader)**  
📧 Contact: [shreenath.ventures17@gmail.com](mailto:shreenath.ventures17@gmail.com).<br>
💬  For any queries, suggestions, or collaborations, feel free to reach out via email.

Built as a portfolio project demonstrating end-to-end data science skills including data engineering, exploratory analysis, machine learning, model explainability, and production deployment.