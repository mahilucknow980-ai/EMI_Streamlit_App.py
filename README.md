# EMI Prediction System

A Machine Learning based EMI Prediction and Loan Risk Analysis application built with Python, Scikit-learn, XGBoost, and Streamlit.

## Features

- Predicts Maximum Monthly EMI using XGBoost Regression
- Loan risk/eligibility analysis using Random Forest and XGBoost classifiers
- Displays Estimated Requested EMI
- Displays Available Monthly Income
- Shows Debt-to-Income and Expense-to-Income ratios
- Shows Credit Score
- Shows Emergency Fund Coverage
- Provides financial indicators in a Streamlit dashboard

## Machine Learning Models

### Regression
- XGBoost Regressor
- Target: `max_monthly_emi`
- Verified transformed input: 74 features

### Classification
- Random Forest Classifier
- XGBoost Classifier
- Classification preprocessor: Scikit-learn ColumnTransformer
- 43 application inputs are transformed into 74 model features

## Dataset

The engineered EMI dataset contains 404,342 records and 45 columns.

Important engineered features include:

- `total_monthly_expenses`
- `total_monthly_obligations`
- `debt_to_income_ratio`
- `expense_to_income_ratio`
- `obligation_to_income_ratio`
- `available_monthly_income`
- `affordability_ratio`
- `estimated_requested_emi`
- `loan_to_income_ratio`
- `projected_emi_burden`
- `projected_emi_to_income_ratio`
- `emergency_fund_coverage_months`
- `bank_balance_to_salary_ratio`
- `employment_stability_score`
- `credit_score_normalized`
- `credit_risk_level`
- `dependent_ratio`
- `financial_stress_indicator`

## Project Structure

```text
EMI/
├── streamlit_app/
│   ├── app.py
│   ├── best_emi_regression_model.pkl
│   ├── classification_preprocessor.pkl
│   ├── random_forest_classifier.pkl
│   ├── xgboost_classifier.pkl
│   └── xgb_label_encoder.pkl
├── EMI_dataset_feature_engineered.csv
├── final_feature_importance.csv
├── requirements.txt
└── README.md
```

## Installation

Open Anaconda Prompt and move to the Streamlit application folder:

```bash
cd "C:\Users\MAHESH KUMAR\OneDrive\Desktop\EMI\streamlit_app"
```

Install the required packages:

```bash
pip install -r ..\requirements.txt
```

If `requirements.txt` is copied into the `streamlit_app` folder, use:

```bash
pip install -r requirements.txt
```

## Run the Application

From the folder containing `app.py`:

```bash
streamlit run app.py
```

The application normally opens at:

```text
http://localhost:8501
```

## Required Model Files

The following trained files must be available to `app.py`:

```text
best_emi_regression_model.pkl
classification_preprocessor.pkl
random_forest_classifier.pkl
xgboost_classifier.pkl
xgb_label_encoder.pkl
```

## Classification Input Columns

The verified classification preprocessor expects these 43 columns:

```text
age
gender
marital_status
education
monthly_salary
employment_type
years_of_employment
company_type
house_type
monthly_rent
family_size
dependents
school_fees
college_fees
travel_expenses
groceries_utilities
other_monthly_expenses
existing_loans
current_emi_amount
credit_score
bank_balance
emergency_fund
emi_scenario
requested_amount
requested_tenure
total_monthly_expenses
total_monthly_obligations
debt_to_income_ratio
expense_to_income_ratio
obligation_to_income_ratio
available_monthly_income
affordability_ratio
estimated_requested_emi
loan_to_income_ratio
projected_emi_burden
projected_emi_to_income_ratio
emergency_fund_coverage_months
bank_balance_to_salary_ratio
employment_stability_score
credit_score_normalized
credit_risk_level
dependent_ratio
financial_stress_indicator
```

The preprocessing pipeline converts these 43 inputs into exactly 74 transformed features.

## Verified Regression Compatibility

The regression pipeline was tested successfully.

Example test predictions:

```text
Actual:
500.00
700.00
27775.00
16170.00
500.00

Predicted:
408.16
705.01
28376.04
16465.09
733.01
```

This confirms that the preprocessing output and XGBoost regression model are compatible.

## Application Workflow

```text
User Input
    ↓
Feature Engineering
    ↓
Classification Preprocessor
    ↓
74 Transformed Features
    ↓
 ┌───────────────────────┐
 │ XGBoost Regression    │
 │ Maximum Monthly EMI   │
 └───────────────────────┘
    ↓
 ┌───────────────────────┐
 │ Random Forest         │
 │ Loan Risk Analysis    │
 └───────────────────────┘
    ↓
 ┌───────────────────────┐
 │ XGBoost Classification│
 │ Loan Risk Analysis    │
 └───────────────────────┘
    ↓
Streamlit Dashboard
```

## Technologies

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Joblib
- Matplotlib
- Jupyter Notebook

## Feature Importance

The project includes:

```text
final_feature_importance.csv
```

which stores the generated feature-importance information.

## Important Note

This application is an educational Machine Learning prediction system. Its predictions are not guaranteed bank approval decisions and should not be treated as professional financial advice.

Do not upload real personal financial information or sensitive datasets to a public repository.

## Future Improvements

- Prediction history
- Downloadable prediction report
- Interactive charts
- SHAP explainability
- Authentication
- Cloud deployment
- Model monitoring
- Automated retraining

## Author

**Mahesh Kumar**

B.Tech - Computer Science and Engineering  
Artificial Intelligence & Data Science

## License

Educational and project demonstration use.
