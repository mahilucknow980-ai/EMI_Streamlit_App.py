
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="EMI Prediction System",
    page_icon="💰",
    layout="wide"
)


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


REGRESSION_MODEL_PATH = (
    BASE_DIR / "best_emi_regression_model.pkl"
)

PREPROCESSOR_PATH = (
    BASE_DIR / "classification_preprocessor.pkl"
)

RF_MODEL_PATH = (
    BASE_DIR / "random_forest_classifier.pkl"
)

XGB_MODEL_PATH = (
    BASE_DIR / "xgboost_classifier.pkl"
)

LABEL_ENCODER_PATH = (
    BASE_DIR / "xgb_label_encoder.pkl"
)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    regression_model = joblib.load(
        REGRESSION_MODEL_PATH
    )

    preprocessor = joblib.load(
        PREPROCESSOR_PATH
    )

    rf_model = joblib.load(
        RF_MODEL_PATH
    )

    xgb_model = joblib.load(
        XGB_MODEL_PATH
    )

    label_encoder = joblib.load(
        LABEL_ENCODER_PATH
    )

    return (
        regression_model,
        preprocessor,
        rf_model,
        xgb_model,
        label_encoder
    )


try:

    (
        regression_model,
        preprocessor,
        rf_model,
        xgb_model,
        label_encoder
    ) = load_models()

    models_loaded = True

except Exception as e:

    models_loaded = False

    st.error(
        "Model loading failed."
    )

    st.exception(e)


# ============================================================
# TITLE
# ============================================================

st.title(
    "💰 EMI Prediction & Financial Risk Analysis"
)

st.markdown(
    """
    ### AI-Powered EMI Decision Support System

    Enter the applicant's financial information below to estimate
    the maximum affordable monthly EMI and analyse loan eligibility.
    """
)


if not models_loaded:

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "Application Information"
)

st.sidebar.info(
    """
    This application uses trained Machine Learning models
    developed for the EMI Prediction Project.
    """
)

st.sidebar.markdown(
    """
    **Models**

    • XGBoost Regression  
    • Random Forest Classification  
    • XGBoost Classification
    """
)


# ============================================================
# PERSONAL INFORMATION
# ============================================================

st.header(
    "👤 Personal Information"
)

col1, col2, col3 = st.columns(3)


with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=80,
        value=30,
        step=1
    )


with col2:

    gender = st.selectbox(
        "Gender",
        [
            "Male",
            "Female"
        ]
    )


with col3:

    marital_status = st.selectbox(
        "Marital Status",
        [
            "Single",
            "Married"
        ]
    )


col1, col2, col3 = st.columns(3)


with col1:

    education = st.selectbox(
        "Education",
        [
            "High School",
            "Graduate",
            "Post Graduate",
            "Professional"
        ]
    )


with col2:

    family_size = st.number_input(
        "Family Size",
        min_value=1,
        max_value=20,
        value=4,
        step=1
    )


with col3:

    dependents = st.number_input(
        "Dependents",
        min_value=0,
        max_value=15,
        value=2,
        step=1
    )


# ============================================================
# EMPLOYMENT
# ============================================================

st.header(
    "💼 Employment Information"
)

col1, col2, col3 = st.columns(3)


with col1:

    monthly_salary = st.number_input(
        "Monthly Salary (₹)",
        min_value=5000.0,
        max_value=10000000.0,
        value=50000.0,
        step=1000.0
    )


with col2:

    employment_type = st.selectbox(
        "Employment Type",
        [
            "Private",
            "Government",
            "Self-employed"
        ]
    )


with col3:

    years_of_employment = st.number_input(
        "Years of Employment",
        min_value=0.0,
        max_value=50.0,
        value=5.0,
        step=0.5
    )


col1, col2 = st.columns(2)


with col1:

    company_type = st.selectbox(
        "Company Type",
        [
            "MNC",
            "Large Indian",
            "Mid-size",
            "Small",
            "Startup"
        ]
    )


with col2:

    house_type = st.selectbox(
        "House Type",
        [
            "Own",
            "Rented",
            "Family"
        ]
    )


# ============================================================
# MONTHLY EXPENSES
# ============================================================

st.header(
    "🏠 Monthly Expenses"
)

col1, col2, col3 = st.columns(3)


with col1:

    monthly_rent = st.number_input(
        "Monthly Rent (₹)",
        min_value=0.0,
        max_value=1000000.0,
        value=10000.0,
        step=500.0
    )


with col2:

    school_fees = st.number_input(
        "School Fees (₹)",
        min_value=0.0,
        max_value=500000.0,
        value=3000.0,
        step=500.0
    )


with col3:

    college_fees = st.number_input(
        "College Fees (₹)",
        min_value=0.0,
        max_value=500000.0,
        value=0.0,
        step=500.0
    )


col1, col2, col3 = st.columns(3)


with col1:

    travel_expenses = st.number_input(
        "Travel Expenses (₹)",
        min_value=0.0,
        max_value=500000.0,
        value=3000.0,
        step=500.0
    )


with col2:

    groceries_utilities = st.number_input(
        "Groceries & Utilities (₹)",
        min_value=0.0,
        max_value=500000.0,
        value=8000.0,
        step=500.0
    )


with col3:

    other_monthly_expenses = st.number_input(
        "Other Monthly Expenses (₹)",
        min_value=0.0,
        max_value=500000.0,
        value=3000.0,
        step=500.0
    )


# ============================================================
# EXISTING FINANCIAL INFORMATION
# ============================================================

st.header(
    "💳 Existing Financial Information"
)

col1, col2, col3 = st.columns(3)


with col1:

    existing_loans = st.selectbox(
        "Existing Loans",
        [
            "No",
            "Yes"
        ]
    )


with col2:

    current_emi_amount = st.number_input(
        "Current EMI Amount (₹)",
        min_value=0.0,
        max_value=1000000.0,
        value=0.0,
        step=500.0
    )


with col3:

    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=900,
        value=750,
        step=1
    )


col1, col2 = st.columns(2)


with col1:

    bank_balance = st.number_input(
        "Bank Balance (₹)",
        min_value=0.0,
        max_value=100000000.0,
        value=100000.0,
        step=5000.0
    )


with col2:

    emergency_fund = st.number_input(
        "Emergency Fund (₹)",
        min_value=0.0,
        max_value=100000000.0,
        value=100000.0,
        step=5000.0
    )


# ============================================================
# LOAN REQUEST
# ============================================================

st.header(
    "🏦 Loan / EMI Request"
)

col1, col2, col3 = st.columns(3)


with col1:

    emi_scenario = st.selectbox(
        "EMI Scenario",
        [
            "E-commerce Shopping EMI",
            "Education EMI",
            "Home Appliances EMI",
            "Personal Loan EMI",
            "Vehicle EMI"
        ]
    )


with col2:

    requested_amount = st.number_input(
        "Requested Amount (₹)",
        min_value=1000.0,
        max_value=10000000.0,
        value=100000.0,
        step=5000.0
    )


with col3:

    requested_tenure = st.number_input(
        "Requested Tenure (Months)",
        min_value=3,
        max_value=120,
        value=24,
        step=1
    )


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def calculate_features():

    total_monthly_expenses = (
        monthly_rent
        + school_fees
        + college_fees
        + travel_expenses
        + groceries_utilities
        + other_monthly_expenses
    )

    total_monthly_obligations = (
        total_monthly_expenses
        + current_emi_amount
    )

    safe_salary = max(
        monthly_salary,
        1.0
    )

    debt_to_income_ratio = (
        current_emi_amount
        / safe_salary
    )

    expense_to_income_ratio = (
        total_monthly_expenses
        / safe_salary
    )

    obligation_to_income_ratio = (
        total_monthly_obligations
        / safe_salary
    )

    available_monthly_income = (
        monthly_salary
        - total_monthly_obligations
    )

    affordability_ratio = (
        available_monthly_income
        / safe_salary
    )

    # Approximate EMI calculation.
    # This is used only to reproduce the engineered input
    # structure required by the trained model.

    annual_interest_rate = 0.12

    monthly_rate = (
        annual_interest_rate / 12
    )

    if monthly_rate > 0:

        estimated_requested_emi = (
            requested_amount
            * monthly_rate
            * (1 + monthly_rate) ** requested_tenure
            /
            (
                (1 + monthly_rate) ** requested_tenure
                - 1
            )
        )

    else:

        estimated_requested_emi = (
            requested_amount
            / requested_tenure
        )

    loan_to_income_ratio = (
        requested_amount
        / max(monthly_salary * 12, 1.0)
    )

    projected_emi_burden = (
        current_emi_amount
        + estimated_requested_emi
    )

    projected_emi_to_income_ratio = (
        projected_emi_burden
        / safe_salary
    )

    emergency_fund_coverage_months = (
        emergency_fund
        / max(
            total_monthly_expenses,
            1.0
        )
    )

    bank_balance_to_salary_ratio = (
        bank_balance
        / safe_salary
    )

    employment_stability_score = min(
        years_of_employment / 10.0,
        1.0
    )

    credit_score_normalized = (
        (credit_score - 300)
        / 600
    )

    dependent_ratio = (
        dependents
        / max(family_size, 1)
    )

    # --------------------------------------------------------
    # Credit risk
    # --------------------------------------------------------

    if credit_score >= 750:

        credit_risk_level = "Very_Low_Risk"

    elif credit_score >= 700:

        credit_risk_level = "Low_Risk"

    elif credit_score >= 650:

        credit_risk_level = "Moderate_Risk"

    elif credit_score >= 600:

        credit_risk_level = "High_Risk"

    else:

        credit_risk_level = "Very_High_Risk"

    # --------------------------------------------------------
    # Financial stress
    # --------------------------------------------------------

    stress_ratio = (
        total_monthly_obligations
        / safe_salary
    )

    if stress_ratio < 0.30:

        financial_stress_indicator = "Low_Stress"

    elif stress_ratio < 0.50:

        financial_stress_indicator = "Moderate_Stress"

    elif stress_ratio < 0.70:

        financial_stress_indicator = "High_Stress"

    else:

        financial_stress_indicator = "Very_High_Stress"

    return {

        "age": age,
        "gender": gender,
        "marital_status": marital_status,
        "education": education,
        "monthly_salary": monthly_salary,
        "employment_type": employment_type,
        "years_of_employment": years_of_employment,
        "company_type": company_type,
        "house_type": house_type,
        "monthly_rent": monthly_rent,
        "family_size": family_size,
        "dependents": dependents,
        "school_fees": school_fees,
        "college_fees": college_fees,
        "travel_expenses": travel_expenses,
        "groceries_utilities": groceries_utilities,
        "other_monthly_expenses": other_monthly_expenses,
        "existing_loans": existing_loans,
        "current_emi_amount": current_emi_amount,
        "credit_score": credit_score,
        "bank_balance": bank_balance,
        "emergency_fund": emergency_fund,
        "emi_scenario": emi_scenario,
        "requested_amount": requested_amount,
        "requested_tenure": requested_tenure,

        "total_monthly_expenses":
            total_monthly_expenses,

        "total_monthly_obligations":
            total_monthly_obligations,

        "debt_to_income_ratio":
            debt_to_income_ratio,

        "expense_to_income_ratio":
            expense_to_income_ratio,

        "obligation_to_income_ratio":
            obligation_to_income_ratio,

        "available_monthly_income":
            available_monthly_income,

        "affordability_ratio":
            affordability_ratio,

        "estimated_requested_emi":
            estimated_requested_emi,

        "loan_to_income_ratio":
            loan_to_income_ratio,

        "projected_emi_burden":
            projected_emi_burden,

        "projected_emi_to_income_ratio":
            projected_emi_to_income_ratio,

        "emergency_fund_coverage_months":
            emergency_fund_coverage_months,

        "bank_balance_to_salary_ratio":
            bank_balance_to_salary_ratio,

        "employment_stability_score":
            employment_stability_score,

        "credit_score_normalized":
            credit_score_normalized,

        "credit_risk_level":
            credit_risk_level,

        "dependent_ratio":
            dependent_ratio,

        "financial_stress_indicator":
            financial_stress_indicator
    }


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()

predict_button = st.button(
    "🚀 Predict EMI & Analyse Risk",
    type="primary",
    use_container_width=True
)


if predict_button:

    try:

        # ----------------------------------------------------
        # Create input
        # ----------------------------------------------------

        input_data = calculate_features()

        input_df = pd.DataFrame(
            [input_data]
        )

        # ----------------------------------------------------
        # Ensure exact preprocessor columns
        # ----------------------------------------------------

        required_columns = list(
            preprocessor.feature_names_in_
        )

        input_df = input_df[
            required_columns
        ]

        # ----------------------------------------------------
        # Transform
        # ----------------------------------------------------

        transformed_input = (
            preprocessor.transform(
                input_df
            )
        )

        # ----------------------------------------------------
        # Verify 74 features
        # ----------------------------------------------------

        if transformed_input.shape[1] != 74:

            st.error(
                "Unexpected transformed feature count."
            )

            st.write(
                "Expected: 74"
            )

            st.write(
                "Received:",
                transformed_input.shape[1]
            )

            st.stop()

        # ----------------------------------------------------
        # Regression
        # ----------------------------------------------------

        predicted_emi = (
            regression_model.predict(
                transformed_input
            )[0]
        )

        predicted_emi = max(
            float(predicted_emi),
            0.0
        )

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        rf_prediction = (
            rf_model.predict(
                transformed_input
            )[0]
        )

        xgb_prediction = (
            xgb_model.predict(
                transformed_input
            )[0]
        )

        # ----------------------------------------------------
        # Decode classification
        # ----------------------------------------------------

        try:

            rf_label = label_encoder.inverse_transform(
                [rf_prediction]
            )[0]

        except Exception:

            rf_label = str(
                rf_prediction
            )


        try:

            xgb_label = label_encoder.inverse_transform(
                [xgb_prediction]
            )[0]

        except Exception:

            xgb_label = str(
                xgb_prediction
            )

        # ----------------------------------------------------
        # Display prediction
        # ----------------------------------------------------

        st.success(
            "Prediction completed successfully!"
        )

        st.header(
            "📊 Prediction Results"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Predicted Maximum Monthly EMI",
                f"₹{predicted_emi:,.2f}"
            )

        with col2:

            st.metric(
                "Estimated Requested EMI",
                f"₹{input_data['estimated_requested_emi']:,.2f}"
            )

        with col3:

            st.metric(
                "Available Monthly Income",
                f"₹{input_data['available_monthly_income']:,.2f}"
            )

        # ----------------------------------------------------
        # Classification results
        # ----------------------------------------------------

        st.header(
            "🏦 Loan Risk Analysis"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.subheader(
                "Random Forest"
            )

            st.info(
                str(rf_label)
            )

        with col2:

            st.subheader(
                "XGBoost"
            )

            st.info(
                str(xgb_label)
            )

        # ----------------------------------------------------
        # Financial indicators
        # ----------------------------------------------------

        st.header(
            "📈 Financial Indicators"
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Debt-to-Income",
                f"{input_data['debt_to_income_ratio']:.2%}"
            )

        with col2:

            st.metric(
                "Expense-to-Income",
                f"{input_data['expense_to_income_ratio']:.2%}"
            )

        with col3:

            st.metric(
                "Credit Score",
                f"{credit_score}"
            )

        with col4:

            st.metric(
                "Emergency Coverage",
                f"{input_data['emergency_fund_coverage_months']:.1f} months"
            )

        # ----------------------------------------------------
        # User-friendly interpretation
        # ----------------------------------------------------

        st.header(
            "💡 Financial Interpretation"
        )

        if (
            predicted_emi
            <= input_data["available_monthly_income"]
        ):

            st.success(
                "The predicted maximum EMI is within "
                "the applicant's available monthly income."
            )

        else:

            st.warning(
                "The predicted maximum EMI is higher than "
                "the applicant's currently available monthly income."
            )

        if (
            input_data["debt_to_income_ratio"]
            <= 0.30
        ):

            st.success(
                "Existing EMI burden is relatively low."
            )

        elif (
            input_data["debt_to_income_ratio"]
            <= 0.50
        ):

            st.warning(
                "Existing EMI burden is moderate."
            )

        else:

            st.error(
                "Existing EMI burden is high."
            )

        # ----------------------------------------------------
        # Technical information
        # ----------------------------------------------------

        with st.expander(
            "🔍 Technical Model Information"
        ):

            st.write(
                "Input features:",
                len(required_columns)
            )

            st.write(
                "Transformed features:",
                transformed_input.shape[1]
            )

            st.write(
                "Regression model:",
                type(regression_model).__name__
            )

            st.write(
                "Random Forest:",
                type(rf_model).__name__
            )

            st.write(
                "XGBoost:",
                type(xgb_model).__name__
            )

    except Exception as e:

        st.error(
            "Prediction failed."
        )

        st.exception(e)
