import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1200px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        div[data-testid="stHorizontalBlock"]:has(
            button[data-testid="stBaseButton-secondary"]
        ) {
            align-items: center;
        }

        div[data-testid="stButton"] > button {
            border-radius: 8px;
            font-weight: 600;
            min-height: 2.5rem;
        }

        div[data-testid="stButton"] > button:hover {
            border-color: #2563eb;
            color: #2563eb;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "cleaned_telco_churn.csv"
MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"
COMPARISON_PATH = BASE_DIR / "models" / "model_comparison.csv"

# --------------------------------------------------
# LOAD DATA AND MODEL
# --------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    df = load_data()
    model = load_model()
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("Analyze customer churn and predict whether a customer is likely to leave.")

# --------------------------------------------------
# TOP WEB NAVIGATION
# --------------------------------------------------

pages = {
    "Dashboard": "🏠 Dashboard",
    "Exploratory Data Analysis": "📈 Analytics",
    "Model Comparison": "⚖️ Models",
    "Customer Prediction": "🔮 Predict"
}

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

nav_columns = st.columns([1.5, 1.3, 1.1, 1.1], gap="small")

for column, (page_name, label) in zip(nav_columns, pages.items()):
    with column:
        if st.button(
            label,
            key=f"nav_{page_name}",
            use_container_width=True,
            type="primary" if st.session_state.page == page_name else "secondary"
        ):
            st.session_state.page = page_name
            st.rerun()

page = st.session_state.page
st.divider()

# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

if page == "Dashboard":

    st.header("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    total_customers = len(df)

    if "Churn" in df.columns:
        churned_customers = (df["Churn"] == 1).sum()
        churn_rate = (churned_customers / total_customers) * 100
    else:
        churned_customers = 0
        churn_rate = 0

    col1.metric("Total Customers", total_customers)
    col2.metric("Churned Customers", churned_customers)
    col3.metric("Churn Rate", f"{churn_rate:.2f}%")
    col4.metric("Number of Features", df.shape[1])

    st.divider()

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True, height=300)

    st.subheader("Dataset Information")
    st.write(f"**Rows:** {df.shape[0]}")
    st.write(f"**Columns:** {df.shape[1]}")

    st.subheader("Missing Values")

    missing_values = df.isnull().sum()

    missing_df = pd.DataFrame({
        "Column": missing_values.index,
        "Missing Values": missing_values.values
    })

    st.dataframe(missing_df, use_container_width=True, height=300)

# --------------------------------------------------
# EDA
# --------------------------------------------------

elif page == "Exploratory Data Analysis":

    st.header("Exploratory Data Analysis")

    plots_dir = BASE_DIR / "plots"

    plot_files = {
        "Churn Distribution": "churn_distribution.png",
        "Churn by Contract": "churn_by_contract.png",
        "Churn by Internet Service": "churn_by_internet_service.png",
        "Churn by Online Security": "churn_by_online_security.png",
        "Churn by Partner": "churn_by_partner.png",
        "Churn by Payment Method": "churn_by_payment_method.png",
        "Churn by Senior Citizen": "churn_by_senior_citizen.png",
        "Churn by Tech Support": "churn_by_tech_support.png",
        "Churn by Tenure": "churn_by_tenure.png",
        "Correlation Heatmap": "correlation_heatmap.png",
        "Monthly Charges by Churn": "monthly_charges_by_churn.png",
        "Tenure by Churn": "tenure_by_churn.png",
        "Total Charges by Churn": "total_charges_by_churn.png"
    }

    available_plots = {
        name: plots_dir / filename
        for name, filename in plot_files.items()
        if (plots_dir / filename).exists()
    }

    if available_plots:

        selected_plot = st.selectbox(
            "Select a visualization",
            list(available_plots.keys())
        )

        chart_column = st.columns([0.1, 0.8, 0.1])[1]
        with chart_column:
            st.image(
                str(available_plots[selected_plot]),
                caption=selected_plot,
                use_container_width=True
            )

    else:
        st.warning("No plots found. Run eda.py first.")

# --------------------------------------------------
# MODEL COMPARISON
# --------------------------------------------------

elif page == "Model Comparison":

    st.header("Model Comparison")

    if COMPARISON_PATH.exists():

        comparison_df = pd.read_csv(COMPARISON_PATH)

        st.dataframe(
            comparison_df,
            use_container_width=True,
            height=220
        )

        st.subheader("Performance Comparison")

        numeric_columns = [
            "Accuracy",
            "Precision",
            "Recall",
            "F1-score",
            "ROC-AUC"
        ]

        available_columns = [
            col for col in numeric_columns
            if col in comparison_df.columns
        ]

        if available_columns:

            chart_df = comparison_df.set_index("Model")[available_columns]

            st.bar_chart(chart_df, height=360, use_container_width=True)

    else:
        st.warning("model_comparison.csv not found. Run model.py first.")

# --------------------------------------------------
# CUSTOMER PREDICTION
# --------------------------------------------------

elif page == "Customer Prediction":

    st.header("Customer Churn Prediction")

    st.write("Enter customer information below.")

    col1, col2 = st.columns(2)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )

        senior_citizen = st.selectbox(
            "Senior Citizen",
            [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )

        partner = st.selectbox(
            "Partner",
            ["No", "Yes"]
        )

        dependents = st.selectbox(
            "Dependents",
            ["No", "Yes"]
        )

        tenure = st.number_input(
            "Tenure (months)",
            min_value=0,
            max_value=100,
            value=12
        )

        phone_service = st.selectbox(
            "Phone Service",
            ["No", "Yes"]
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No phone service", "No", "Yes"]
        )

        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        online_security = st.selectbox(
            "Online Security",
            ["No internet service", "No", "Yes"]
        )

        online_backup = st.selectbox(
            "Online Backup",
            ["No internet service", "No", "Yes"]
        )

    with col2:

        device_protection = st.selectbox(
            "Device Protection",
            ["No internet service", "No", "Yes"]
        )

        tech_support = st.selectbox(
            "Tech Support",
            ["No internet service", "No", "Yes"]
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            ["No internet service", "No", "Yes"]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["No internet service", "No", "Yes"]
        )

        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )

        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["No", "Yes"]
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=70.70
        )

        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=151.65
        )

    predict_button = st.button(
        "🔍 Predict Churn",
        type="primary"
    )

    if predict_button:

        input_data = pd.DataFrame({
            "gender": [gender],
            "SeniorCitizen": [senior_citizen],
            "Partner": [partner],
            "Dependents": [dependents],
            "tenure": [tenure],
            "PhoneService": [phone_service],
            "MultipleLines": [multiple_lines],
            "InternetService": [internet_service],
            "OnlineSecurity": [online_security],
            "OnlineBackup": [online_backup],
            "DeviceProtection": [device_protection],
            "TechSupport": [tech_support],
            "StreamingTV": [streaming_tv],
            "StreamingMovies": [streaming_movies],
            "Contract": [contract],
            "PaperlessBilling": [paperless_billing],
            "PaymentMethod": [payment_method],
            "MonthlyCharges": [monthly_charges],
            "TotalCharges": [total_charges]
        })

        try:

            prediction = model.predict(input_data)

            probability = model.predict_proba(input_data)[0][1]

            st.divider()

            st.subheader("Prediction Result")

            if prediction[0] == 1:

                st.error("⚠️ Customer is likely to CHURN.")

            else:

                st.success("✅ Customer is likely to STAY.")

            st.metric(
                "Churn Probability",
                f"{probability * 100:.2f}%"
            )

            st.progress(float(probability))

        except Exception as e:

            st.error(f"Prediction error: {e}")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption("Customer Churn Prediction Project | Machine Learning")