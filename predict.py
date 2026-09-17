
import joblib
import pandas as pd
from pathlib import Path


# Load the saved best model
MODEL_PATH = Path("models/best_model.joblib")

model = joblib.load(MODEL_PATH)


print("=" * 60)
print("CUSTOMER CHURN PREDICTION")
print("=" * 60)

# Collect customer information
gender = input("Gender (Male/Female): ")
senior_citizen = int(input("Senior Citizen (0 = No, 1 = Yes): "))
partner = input("Partner (Yes/No): ")
dependents = input("Dependents (Yes/No): ")
tenure = int(input("Tenure in months: "))
phone_service = input("Phone Service (Yes/No): ")
multiple_lines = input("Multiple Lines (Yes/No/No phone service): ")
internet_service = input("Internet Service (DSL/Fiber optic/No): ")
online_security = input("Online Security (Yes/No/No internet service): ")
online_backup = input("Online Backup (Yes/No/No internet service): ")
device_protection = input("Device Protection (Yes/No/No internet service): ")
tech_support = input("Tech Support (Yes/No/No internet service): ")
streaming_tv = input("Streaming TV (Yes/No/No internet service): ")
streaming_movies = input("Streaming Movies (Yes/No/No internet service): ")
contract = input("Contract (Month-to-month/One year/Two year): ")
paperless_billing = input("Paperless Billing (Yes/No): ")
payment_method = input(
    "Payment Method (Electronic check/Mailed check/"
    "Bank transfer (automatic)/Credit card (automatic)): "
)
monthly_charges = float(input("Monthly Charges: "))
total_charges = float(input("Total Charges: "))


# Create a DataFrame with the same feature names as the training data
customer = pd.DataFrame([{
    "gender": gender,
    "SeniorCitizen": senior_citizen,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges
}])


# Make prediction
prediction = model.predict(customer)[0]
probability = model.predict_proba(customer)[0][1]


print("\n" + "=" * 60)
print("PREDICTION RESULT")
print("=" * 60)

if prediction == 1:
    print("Customer is likely to CHURN.")
else:
    print("Customer is likely to STAY.")

print(f"Churn probability: {probability:.2%}")
print("=" * 60)

