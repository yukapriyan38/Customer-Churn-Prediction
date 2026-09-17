import pandas as pd

# --------------------------------------------------
# 1. Load the dataset
# --------------------------------------------------

file_path = "data/telco-customer-churn.csv"

df = pd.read_csv("data\WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Original shape:", df.shape)

# --------------------------------------------------
# 2. Remove unnecessary column
# --------------------------------------------------

df = df.drop(columns=["customerID"])

print("Shape after removing customerID:", df.shape)

# --------------------------------------------------
# 3. Convert TotalCharges to numeric
# --------------------------------------------------

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# --------------------------------------------------
# 4. Check missing values
# --------------------------------------------------

print("\nMissing values after converting TotalCharges:")
print(df.isnull().sum())

# --------------------------------------------------
# 5. Handle missing TotalCharges
# --------------------------------------------------

# TotalCharges is missing for some new customers.
# We replace missing values with 0 because
# their tenure is 0 and they have not accumulated charges.

df["TotalCharges"] = df["TotalCharges"].fillna(0)

# --------------------------------------------------
# 6. Convert target variable
# --------------------------------------------------

df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})

# --------------------------------------------------
# 7. Check final data
# --------------------------------------------------

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nData types after cleaning:")
print(df.dtypes)

print("\nFirst 5 rows after cleaning:")
print(df.head())

# --------------------------------------------------
# 8. Save cleaned dataset
# --------------------------------------------------

output_path = "data/cleaned_telco_churn.csv"

df.to_csv(output_path, index=False)

print("\nCleaned dataset saved successfully!")
print("Saved to:", "/data")