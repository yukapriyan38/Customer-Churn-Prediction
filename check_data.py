import pandas as pd

# --------------------------------------------------
# 1. Load the dataset
# --------------------------------------------------

df = pd.read_csv(
    "data\cleaned_telco_churn.csv"
)

# --------------------------------------------------
# 2. Display first 5 rows
# --------------------------------------------------

print("\nFirst 5 rows:")
print(df.head())

# --------------------------------------------------
# 3. Display last 5 rows
# --------------------------------------------------

print("\nLast 5 rows:")
print(df.tail())

# --------------------------------------------------
# 4. Display number of rows and columns
# --------------------------------------------------

print("\nDataset shape:")
print(df.shape)

# --------------------------------------------------
# 5. Display column names
# --------------------------------------------------

print("\nColumn names:")
print(df.columns.tolist())

# --------------------------------------------------
# 6. Display data types
# --------------------------------------------------

print("\nData types:")
print(df.dtypes)

# --------------------------------------------------
# 7. Display missing values
# --------------------------------------------------

print("\nMissing values:")
print(df.isnull().sum())

# --------------------------------------------------
# 8. Display dataset information
# --------------------------------------------------

print("\nDataset information:")
df.info()

# --------------------------------------------------
# 9. Display target distribution
# --------------------------------------------------

print("\nChurn distribution:")
print(df["Churn"].value_counts())

print("\nChurn percentage:")
print(df["Churn"].value_counts(normalize=True) * 100)