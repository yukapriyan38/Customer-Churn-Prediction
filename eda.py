import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# ============================================================
# 1. LOAD CLEANED DATA
# ============================================================

# Project directory
BASE_DIR = Path(__file__).resolve().parent

# Path to cleaned dataset
file_path = BASE_DIR / "data" / "cleaned_data.csv"

df = pd.read_csv("data/cleaned_telco_churn.csv")

print("=" * 60)
print("EXPLORATORY DATA ANALYSIS - CUSTOMER CHURN")
print("=" * 60)

print("\nDataset shape:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())


# ============================================================
# 2. BASIC INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print(df.info())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())


# ============================================================
# 3. DESCRIPTIVE STATISTICS
# ============================================================

print("\n" + "=" * 60)
print("DESCRIPTIVE STATISTICS")
print("=" * 60)

print("\nNumerical columns:")
print(df.describe())

print("\nCategorical columns:")
print(df.describe(include="object"))


# ============================================================
# 4. CHURN DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("CHURN DISTRIBUTION")
print("=" * 60)

churn_counts = df["Churn"].value_counts()

print("\nChurn counts:")
print(churn_counts)

churn_percentage = df["Churn"].value_counts(normalize=True) * 100

print("\nChurn percentage:")
print(churn_percentage)


# ============================================================
# 5. SETUP FOR VISUALIZATIONS
# ============================================================

sns.set_theme(style="whitegrid")

# Create folder for plots
plots_dir = BASE_DIR / "plots"
plots_dir.mkdir(exist_ok=True)


# ============================================================
# 6. CHURN COUNT PLOT
# ============================================================

plt.figure(figsize=(7, 5))

sns.countplot(data=df, x="Churn")

plt.title("Customer Churn Distribution")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.savefig(plots_dir / "churn_distribution.png")
plt.show()


# ============================================================
# 7. CHURN RATE BY CONTRACT
# ============================================================

print("\n" + "=" * 60)
print("CHURN RATE BY CONTRACT")
print("=" * 60)

contract_churn = pd.crosstab(
    df["Contract"],
    df["Churn"],
    normalize="index"
) * 100

print(contract_churn)

# Convert to percentages for plotting
contract_churn.plot(
    kind="bar",
    stacked=True,
    figsize=(8, 5)
)

plt.title("Churn Rate by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Percentage (%)")
plt.legend(title="Churn", labels=["No", "Yes"])
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig(plots_dir / "churn_by_contract.png")
plt.show()


# ============================================================
# 8. CHURN RATE BY PAYMENT METHOD
# ============================================================

print("\n" + "=" * 60)
print("CHURN RATE BY PAYMENT METHOD")
print("=" * 60)

payment_churn = pd.crosstab(
    df["PaymentMethod"],
    df["Churn"],
    normalize="index"
) * 100

print(payment_churn)

payment_churn.plot(
    kind="bar",
    stacked=True,
    figsize=(10, 5)
)

plt.title("Churn Rate by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Percentage (%)")
plt.legend(title="Churn", labels=["No", "Yes"])
plt.xticks(rotation=30, ha="right")

plt.tight_layout()
plt.savefig(plots_dir / "churn_by_payment_method.png")
plt.show()


# ============================================================
# 9. CHURN RATE BY SENIOR CITIZEN
# ============================================================

print("\n" + "=" * 60)
print("CHURN RATE BY SENIOR CITIZEN")
print("=" * 60)

senior_churn = pd.crosstab(
    df["SeniorCitizen"],
    df["Churn"],
    normalize="index"
) * 100

print(senior_churn)

senior_churn.plot(
    kind="bar",
    stacked=True,
    figsize=(7, 5)
)

plt.title("Churn Rate by Senior Citizen Status")
plt.xlabel("Senior Citizen (0 = No, 1 = Yes)")
plt.ylabel("Percentage (%)")
plt.legend(title="Churn", labels=["No", "Yes"])
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig(plots_dir / "churn_by_senior_citizen.png")
plt.show()


# ============================================================
# 10. CHURN RATE BY PARTNER
# ============================================================

print("\n" + "=" * 60)
print("CHURN RATE BY PARTNER")
print("=" * 60)

partner_churn = pd.crosstab(
    df["Partner"],
    df["Churn"],
    normalize="index"
) * 100

print(partner_churn)

partner_churn.plot(
    kind="bar",
    stacked=True,
    figsize=(7, 5)
)

plt.title("Churn Rate by Partner Status")
plt.xlabel("Partner")
plt.ylabel("Percentage (%)")
plt.legend(title="Churn", labels=["No", "Yes"])
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig(plots_dir / "churn_by_partner.png")
plt.show()


# ============================================================
# 11. CHURN RATE BY INTERNET SERVICE
# ============================================================

print("\n" + "=" * 60)
print("CHURN RATE BY INTERNET SERVICE")
print("=" * 60)

internet_churn = pd.crosstab(
    df["InternetService"],
    df["Churn"],
    normalize="index"
) * 100

print(internet_churn)

internet_churn.plot(
    kind="bar",
    stacked=True,
    figsize=(8, 5)
)

plt.title("Churn Rate by Internet Service")
plt.xlabel("Internet Service")
plt.ylabel("Percentage (%)")
plt.legend(title="Churn", labels=["No", "Yes"])
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig(plots_dir / "churn_by_internet_service.png")
plt.show()


# ============================================================
# 12. CHURN RATE BY TENURE
# ============================================================

print("\n" + "=" * 60)
print("CHURN RATE BY TENURE")
print("=" * 60)

# Create tenure groups
df["TenureGroup"] = pd.cut(
    df["tenure"],
    bins=[-1, 12, 24, 48, 60, 100],
    labels=[
        "0-12 months",
        "13-24 months",
        "25-48 months",
        "49-60 months",
        "60+ months"
    ]
)

tenure_churn = pd.crosstab(
    df["TenureGroup"],
    df["Churn"],
    normalize="index"
) * 100

print(tenure_churn)

tenure_churn.plot(
    kind="bar",
    stacked=True,
    figsize=(9, 5)
)

plt.title("Churn Rate by Tenure Group")
plt.xlabel("Tenure Group")
plt.ylabel("Percentage (%)")
plt.legend(title="Churn", labels=["No", "Yes"])
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig(plots_dir / "churn_by_tenure.png")
plt.show()


# ============================================================
# 13. MONTHLY CHARGES DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("MONTHLY CHARGES ANALYSIS")
print("=" * 60)

print("\nAverage Monthly Charges by Churn:")
print(df.groupby("Churn")["MonthlyCharges"].mean())

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Churn",
    y="MonthlyCharges"
)

plt.title("Monthly Charges by Churn")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Monthly Charges")

plt.tight_layout()
plt.savefig(plots_dir / "monthly_charges_by_churn.png")
plt.show()


# ============================================================
# 14. TOTAL CHARGES DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("TOTAL CHARGES ANALYSIS")
print("=" * 60)

print("\nAverage Total Charges by Churn:")
print(df.groupby("Churn")["TotalCharges"].mean())

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Churn",
    y="TotalCharges"
)

plt.title("Total Charges by Churn")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Total Charges")

plt.tight_layout()
plt.savefig(plots_dir / "total_charges_by_churn.png")
plt.show()


# ============================================================
# 15. TENURE DISTRIBUTION BY CHURN
# ============================================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Churn",
    y="tenure"
)

plt.title("Tenure Distribution by Churn")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Tenure (Months)")

plt.tight_layout()
plt.savefig(plots_dir / "tenure_by_churn.png")
plt.show()


# ============================================================
# 16. CHURN RATE BY ONLINE SECURITY
# ============================================================

print("\n" + "=" * 60)
print("CHURN RATE BY ONLINE SECURITY")
print("=" * 60)

security_churn = pd.crosstab(
    df["OnlineSecurity"],
    df["Churn"],
    normalize="index"
) * 100

print(security_churn)

security_churn.plot(
    kind="bar",
    stacked=True,
    figsize=(8, 5)
)

plt.title("Churn Rate by Online Security")
plt.xlabel("Online Security")
plt.ylabel("Percentage (%)")
plt.legend(title="Churn", labels=["No", "Yes"])
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig(plots_dir / "churn_by_online_security.png")
plt.show()


# ============================================================
# 17. CHURN RATE BY TECH SUPPORT
# ============================================================

print("\n" + "=" * 60)
print("CHURN RATE BY TECH SUPPORT")
print("=" * 60)

tech_churn = pd.crosstab(
    df["TechSupport"],
    df["Churn"],
    normalize="index"
) * 100

print(tech_churn)

tech_churn.plot(
    kind="bar",
    stacked=True,
    figsize=(8, 5)
)

plt.title("Churn Rate by Tech Support")
plt.xlabel("Tech Support")
plt.ylabel("Percentage (%)")
plt.legend(title="Churn", labels=["No", "Yes"])
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig(plots_dir / "churn_by_tech_support.png")
plt.show()


# ============================================================
# 18. CORRELATION HEATMAP
# ============================================================

print("\n" + "=" * 60)
print("CORRELATION ANALYSIS")
print("=" * 60)

# Select numerical columns
numeric_df = df.select_dtypes(include=["int64", "float64"])

print("\nCorrelation with Churn:")
print(numeric_df.corr()["Churn"].sort_values(ascending=False))

plt.figure(figsize=(8, 6))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.tight_layout()
plt.savefig(plots_dir / "correlation_heatmap.png")
plt.show()


# ============================================================
# 19. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("EDA COMPLETED")
print("=" * 60)

print(f"\nTotal customers: {len(df)}")

print(f"Customers who churned: {df['Churn'].sum()}")

print(
    f"Overall churn rate: "
    f"{df['Churn'].mean() * 100:.2f}%"
)

print("\nPlots saved in:")
print(plots_dir)

print("\nEDA analysis completed successfully!")