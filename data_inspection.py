# ============================================================
# Car Price Prediction with Machine Learning
# STEP 1: DATA INSPECTION (read-only - nothing is modified)
# ============================================================
import pandas as pd

CSV_PATH = "car data.csv"

# Load the dataset (pandas only READS the file, it never changes it)
df = pd.read_csv(CSV_PATH)

print("=" * 60)
print("1) FIRST 5 ROWS")
print("=" * 60)
print(df.head())

print("\n" + "=" * 60)
print("2) COLUMNS")
print("=" * 60)
for i, col in enumerate(df.columns, start=1):
    print(f"  {i}. {col}")

print("\n" + "=" * 60)
print("3) SHAPE (rows, columns)")
print("=" * 60)
print(f"  Rows: {df.shape[0]}   Columns: {df.shape[1]}")

print("\n" + "=" * 60)
print("4) DATA TYPES")
print("=" * 60)
print(df.dtypes.to_string())

print("\n" + "=" * 60)
print("5) MISSING VALUES (per column)")
print("=" * 60)
missing = df.isnull().sum()
print(missing.to_string())
print(f"\n  Total missing cells in whole dataset: {missing.sum()}")

print("\n" + "=" * 60)
print("6) DUPLICATE ROWS")
print("=" * 60)
dup_count = df.duplicated().sum()
print(f"  Fully duplicated rows: {dup_count}")
if dup_count > 0:
    print("  Examples of duplicated rows:")
    print(df[df.duplicated(keep=False)].sort_values(list(df.columns)).head(10).to_string())

print("\n" + "=" * 60)
print("7) NUMERIC SUMMARY (describe)")
print("=" * 60)
print(df.describe().round(2).to_string())

print("\n" + "=" * 60)
print("8) CATEGORICAL VALUE COUNTS")
print("=" * 60)
cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
for col in cat_cols:
    print(f"\n  {col}:")
    for value, count in df[col].value_counts().items():
        print(f"    {value}: {count}")

print("\n" + "=" * 60)
print("9) TARGET VARIABLE CHECK")
print("=" * 60)
print("  Candidate target: 'Selling_Price' (the price the car was sold for).")
print("  All other columns are INPUT features used to predict it.")

print("\n" + "=" * 60)
print("10) QUICK SANITY CHECKS")
print("=" * 60)
print(f"  Years range:        {df['Year'].min()} to {df['Year'].max()}")
print(f"  Negative prices?    {(df['Selling_Price'] < 0).sum()}")
print(f"  Negative kms?       {(df['Driven_kms'] < 0).sum()}")
print(f"  Owner values:       {sorted(df['Owner'].unique())}")
print(f"  Unique car names:   {df['Car_Name'].nunique()}")
print("\nInspection complete - dataset was NOT modified.")
