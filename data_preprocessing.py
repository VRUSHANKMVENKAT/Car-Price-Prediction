# ============================================================
# Car Price Prediction with Machine Learning
# STEP 2: DATA CLEANING & PREPROCESSING
# ------------------------------------------------------------
# Goal: turn the raw dataset into a clean DataFrame called
#       "df_clean", ready for feature engineering and model
#       training in the next steps.
#
# This script will NOT (on purpose):
#   - split the data into train/test sets  (later step)
#   - train any machine learning model      (later step)
#   - build any web interface               (later step)
# ============================================================

import pandas as pd

# ------------------------------------------------------------
# 1) LOAD THE DATASET
# ------------------------------------------------------------
# read_csv only READS the file. The original CSV is never modified.
df = pd.read_csv("car data.csv")

# Snapshot of the "before" state (we need this for the final summary)
original_rows = df.shape[0]
original_cols = df.shape[1]

print("=" * 60)
print("STEP 2A: LOADED RAW DATASET")
print("=" * 60)
print(f"  Raw dataset loaded: {original_rows} rows x {original_cols} columns")

# ------------------------------------------------------------
# 2) CHECK FOR MISSING / NULL VALUES IN EVERY COLUMN
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2B: MISSING VALUES PER COLUMN")
print("=" * 60)
missing_per_column = df.isnull().sum()
print(missing_per_column.to_string())
total_missing = int(missing_per_column.sum())
print(f"\n  TOTAL missing cells: {total_missing}")

# Decision rule (beginner-friendly): if a column had missing values we
# would FILL numeric columns with the median and categorical columns
# with the mode (most frequent value). That keeps every row instead of
# deleting data. In THIS dataset the count is 0, so nothing is filled.
if total_missing > 0:
    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                fill_value = df[col].median()          # middle value -> robust to outliers
                df[col] = df[col].fillna(fill_value)
                print(f"  Filled missing '{col}' with median = {fill_value}")
            else:
                fill_value = df[col].mode()[0]         # most frequent value
                df[col] = df[col].fillna(fill_value)
                print(f"  Filled missing '{col}' with mode = '{fill_value}'")
else:
    print("  -> No missing values found, so nothing needed filling.")

# ------------------------------------------------------------
# 3) CHECK FOR DUPLICATE ROWS
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2C: DUPLICATE ROWS")
print("=" * 60)
n_duplicates = int(df.duplicated().sum())
print(f"  Fully duplicated rows found: {n_duplicates}")
if n_duplicates > 0:
    print("  These rows appear more than once (evidence before removal):")
    print(df[df.duplicated(keep=False)].sort_values(list(df.columns)).to_string())

# ------------------------------------------------------------
# 4) CHECK FOR INVALID OR UNUSUAL VALUES
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2D: INVALID / UNUSUAL VALUE CHECKS")
print("=" * 60)
unusual_found = False

# 4a) Negative prices or kilometers are impossible -> they would be errors.
negative_price = int((df["Selling_Price"] < 0).sum())
negative_present = int((df["Present_Price"] < 0).sum())
negative_kms = int((df["Driven_kms"] < 0).sum())
print(f"  Negative Selling_Price values:  {negative_price}")
print(f"  Negative Present_Price values:  {negative_present}")
print(f"  Negative Driven_kms values:     {negative_kms}")
if negative_price or negative_present or negative_kms:
    unusual_found = True

# 4b) Car year should be a sensible range (not in the future, not ancient).
print(f"  Year range: {df['Year'].min()} to {df['Year'].max()}  (no future years -> looks valid)")
if (df["Year"] > 2026).any():
    unusual_found = True
    print("  WARNING: future years found!")

# 4c) Selling price should be smaller than present price for most cars
#     (used cars lose value). We only REPORT this - we do NOT delete rows.
n_price_anomaly = int((df["Selling_Price"] > df["Present_Price"]).sum())
print(f"  Rows where Selling_Price > Present_Price: {n_price_anomaly} (kept, only reported)")

# 4d) Extreme (outlier) values - we only REPORT them, we do NOT delete.
#     Rule of thumb: anything beyond mean +/- 3*std is very unusual.
for col in ["Selling_Price", "Present_Price", "Driven_kms"]:
    mean = df[col].mean()
    std = df[col].std()
    upper = mean + 3 * std
    n_outliers = int((df[col] > upper).sum())
    print(f"  Outliers in {col} (>{upper:.2f}): {n_outliers} (kept, only reported)")

# 4e) Check the categorical columns only contain expected labels.
print(f"  Fuel_Type values:    {sorted(df['Fuel_Type'].unique())}")
print(f"  Selling_type values: {sorted(df['Selling_type'].unique())}")
print(f"  Transmission values: {sorted(df['Transmission'].unique())}")
print(f"  Owner values:        {sorted(df['Owner'].unique())}  (0 = first owner)")
if not unusual_found:
    print("  -> No impossible/invalid values found. Outliers are kept for now;")
print("     we will visualise them in the EDA step, not delete them blindly.")

# ------------------------------------------------------------
# 5) CHECK DATA TYPES OF ALL COLUMNS
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2E: DATA TYPES BEFORE CONVERSION")
print("=" * 60)
print(df.dtypes.to_string())

# 5b) CONVERT TO APPROPRIATE TYPES IF NECESSARY
#     - Year is stored as a whole number (int64) already -> correct.
#     - Owner is a count (0, 1, 3) -> int64 is correct.
#     - The 4 text columns should be 'category' type: they take only a
#       few fixed labels, and category type uses less memory and is
#       ready for encoding later.
text_columns = ["Car_Name", "Fuel_Type", "Selling_type", "Transmission"]
for col in text_columns:
    df[col] = df[col].astype("category")

print("\n  Data types AFTER conversion (text columns are now 'category'):")
print(df.dtypes.to_string())

# ------------------------------------------------------------
# 6) REMOVE DUPLICATE ROWS (only exact copies are removed)
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2F: REMOVE DUPLICATE ROWS")
print("=" * 60)
rows_before_dedup = df.shape[0]
# keep="first" means: keep the first occurrence, drop the exact copies.
# We do NOT reset the index yet - we need the original row positions
# for the target-check below.
df_clean = df.drop_duplicates(keep="first")
rows_removed = rows_before_dedup - df_clean.shape[0]
print(f"  Rows before removing duplicates: {rows_before_dedup}")
print(f"  Exact duplicate rows removed:    {rows_removed}")
print(f"  Rows after removing duplicates:  {df_clean.shape[0]}")
print("  (Every remaining row is still a REAL record - only exact copies went.)")

# ------------------------------------------------------------
# 7) FINAL SAFETY CHECK ON THE TARGET COLUMN
# ------------------------------------------------------------
# Requirement: the target "Selling_Price" must stay UNCHANGED.
# We prove it: every KEPT row's target value must be exactly identical
# to the value in the original file at that same row position.
target_before = pd.read_csv("car data.csv")["Selling_Price"]
target_preserved = df_clean["Selling_Price"].equals(
    target_before.loc[df_clean.index]
)
print("\n" + "=" * 60)
print("STEP 2G: TARGET COLUMN CHECK")
print("=" * 60)
print(f"  Selling_Price dtype:      {df_clean['Selling_Price'].dtype}")
print(f"  All kept target values identical to the original file: {target_preserved}")
print("  (Only whole duplicate ROWS were removed; no target value was edited.)")

# Now give df_clean tidy row numbers (0, 1, 2, ...) for the next steps.
df_clean = df_clean.reset_index(drop=True)

# ------------------------------------------------------------
# 8) BEFORE-AND-AFTER SUMMARY
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("FINAL BEFORE / AFTER SUMMARY")
print("=" * 60)
print(f"  Original number of rows:      {original_rows}")
print(f"  Original number of columns:   {original_cols}")
print(f"  Number of missing values:     {total_missing}")
print(f"  Number of duplicate rows:     {n_duplicates}")
print(f"  Final number of rows:         {df_clean.shape[0]}")
print(f"  Final number of columns:      {df_clean.shape[1]}")

# ------------------------------------------------------------
# 9) SHOW THE CLEANED DATAFRAME
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("df_clean.head()")
print("=" * 60)
print(df_clean.head().to_string())

print("\n" + "=" * 60)
print("df_clean.info()")
print("=" * 60)
df_clean.info()

print("\n" + "=" * 60)
print("df_clean.describe()")
print("=" * 60)
print(df_clean.describe().round(2).to_string())

print("\n" + "=" * 60)
print("Preprocessing complete. df_clean is ready for the next step.")
print("No train/test split, no model training, no web interface - as requested.")
print("=" * 60)
