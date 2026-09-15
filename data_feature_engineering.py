# ============================================================
# Car Price Prediction with Machine Learning
# STEP 3: FEATURE ENGINEERING
# ------------------------------------------------------------
# Goal: create new, more useful input features from the cleaned
#       DataFrame "df_clean" (built in Step 2) and save the
#       result as "df_features".
#
# This script will NOT (on purpose):
#   - use Selling_Price to build any feature (DATA LEAKAGE!)
#   - apply scaling (depends on the model, decided later)
#   - train any machine learning model      (later step)
#   - build any web interface               (later step)
# ============================================================

import pandas as pd

# ------------------------------------------------------------
# 1) REBUILD df_clean (same cleaning steps as Step 2)
# ------------------------------------------------------------
# Each script stays self-contained: read the raw file and repeat
# the two cleaning actions we decided in Step 2.
df_clean = pd.read_csv("car data.csv")

# Cleaning action 1 (Step 2F): drop the 2 exact duplicate rows
df_clean = df_clean.drop_duplicates(keep="first").reset_index(drop=True)

# Cleaning action 2 (Step 2E): text columns -> category type
text_columns = ["Car_Name", "Fuel_Type", "Selling_type", "Transmission"]
for col in text_columns:
    df_clean[col] = df_clean[col].astype("category")

print("=" * 60)
print("STARTING POINT: df_clean")
print("=" * 60)
print(f"  df_clean loaded: {df_clean.shape[0]} rows x {df_clean.shape[1]} columns")

# ------------------------------------------------------------
# 2) CREATE THE NEW FEATURE: Car_Age
# ------------------------------------------------------------
# A car's age is what really drives its price (depreciation).
# The requirement says to use 2026 as the current year.
CURRENT_YEAR = 2026
df_features = df_clean.copy()                 # work on a copy, keep df_clean intact
df_features["Car_Age"] = CURRENT_YEAR - df_features["Year"]

# Sanity check: ages must be positive whole numbers.
print("\n" + "=" * 60)
print("STEP 3A: NEW FEATURE 'Car_Age' = 2026 - Year")
print("=" * 60)
print(f"  Car_Age range: {df_features['Car_Age'].min()} to {df_features['Car_Age'].max()} years")
print(f"  Negative ages (would be an error): {(df_features['Car_Age'] < 0).sum()}")

# Requirement: KEEP the original Year column so we can compare
# Year vs Car_Age later. Nothing is dropped here.

# ------------------------------------------------------------
# 3) WHY NOT BUILD FEATURES FROM Selling_Price?
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3B: DATA LEAKAGE CHECK")
print("=" * 60)
print("""  Selling_Price is our TARGET (the answer we want to predict).
  If we built an input feature from it (e.g. 'price per km'), the model
  would secretly see the answer while training -> unrealistically good
  scores that collapse in the real world. This is called DATA LEAKAGE.
  -> Features are built ONLY from Year, Present_Price, Driven_kms,
     Fuel_Type, Selling_type, Transmission and Owner. NEVER from
     Selling_Price.""")

# ------------------------------------------------------------
# 4) DO NUMERICAL FEATURES NEED SCALING?
# ------------------------------------------------------------
# Scaling = shrinking features onto a similar range (e.g. 0 to 1).
# Whether we need it depends on the MODEL we will choose:
#   - Tree models (Decision Tree / Random Forest): split with simple
#     questions like "kms > 30000?" -> units do not matter -> NOT needed.
#   - Linear Regression (ordinary least squares): solved
#     mathematically -> NOT strictly needed.
#   - Distance-based models (KNN) or regularized/gradient models:
#     REQUIRED, because big numbers would dominate.
# We print the scales now, decide LATER together with the model choice.
numeric_features = ["Year", "Car_Age", "Present_Price", "Driven_kms", "Owner"]
print("\n" + "=" * 60)
print("STEP 3C: SCALING CHECK (numbers only, no target)")
print("=" * 60)
print(df_features[numeric_features].agg(["min", "max", "mean"]).round(2).to_string())
print("""  Observation: the scales are very different (Driven_kms goes up to
  500,000 while Owner is 0-3). BUT scaling is a MODEL requirement, not
  a data-cleaning step, so:
  -> DECISION: no scaling is applied now. If the model we choose later
     needs it, we will scale inside a scikit-learn Pipeline (fit on
     train data only) to avoid leakage.""")

# ------------------------------------------------------------
# 5) INSPECT THE CATEGORICAL COLUMNS
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3D: UNIQUE VALUES OF CATEGORICAL COLUMNS")
print("=" * 60)
for col in ["Fuel_Type", "Selling_type", "Transmission"]:
    values = list(df_features[col].unique())
    print(f"  {col} ({len(values)} unique): {values}")
    print(f"    value counts: {dict(df_features[col].value_counts())}")

# Extra context for the viva:
print(f"\n  Owner (numeric, 0 = first owner): {sorted(df_features['Owner'].unique())}")
print(f"  Car_Name: {df_features['Car_Name'].nunique()} unique names (98 car/bike models).")
print("""  Note for later: Fuel_Type / Selling_type / Transmission have only a
  few labels -> perfect for One-Hot Encoding in the next step.
  Car_Name has 98 labels -> one-hot would add 98 columns; we will
  discuss dropping it (Present_Price already captures the car's class).""")

# ------------------------------------------------------------
# 6) QUICK PROOF THAT Car_Age IS USEFUL
# ------------------------------------------------------------
# Correlation: how strongly each feature moves with Selling_Price
# (-1 or +1 = strong relationship, ~0 = weak relationship).
print("\n" + "=" * 60)
print("STEP 3E: CORRELATION WITH SELLING_PRICE (justification)")
print("=" * 60)
corr = df_features[["Year", "Car_Age", "Present_Price", "Driven_kms"]].corrwith(
    df_features["Selling_Price"]
).round(3)
print(corr.to_string())
print("""  Reading: Year correlates POSITIVELY (newer -> costlier) and Car_Age
  mirrors it NEGATIVELY (older -> cheaper). Same information, but
  Car_Age is directly interpretable: 'this car is 12 years old'.""")

# ------------------------------------------------------------
# 7) SHOW THE RESULT
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3F: FIRST 10 ROWS WITH THE NEW Car_Age COLUMN")
print("=" * 60)
print(df_features[["Car_Name", "Year", "Car_Age", "Selling_Price"]].head(10).to_string())

print("\n" + "=" * 60)
print("df_features.info()")
print("=" * 60)
df_features.info()

# ------------------------------------------------------------
# 8) SAVE THE RESULT
# ------------------------------------------------------------
# Saved as a NEW file for the next steps. The original CSV is untouched.
df_features.to_csv("df_features.csv", index=False)
print("\n  Saved: df_features.csv "
      f"({df_features.shape[0]} rows x {df_features.shape[1]} columns)")

print("\n" + "=" * 60)
print("Feature engineering complete. No model trained, no scaling applied,")
print("no web interface - as requested. Waiting for the next step.")
print("=" * 60)
