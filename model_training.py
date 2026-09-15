# ============================================================
# Car Price Prediction with Machine Learning
# STEP 5: MODEL TRAINING AND EVALUATION
# ------------------------------------------------------------
# Goal: train 3 regression models to predict Selling_Price,
#       compare them with MAE / RMSE / R2, pick and save the
#       best one (preprocessing + model together in a Pipeline).
#
# This script will NOT (on purpose):
#   - use Selling_Price as an input (DATA LEAKAGE!)
#   - use Car_Name (98 categories incl. bikes -> overfitting risk)
#       or Year (Car_Age already carries the age information)
#   - create any GUI / web app       (later step)
#
# IMPORTANT RULES respected:
#   - "car data.csv" and "df_features.csv" are never modified
#   - no rows are removed, no target values are changed
#   - nothing is fabricated: only real dataset columns are used
# ============================================================

import os

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

matplotlib.use("Agg")  # save charts to file (no screen needed)

# ------------------------------------------------------------
# 2) LOAD DATA (cleaned/engineered dataset from Step 3)
# ------------------------------------------------------------
if not os.path.exists("df_features.csv"):
    raise SystemExit("df_features.csv not found - please run Step 3 "
                     "(data_feature_engineering.py) first.")

df = pd.read_csv("df_features.csv")

print("=" * 60)
print("STEP 5A: LOADED DATA")
print("=" * 60)
print(f"  Dataset shape: {df.shape}")
print(f"  Column names:  {list(df.columns)}")
print("\n  First 5 rows:")
print(df.head().to_string())

# ------------------------------------------------------------
# 3 + 4) DEFINE TARGET AND SELECT FEATURES
# ------------------------------------------------------------
# Target = Selling_Price (in Lakhs). It must NEVER be an input.
TARGET = "Selling_Price"

# Inputs chosen in the task description:
numeric_features = ["Present_Price", "Driven_kms", "Owner", "Car_Age"]
categorical_features = ["Fuel_Type", "Selling_type", "Transmission"]
feature_columns = numeric_features + categorical_features

X = df[feature_columns]      # inputs (8 columns after encoding)
y = df[TARGET]               # the answer we want to predict

# ------------------------------------------------------------
# 5) TRAIN / TEST SPLIT
# ------------------------------------------------------------
# The model LEARNS from the training set and is then EXAMINED on
# the test set (rows it has never seen) - like an exam with new
# questions. random_state=42 makes the split repeatable.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print("\n" + "=" * 60)
print("STEP 5B: TRAIN / TEST SPLIT")
print("=" * 60)
print(f"  X shape: {X.shape}   y shape: {y.shape}")
print(f"  Training rows: {len(X_train)}   Testing rows: {len(X_test)}")

# ------------------------------------------------------------
# 6) PREPROCESSING (ColumnTransformer inside a Pipeline)
# ------------------------------------------------------------
# Numeric: fill (never-seen) gaps with the MEDIAN.
# Categorical: fill gaps with the MOST FREQUENT value, then
# One-Hot Encode text labels into 0/1 columns.
#
# Everything lives INSIDE the Pipeline, so it is fitted on the
# TRAINING data only -> no data leakage.
try:
    # modern scikit-learn
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    # very old scikit-learn
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), numeric_features),
        ("cat", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", ohe),
        ]), categorical_features),
    ]
)

# ------------------------------------------------------------
# 7) THE THREE MODELS (each: preprocessor -> model)
# ------------------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=300, random_state=42, max_depth=None
    ),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
}

pipelines = {
    name: Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    for name, model in models.items()
}

# ------------------------------------------------------------
# 8 + 9) TRAIN, PREDICT AND EVALUATE EVERY MODEL
# ------------------------------------------------------------
# MAE  = average absolute error (in Lakhs)        -> easy to explain
# RMSE = like MAE but punishes BIG mistakes more  -> in Lakhs
# R2   = share of price variation the model explains (1.0 = perfect)
results = []

print("\n" + "=" * 60)
print("STEP 5C: TRAINING AND EVALUATION")
print("=" * 60)
for name, pipe in pipelines.items():
    pipe.fit(X_train, y_train)                 # learn ONLY from training data
    predictions = pipe.predict(X_test)         # exam on unseen data

    mae = mean_absolute_error(y_test, predictions)
    # np.sqrt(mse) works on every scikit-learn version
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2})
    print(f"  Trained: {name:20s} MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.4f}")

# ------------------------------------------------------------
# 10) SAVE MODEL COMPARISON
# ------------------------------------------------------------
comparison = pd.DataFrame(results)
comparison.to_csv("model_comparison.csv", index=False)

print("\n" + "=" * 60)
print("MODEL COMPARISON TABLE (errors in Lakhs)")
print("=" * 60)
print(comparison.round(4).to_string(index=False))

# ------------------------------------------------------------
# 11) SELECT THE BEST MODEL (highest R2; tie -> lower RMSE)
# ------------------------------------------------------------
# Sort by R2 descending, then by RMSE ascending as tie-break.
ranked = comparison.sort_values(by=["R2", "RMSE"], ascending=[False, True])
best_row = ranked.iloc[0]
best_name = best_row["Model"]
best_pipe = pipelines[best_name]               # already fitted

print("\n" + "=" * 60)
print("BEST MODEL (chosen by highest R2, RMSE as tie-break)")
print("=" * 60)
print(f"  BEST MODEL:    {best_name}")
print(f"  BEST MODEL R2: {best_row['R2']:.4f}")
print(f"  BEST MODEL MAE:  {best_row['MAE']:.4f} Lakhs")
print(f"  BEST MODEL RMSE: {best_row['RMSE']:.4f} Lakhs")

# ------------------------------------------------------------
# 12) SAVE THE BEST MODEL (preprocessing + model together)
# ------------------------------------------------------------
# The WHOLE Pipeline is saved, so later (web app) we can feed it
# raw feature values in the original format.
joblib.dump(best_pipe, "car_price_model.pkl")
print("\n  Best model saved as: car_price_model.pkl")

# ------------------------------------------------------------
# 13) FEATURE IMPORTANCE (only for tree-based models)
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)
best_model_step = best_pipe.named_steps["model"]

if hasattr(best_model_step, "feature_importances_"):
    # Feature names come from the FITTED ColumnTransformer.
    raw_names = best_pipe.named_steps["preprocessor"].get_feature_names_out()
    # Remove the 'num__' / 'cat__' prefixes for readability.
    clean_names = [n.split("__", 1)[1] for n in raw_names]

    importance_df = pd.DataFrame({
        "Feature": clean_names,
        "Importance": best_model_step.feature_importances_,
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    importance_df.to_csv("feature_importance.csv", index=False)
    print("  Top 10 most important features:")
    print(importance_df.head(10).round(4).to_string(index=False))
    print("  Saved: feature_importance.csv")
else:
    print("  The best model (Linear Regression) does not provide "
          "feature_importances_, so no importance table is created. "
          "(Its coefficients could be inspected instead.)")

# ------------------------------------------------------------
# 14) ACTUAL vs PREDICTED DATA (best model)
# ------------------------------------------------------------
best_predictions = best_pipe.predict(X_test)
predictions_df = pd.DataFrame({
    "Actual_Selling_Price": y_test.values,
    "Predicted_Selling_Price": best_predictions,
})
predictions_df.to_csv("predictions.csv", index=False)

# ------------------------------------------------------------
# 15) BASIC MODEL VALIDATION
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("BASIC MODEL VALIDATION")
print("=" * 60)
print(f"  Number of training samples:      {len(X_train)}")
print(f"  Number of testing samples:       {len(X_test)}")
print(f"  Any NaN in predictions?          {bool(np.isnan(best_predictions).any())}")
print(f"  Minimum predicted price:         {best_predictions.min():.2f} Lakhs")
print(f"  Maximum predicted price:         {best_predictions.max():.2f} Lakhs")
print("  (Predictions were NOT modified to improve the metrics.)")

# ------------------------------------------------------------
# 16) ACTUAL vs PREDICTED GRAPH
# ------------------------------------------------------------
os.makedirs("charts", exist_ok=True)
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(y_test.values, best_predictions, alpha=0.7,
           color="steelblue", edgecolor="k", linewidth=0.3,
           label="Test set predictions")
# Perfect-prediction diagonal (y = x): points ON the line are exact.
limits = [0, max(y_test.max(), best_predictions.max()) * 1.05]
ax.plot(limits, limits, color="red", linestyle="--", label="Perfect prediction (y = x)")
ax.set_title(f"Actual vs Predicted Selling Price - {best_name}")
ax.set_xlabel("Actual Selling Price (in Lakhs)")
ax.set_ylabel("Predicted Selling Price (in Lakhs)")
ax.set_xlim(limits)
ax.set_ylim(limits)
ax.legend()
ax.grid(True, linestyle=":", alpha=0.6)
fig.savefig("charts/11_actual_vs_predicted.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("\n  Saved: charts/11_actual_vs_predicted.png")

# ------------------------------------------------------------
# 17) FINAL CONSOLE SUMMARY
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("CAR PRICE PREDICTION - MODEL RESULTS")
print("=" * 60)
print(f"""
Dataset:
{len(df)} rows after preprocessing/duplicate removal

Target:
{TARGET}

Models tested:
1. Linear Regression
2. Random Forest Regression
3. Gradient Boosting Regression
""")
print(comparison.round(4).to_string(index=False))
print(f"""
Best Model: {best_name}
R2:   {best_row['R2']:.4f}
MAE:  {best_row['MAE']:.4f} Lakhs
RMSE: {best_row['RMSE']:.4f} Lakhs

Metric meanings:
- MAE  = the average absolute prediction error (in Lakhs).
- RMSE = like MAE but penalizes larger errors more strongly.
- R2   = how much variation in selling price the model explains
         (1.0 = perfect, 0 = no better than always guessing the mean).
""")
print(f"scikit-learn version used: {sklearn.__version__}")
print("=" * 60)
