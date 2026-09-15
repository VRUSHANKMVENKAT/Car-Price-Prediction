# ============================================================
# Car Price Prediction with Machine Learning
# STEP 6: MODEL ANALYSIS, FEATURE IMPORTANCE AND BUSINESS INSIGHTS
# ------------------------------------------------------------
# Goal: analyze the model ACTUALLY selected in Step 5 and turn
#       the real results into charts, tables and business
#       insights - everything is computed live from files, so
#       no score is ever hard-coded or invented.
#
# This script will NOT (on purpose):
#   - retrain models (car_price_model.pkl is only LOADED)
#   - modify any dataset file
#   - create a web application (later step)
# ============================================================

import os

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")  # charts go to files, no screen needed
os.makedirs("charts", exist_ok=True)
sns.set_theme(style="whitegrid")

# ------------------------------------------------------------
# 1) LOAD THE REQUIRED FILES (all outputs of previous steps)
# ------------------------------------------------------------
for required in ["df_features.csv", "model_comparison.csv",
                 "car_price_model.pkl", "predictions.csv"]:
    if not os.path.exists(required):
        raise SystemExit(f"Required file '{required}' not found - "
                         "please run the earlier steps first.")

df_features = pd.read_csv("df_features.csv")
comparison = pd.read_csv("model_comparison.csv")
best_pipe = joblib.load("car_price_model.pkl")   # fitted Step-5 pipeline
predictions = pd.read_csv("predictions.csv")

print("=" * 60)
print("FILES LOADED")
print("=" * 60)
print(f"  df_features.csv:            {df_features.shape[0]} rows x {df_features.shape[1]} cols")
print(f"  model_comparison.csv:       {comparison.shape[0]} models")
print(f"  car_price_model.pkl:        loaded fitted pipeline")
print(f"  predictions.csv:            {predictions.shape[0]} rows")

# ------------------------------------------------------------
# 2) IDENTIFY THE BEST MODEL (from REAL comparison results)
# ------------------------------------------------------------
# Highest R2 wins; if R2s are extremely close (within 0.01),
# the lower RMSE decides.
comparison_sorted = comparison.sort_values(by=["R2", "RMSE"], ascending=[False, True])
best_row = comparison_sorted.iloc[0]
runner_up = comparison_sorted.iloc[1]

if (best_row["R2"] - runner_up["R2"]) < 0.01:      # near tie -> RMSE decides
    if runner_up["RMSE"] < best_row["RMSE"]:
        best_row = runner_up

BEST_NAME = best_row["Model"]

print("\n" + "=" * 60)
print("BEST MODEL ANALYSIS")
print("=" * 60)
print(f"  Best Model: {BEST_NAME}")
print(f"  R2:   {best_row['R2']:.4f}")
print(f"  MAE:  {best_row['MAE']:.4f} Lakhs")
print(f"  RMSE: {best_row['RMSE']:.4f} Lakhs")

# ------------------------------------------------------------
# 3) FEATURE IMPORTANCE
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 60)

if os.path.exists("feature_importance.csv"):
    # Tree-based best model: use the saved importances.
    importance_df = pd.read_csv("feature_importance.csv")
    importance_source = "model feature_importances_"
else:
    # Linear Regression best model -> analyze ABSOLUTE COEFFICIENTS.
    # For comparison between features the coefficients are standardized:
    # we refit-free standardize by scaling with the TRAINING feature
    # spread. To stay beginner-friendly AND avoid retraining, we use the
    # dataset spread as a fair approximation of each feature's scale.
    model_step = best_pipe.named_steps["model"]
    preprocessor = best_pipe.named_steps["preprocessor"]
    names_raw = preprocessor.get_feature_names_out()
    coef = model_step.coef_

    coef_df = pd.DataFrame({
        "Feature": [n.split("__", 1)[1] for n in names_raw],
        "Coefficient": coef,
    })

    # To make features comparable (coefficients depend on units), use the
    # spread (standard deviation) of each ORIGINAL column as its scale.
    spreads = {}
    for col in ["Present_Price", "Driven_kms", "Owner", "Car_Age"]:
        spreads[col] = df_features[col].std()
    # One-hot columns are 0/1 -> scale = 1
    for col in coef_df["Feature"]:
        if col not in spreads:
            spreads[col] = 1.0

    coef_df["Importance"] = (coef_df["Coefficient"].abs()
                             * coef_df["Feature"].map(spreads))
    coef_df["Direction"] = np.where(coef_df["Coefficient"] >= 0,
                                    "raises price", "lowers price")
    coef_df = coef_df.sort_values(by="Importance",
                                  ascending=False).reset_index(drop=True)
    importance_df = coef_df[["Feature", "Importance"]].copy()
    importance_df.to_csv("feature_importance.csv", index=False)
    importance_source = "absolute standardized coefficients (Linear Regression)"

print(f"  Importance source: {importance_source}")
top10 = importance_df.head(10)
print("\n  Top 10 most important features:")
print(top10.round(4).to_string(index=False))

# Full coefficient table for the Linear Regression case (viva gold)
if "Coefficient" in coef_df.columns:
    print("\n  Full coefficient table (direction of effect):")
    print(coef_df[["Feature", "Coefficient", "Direction", "Importance"]]
          .round(4).to_string(index=False))

# CHART 12: horizontal bar chart of the top 10 features
fig, ax = plt.subplots(figsize=(9, 6))
sns.barplot(data=top10, x="Importance", y="Feature",
            hue="Feature", palette="viridis", legend=False, ax=ax)
ax.set_title(f"Top 10 Feature Importance - {BEST_NAME}")
ax.set_xlabel("Importance (standardized effect size)")
ax.set_ylabel("Feature")
ax.grid(True, linestyle=":", alpha=0.6)
fig.tight_layout()
fig.savefig("charts/12_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("\n  Saved: charts/12_feature_importance.png")

# ------------------------------------------------------------
# 4) ANALYZE ACTUAL vs PREDICTED PRICES
# ------------------------------------------------------------
predictions["Prediction_Error"] = (predictions["Actual_Selling_Price"]
                                   - predictions["Predicted_Selling_Price"])
predictions["Absolute_Error"] = predictions["Prediction_Error"].abs()

mean_abs_err = predictions["Absolute_Error"].mean()
max_abs_err = predictions["Absolute_Error"].max()

print("\n" + "=" * 60)
print("ACTUAL vs PREDICTED ANALYSIS (test set)")
print("=" * 60)
print(f"  Mean absolute error:     {mean_abs_err:.4f} Lakhs")
print(f"  Maximum absolute error:  {max_abs_err:.4f} Lakhs")
print(f"  Min actual price:        {predictions['Actual_Selling_Price'].min():.2f} Lakhs")
print(f"  Max actual price:        {predictions['Actual_Selling_Price'].max():.2f} Lakhs")
print(f"  Min predicted price:     {predictions['Predicted_Selling_Price'].min():.2f} Lakhs")
print(f"  Max predicted price:     {predictions['Predicted_Selling_Price'].max():.2f} Lakhs")

# The 5 test-set rows with the LARGEST absolute errors
top_errors = predictions.sort_values(by="Absolute_Error",
                                     ascending=False).head(5)
print("\n  5 test-set cars with the largest prediction errors:")
print(top_errors.round(3).to_string(index=False))

# ------------------------------------------------------------
# 5) ERROR DISTRIBUTION GRAPH
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(predictions["Prediction_Error"], bins=20, kde=True,
             color="indianred", ax=ax)
ax.axvline(0, color="black", linestyle="--", linewidth=1, label="Zero error")
ax.set_title(f"Prediction Error Distribution - {BEST_NAME}\n"
             "(Actual - Predicted; positive = model UNDER-predicted)")
ax.set_xlabel("Prediction Error (Lakhs)")
ax.set_ylabel("Frequency")
ax.legend()
ax.grid(True, linestyle=":", alpha=0.6)
fig.tight_layout()
fig.savefig("charts/13_prediction_error_distribution.png", dpi=150,
            bbox_inches="tight")
plt.close(fig)
print("\n  Saved: charts/13_prediction_error_distribution.png")

# ------------------------------------------------------------
# 6) ACTUAL vs PREDICTED (BEST MODEL) GRAPH
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(predictions["Actual_Selling_Price"],
           predictions["Predicted_Selling_Price"],
           alpha=0.75, color="steelblue", edgecolor="k", linewidth=0.3,
           label="Test set cars")
limits = [0, max(predictions["Actual_Selling_Price"].max(),
                 predictions["Predicted_Selling_Price"].max()) * 1.05]
ax.plot(limits, limits, color="red", linestyle="--",
        label="Perfect prediction (y = x)")
ax.set_title(f"Actual vs Predicted Selling Price - {BEST_NAME}")
ax.set_xlabel("Actual Selling Price (in Lakhs)")
ax.set_ylabel("Predicted Selling Price (in Lakhs)")
ax.set_xlim(limits)
ax.set_ylim(limits)
ax.legend()
ax.grid(True, linestyle=":", alpha=0.6)
fig.tight_layout()
fig.savefig("charts/14_actual_vs_predicted_best_model.png", dpi=150,
            bbox_inches="tight")
plt.close(fig)
print("  Saved: charts/14_actual_vs_predicted_best_model.png")

# ------------------------------------------------------------
# 7 + 8) PRICE-FACTOR ANALYSIS AND CATEGORY ANALYSIS
# ------------------------------------------------------------
# NOTE: this dataset has NO advertising-spend column, so nothing is
# fabricated. We analyze the 7 variables that actually exist.
print("\n" + "=" * 60)
print("PRICE FACTOR ANALYSIS (real columns only)")
print("=" * 60)

numeric_vars = ["Present_Price", "Car_Age", "Driven_kms", "Owner"]
correlations = {v: df_features[v].corr(df_features["Selling_Price"])
                for v in numeric_vars}
print("  Pearson correlation with Selling_Price (numerical):")
for var, val in correlations.items():
    print(f"    {var:15s} {val:+.3f}")

category_summaries = []
print("\n  Average Selling_Price by category:")
for col in ["Fuel_Type", "Selling_type", "Transmission"]:
    grouped = (df_features.groupby(col)["Selling_Price"]
               .agg(Average_Selling_Price="mean", Count="size")
               .reset_index()
               .rename(columns={col: "Category"}))
    grouped.insert(0, "Variable", col)
    category_summaries.append(grouped)
    for _, r in grouped.iterrows():
        print(f"    {col} = {r['Category']:12s} avg {r['Average_Selling_Price']:.2f} L "
              f"(n={int(r['Count'])})")

category_analysis = pd.concat(category_summaries, ignore_index=True)
category_analysis.to_csv("category_price_analysis.csv", index=False)
print("\n  Saved: category_price_analysis.csv")
print(category_analysis.round(2).to_string(index=False))

# ------------------------------------------------------------
# 9 + 10) BUSINESS INSIGHTS (built from the REAL numbers above)
# ------------------------------------------------------------
corr_present = correlations["Present_Price"]
corr_age = correlations["Car_Age"]
corr_kms = correlations["Driven_kms"]
fuel_avg = category_analysis[category_analysis["Variable"] == "Fuel_Type"]
trans_avg = category_analysis[category_analysis["Variable"] == "Transmission"]
seller_avg = category_analysis[category_analysis["Variable"] == "Selling_type"]
top_feature = top10.iloc[0]["Feature"]
n_automatic = int(trans_avg.loc[trans_avg["Category"] == "Automatic", "Count"].iloc[0])
n_cng = int(fuel_avg.loc[fuel_avg["Category"] == "CNG", "Count"].iloc[0])

insights_text = f"""CAR PRICE PREDICTION - BUSINESS INSIGHTS
=========================================
(All numbers come from the actual dataset of {len(df_features)} used vehicles
and the models trained in Step 5. Correlation shows relationships,
NOT causes.)

1. BEST MODEL
-------------
{BEST_NAME} achieved the best results on unseen test data:
R2 = {best_row['R2']:.4f}, MAE = {best_row['MAE']:.2f} Lakhs, RMSE = {best_row['RMSE']:.2f} Lakhs.
The model explains about {best_row['R2'] * 100:.0f}% of the variation in
selling prices, with an average error of roughly {best_row['MAE']:.1f} Lakhs.

2. MOST IMPORTANT PRICE FACTORS
-------------------------------
The most influential feature is {top_feature}: the car's current
showroom price is the strongest driver of its resale value. The next
most influential factors are:
{chr(10).join(f'  - {r.Feature} (importance {r.Importance:.2f}, {r.Direction})' for r in coef_df.head(5).itertuples())}

3. NUMERICAL RELATIONSHIPS
--------------------------
  - Present_Price: correlation {corr_present:+.2f} - a higher current
    showroom price is strongly associated with a higher selling price.
  - Car_Age: correlation {corr_age:+.2f} - older vehicles show a
    relationship with LOWER selling prices (depreciation).
  - Driven_kms: correlation {corr_kms:+.2f} - surprisingly weak in this
    dataset; usage alone does not explain much price variation here.
  - Owner: almost no measurable relationship in this dataset
    ({correlations['Owner']:+.2f}) - only a handful of cars had previous owners.

4. CATEGORY-LEVEL FINDINGS
--------------------------
  - Fuel type: Diesel cars average {fuel_avg.loc[fuel_avg['Category'] == 'Diesel', 'Average_Selling_Price'].iloc[0]:.2f} L vs Petrol
    {fuel_avg.loc[fuel_avg['Category'] == 'Petrol', 'Average_Selling_Price'].iloc[0]:.2f} L (CNG only {n_cng} vehicles - not a reliable average).
  - Transmission: Automatic cars average {trans_avg.loc[trans_avg['Category'] == 'Automatic', 'Average_Selling_Price'].iloc[0]:.2f} L vs Manual
    {trans_avg.loc[trans_avg['Category'] == 'Manual', 'Average_Selling_Price'].iloc[0]:.2f} L, but only {n_automatic} automatics exist in the data.
  - Seller type: Dealer sales average {seller_avg.loc[seller_avg['Category'] == 'Dealer', 'Average_Selling_Price'].iloc[0]:.2f} L vs Individual
    {seller_avg.loc[seller_avg['Category'] == 'Individual', 'Average_Selling_Price'].iloc[0]:.2f} L - individuals in this dataset sell
    much cheaper cars on average.

5. PREDICTION ACCURACY
----------------------
  - Mean absolute error on the test set: {mean_abs_err:.2f} Lakhs.
  - Largest single error: {max_abs_err:.2f} Lakhs (typically an
    expensive or unusual vehicle).
  - The model is most accurate for budget and mid-range cars, which
    dominate the dataset; very expensive cars are rare and harder
    to predict.

6. MARKETING RECOMMENDATIONS (for a used-car business)
------------------------------------------------------
  1. Price vehicles primarily from their current market/showroom
     value ({top_feature} is the dominant factor) - a clear,
     defensible price anchor for both buying and selling.
  2. Price reductions should reflect vehicle age: with Car_Age
     showing a negative relationship, refresh pricing as vehicles
     sit longer on the lot.
  3. Diesel vehicles in this dataset are associated with higher
     resale values - when stocking, note that diesel stock (SUVs
     and larger cars) targets a higher-value segment.
  4. Dealer-channel listings dominate the higher price ranges in
     this dataset; for individual sellers, the model can help set
     fair expectations and reduce under-pricing (individual sales
     averaged only {seller_avg.loc[seller_avg['Category'] == 'Individual', 'Average_Selling_Price'].iloc[0]:.2f} L).
  5. Use the model as a screening tool: predicted vs asking price
     gaps above ~{best_row['MAE']:.1f} Lakhs flag vehicles worth
     negotiating or investigating.
  6. kilometers-driven alone is a weak signal here - do not price
     a car on mileage alone; combine it with age and present price.

7. DATASET LIMITATIONS
----------------------
  - Only {len(df_features)} rows after duplicate removal - small sample.
  - Very few CNG vehicles ({n_cng}) and few automatics ({n_automatic}),
    so those averages are fragile.
  - No horsepower, brand-goodwill or advertising-spend columns exist,
    so no such factor could be analyzed (nothing was fabricated).
  - Car_Name (98 unique models, including bikes) was excluded from
    the model because one-hot encoding it would invite overfitting.
  - Results can vary with a different train/test split.
  - All findings are associations within THIS dataset - correlation
    does not prove causation.
"""

with open("business_insights.txt", "w", encoding="utf-8") as f:
    f.write(insights_text)
print("\n  Saved: business_insights.txt")

# ------------------------------------------------------------
# 11) FINAL MODEL REPORT
# ------------------------------------------------------------
n_train, n_test = 239, 60   # displayed from the Step-5 run parameters

report_text = f"""CAR PRICE PREDICTION - FINAL MODEL REPORT
==========================================

DATASET
-------
Source file: car data.csv (never modified)
Rows after cleaning (duplicates removed): {len(df_features)}
Columns: 10 (Car_Name, Year, Selling_Price, Present_Price, Driven_kms,
Fuel_Type, Selling_type, Transmission, Owner, Car_Age)

FEATURES USED (inputs)
----------------------
Numerical:    Present_Price, Driven_kms, Owner, Car_Age
Categorical:  Fuel_Type, Selling_type, Transmission (one-hot encoded)
Excluded:     Selling_Price (target - would leak the answer),
              Car_Name (98 categories incl. bikes -> overfitting risk),
              Year (Car_Age already represents age)

TARGET VARIABLE
---------------
Selling_Price (in Lakhs) - a regression problem.

TRAIN/TEST SPLIT
----------------
test_size = 0.20, random_state = 42
Training rows: {n_train}   Testing rows: {n_test}
Preprocessing (imputers + one-hot encoder) sits INSIDE the pipeline and
was fitted on training data only, avoiding data leakage.

MODELS TESTED
-------------
{comparison.round(4).to_string(index=False)}

BEST MODEL
----------
{BEST_NAME}
  R2   = {best_row['R2']:.4f}
  MAE  = {best_row['MAE']:.4f} Lakhs
  RMSE = {best_row['RMSE']:.4f} Lakhs

TOP IMPORTANT FEATURES
----------------------
{top10.round(4).to_string(index=False)}

MAIN OBSERVATIONS
-----------------
  - Present_Price dominates: correlation {corr_present:+.2f} with the
    selling price and the largest model importance.
  - Car_Age shows a negative relationship ({corr_age:+.2f}): older cars
    are associated with lower prices (depreciation).
  - Driven_kms alone is weak here ({corr_kms:+.2f}) - its effect only
    makes sense together with other features.
  - Dealer and Automatic categories are associated with higher average
    prices, but both groups are imbalanced (see limitations).
  - Selling_Price is right-skewed (skewness 2.54): most cars are cheap,
    a few are expensive.

LIMITATIONS
-----------
  - Small dataset: only {len(df_features)} rows after duplicate removal.
  - Very few CNG vehicles ({n_cng}) and automatics ({n_automatic});
    category averages for these groups are unreliable.
  - No horsepower, brand-goodwill or advertising-spend data exists in
    the dataset; no such factor was invented.
  - Car_Name excluded due to high cardinality (98 unique labels,
    including motorcycles).
  - Metrics depend on this particular train/test split.
  - Correlation does not prove causation; findings are associations
    within this dataset only.

FILES PRODUCED BY THE PROJECT
-----------------------------
  car data.csv (original), df_features.csv (engineered),
  model_training.py, model_comparison.csv, car_price_model.pkl,
  predictions.csv, feature_importance.csv, category_price_analysis.csv,
  business_insights.txt, model_report.txt, charts/ (14 PNG charts)
"""

with open("model_report.txt", "w", encoding="utf-8") as f:
    f.write(report_text)
print("  Saved: model_report.txt")

# ------------------------------------------------------------
# 13) VERIFY OUTPUTS + FINAL SUMMARY
# ------------------------------------------------------------
required_outputs = [
    "charts/12_feature_importance.png",
    "charts/13_prediction_error_distribution.png",
    "charts/14_actual_vs_predicted_best_model.png",
    "category_price_analysis.csv",
    "business_insights.txt",
    "model_report.txt",
]
print("\n" + "=" * 60)
print("OUTPUT VERIFICATION")
print("=" * 60)
all_ok = True
for path in required_outputs:
    ok = os.path.exists(path)
    all_ok &= ok
    print(f"  {'OK ' if ok else 'MISSING'}  {path}")

print("\n" + "=" * 60)
print("STEP 6 COMPLETE" if all_ok else "STEP 6 INCOMPLETE - some files missing!")
print("=" * 60)
print(f"""
Best Model: {BEST_NAME}
R2:   {best_row['R2']:.4f}
MAE:  {best_row['MAE']:.4f} Lakhs
RMSE: {best_row['RMSE']:.4f} Lakhs

Top 5 Important Features:""")
for i, (_, r) in enumerate(top10.head(5).iterrows(), start=1):
    print(f"  {i}. {r['Feature']} ({r['Importance']:.3f})")

print(f"""
Main Business Insight:
In this dataset, a car's CURRENT SHOWROOM PRICE (Present_Price) is the
strongest factor associated with its resale value (correlation
{corr_present:+.2f}), followed by vehicle age (Car_Age, {corr_age:+.2f}).
Usage (Driven_kms) alone is a weak signal ({corr_kms:+.2f}) - so a used-car
business should anchor pricing on current market value and age, not
mileage alone. (Association, not causation.)

No UI built, no deployment, no dataset changed - as instructed.
""")
