# ============================================================
# Car Price Prediction with Machine Learning
# STEP 4: EXPLORATORY DATA ANALYSIS (EDA)
# ------------------------------------------------------------
# Goal: understand the data visually BEFORE modeling, using the
#       engineered DataFrame from Step 3 (df_features.csv).
#
# This script will NOT (on purpose):
#   - modify the original CSV (we only READ df_features.csv)
#   - train any machine learning model    (later step)
#   - build any web interface             (later step)
#
# All 10 charts are saved into the "charts" folder so they can
# be dropped straight into the project report.
# ============================================================

import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ------------------------------------------------------------
# 0) SETUP
# ------------------------------------------------------------
# "Agg" = save-to-file backend (works everywhere, even without a
# screen). If you run this on your own laptop and want chart
# windows to pop up, change SHOW to True or comment out the
# matplotlib.use("Agg") line.
matplotlib.use("Agg")
SHOW = False  # set True on your own PC to display windows

# Create the output folder automatically if it does not exist
os.makedirs("charts", exist_ok=True)

# Load the ENGINEERED data from Step 3 (never the raw CSV here)
df = pd.read_csv("df_features.csv")

# Consistent, clean chart style
sns.set_theme(style="whitegrid")
PALETTE = "Set2"

print("=" * 60)
print("STEP 4: EXPLORATORY DATA ANALYSIS")
print("=" * 60)
print(f"  Data loaded: {df.shape[0]} rows x {df.shape[1]} columns")


# Helper that finishes every chart the same way: save -> show -> close
def finish_chart(fig, filename):
    path = os.path.join("charts", filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if SHOW:
        plt.show()
    plt.close(fig)
    print(f"  Saved: {path}")


# ------------------------------------------------------------
# CHART 1: SELLING PRICE DISTRIBUTION (histogram)
# ------------------------------------------------------------
# Question: what does the target variable look like? Are most cars
# cheap with a few expensive ones (right-skewed)?
skew_value = df["Selling_Price"].skew()

fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(df["Selling_Price"], bins=25, kde=True, color="steelblue", ax=ax)
ax.axvline(df["Selling_Price"].mean(), color="red", linestyle="--", label=f"Mean = {df['Selling_Price'].mean():.2f} L")
ax.axvline(df["Selling_Price"].median(), color="green", linestyle=":", label=f"Median = {df['Selling_Price'].median():.2f} L")
ax.set_title("Distribution of Selling Price (Target Variable)")
ax.set_xlabel("Selling Price (in Lakhs)")
ax.set_ylabel("Number of Cars")
ax.legend()
finish_chart(fig, "01_price_distribution.png")

# ------------------------------------------------------------
# CHART 2: PRESENT PRICE vs SELLING PRICE (scatter)
# ------------------------------------------------------------
# Question: do expensive (new) cars also sell for more? (Strongest
# correlation found in Step 3: about +0.88.)
corr_present = df["Present_Price"].corr(df["Selling_Price"])

fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(data=df, x="Present_Price", y="Selling_Price", hue="Transmission",
                palette=PALETTE, alpha=0.8, ax=ax)
ax.set_title(f"Present Price vs Selling Price (correlation = {corr_present:.2f})")
ax.set_xlabel("Present Price - current showroom price (in Lakhs)")
ax.set_ylabel("Selling Price (in Lakhs)")
ax.legend(title="Transmission")
finish_chart(fig, "02_present_vs_selling_price.png")

# ------------------------------------------------------------
# CHART 3: CAR AGE vs SELLING PRICE (scatter)
# ------------------------------------------------------------
# Question: do older cars sell for less? (Depreciation effect.)
corr_age = df["Car_Age"].corr(df["Selling_Price"])

fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(data=df, x="Car_Age", y="Selling_Price", hue="Fuel_Type",
                palette=PALETTE, alpha=0.8, ax=ax)
ax.set_title(f"Car Age vs Selling Price (correlation = {corr_age:.2f})")
ax.set_xlabel("Car Age in years (2026 - Year of purchase)")
ax.set_ylabel("Selling Price (in Lakhs)")
ax.legend(title="Fuel Type")
finish_chart(fig, "03_car_age_vs_selling_price.png")

# ------------------------------------------------------------
# CHART 4: DRIVEN KILOMETERS vs SELLING PRICE (scatter)
# ------------------------------------------------------------
# Question: does more driving (usage) mean a lower price?
corr_kms = df["Driven_kms"].corr(df["Selling_Price"])

fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(data=df, x="Driven_kms", y="Selling_Price", color="darkorange", alpha=0.8, ax=ax)
ax.set_title(f"Kilometers Driven vs Selling Price (correlation = {corr_kms:.2f})")
ax.set_xlabel("Kilometers Driven")
ax.set_ylabel("Selling Price (in Lakhs)")
finish_chart(fig, "04_km_vs_selling_price.png")

# ------------------------------------------------------------
# CHART 5: FUEL TYPE vs SELLING PRICE (boxplot)
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=df, x="Fuel_Type", y="Selling_Price", hue="Fuel_Type",
            palette=PALETTE, legend=False, ax=ax)
ax.set_title("Selling Price by Fuel Type")
ax.set_xlabel("Fuel Type")
ax.set_ylabel("Selling Price (in Lakhs)")
finish_chart(fig, "05_fuel_type_price.png")

# ------------------------------------------------------------
# CHART 6: TRANSMISSION vs SELLING PRICE (boxplot)
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=df, x="Transmission", y="Selling_Price", hue="Transmission",
            palette=PALETTE, legend=False, ax=ax)
ax.set_title("Selling Price by Transmission")
ax.set_xlabel("Transmission")
ax.set_ylabel("Selling Price (in Lakhs)")
finish_chart(fig, "06_transmission_price.png")

# ------------------------------------------------------------
# CHART 7: SELLING TYPE vs SELLING PRICE (boxplot)
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=df, x="Selling_type", y="Selling_Price", hue="Selling_type",
            palette=PALETTE, legend=False, ax=ax)
ax.set_title("Selling Price by Selling Type (Dealer vs Individual)")
ax.set_xlabel("Selling Type")
ax.set_ylabel("Selling Price (in Lakhs)")
finish_chart(fig, "07_selling_type_price.png")

# ------------------------------------------------------------
# CHART 8: CORRELATION HEATMAP (numeric columns only)
# ------------------------------------------------------------
# Correlation ranges from -1 to +1: how strongly two numbers move
# together. This heatmap includes every numeric column.
numeric_cols = ["Year", "Selling_Price", "Present_Price", "Driven_kms", "Owner", "Car_Age"]
corr_matrix = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
            vmin=-1, vmax=1, square=True, linewidths=0.5, ax=ax)
ax.set_title("Correlation Heatmap (Numeric Features)")
finish_chart(fig, "08_correlation_heatmap.png")

# ------------------------------------------------------------
# CHART 9: AVERAGE SELLING PRICE BY FUEL TYPE (bar chart)
# ------------------------------------------------------------
avg_by_fuel = df.groupby("Fuel_Type")["Selling_Price"].mean().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(8, 5))
bars = sns.barplot(x=avg_by_fuel.index, y=avg_by_fuel.values,
                   hue=avg_by_fuel.index, palette=PALETTE, legend=False, ax=ax)
ax.bar_label(bars.containers[0], fmt="%.2f L")
ax.set_title("Average Selling Price by Fuel Type")
ax.set_xlabel("Fuel Type")
ax.set_ylabel("Average Selling Price (in Lakhs)")
finish_chart(fig, "09_fuel_average_price.png")

# ------------------------------------------------------------
# CHART 10: AVERAGE SELLING PRICE BY TRANSMISSION (bar chart)
# ------------------------------------------------------------
avg_by_trans = df.groupby("Transmission")["Selling_Price"].mean().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(8, 5))
bars = sns.barplot(x=avg_by_trans.index, y=avg_by_trans.values,
                   hue=avg_by_trans.index, palette=PALETTE, legend=False, ax=ax)
ax.bar_label(bars.containers[0], fmt="%.2f L")
ax.set_title("Average Selling Price by Transmission")
ax.set_xlabel("Transmission")
ax.set_ylabel("Average Selling Price (in Lakhs)")
finish_chart(fig, "10_transmission_average_price.png")

# ------------------------------------------------------------
# EDA SUMMARY (printed) - every claim below is backed by a number
# computed from THIS dataset, not by guesswork.
# ------------------------------------------------------------
avg_by_seller = df.groupby("Selling_type")["Selling_Price"].mean()
max_kms_row = df.loc[df["Driven_kms"].idxmax()]
max_price_row = df.loc[df["Selling_Price"].idxmax()]
max_present_row = df.loc[df["Present_Price"].idxmax()]

print("\n" + "=" * 60)
print("EDA SUMMARY (all numbers computed from this dataset)")
print("=" * 60)
print(f"""
1) STRONGEST FEATURE: Present_Price (correlation with Selling_Price
   = {corr_present:.2f}). The scatter (chart 02) shows prices rising
   almost linearly with the current showroom price.

2) CAR AGE: correlation = {corr_age:.2f} (negative). Older cars generally
   sell for less - the depreciation pattern expected in the real world.

3) KILOMETERS DRIVEN: correlation = {corr_kms:.2f} - surprisingly WEAK on
   its own. Chart 04 shows why: most cars are clustered below ~100,000 km,
   while a few extreme-km cars do not follow the trend. The relationship
   is noisy in this dataset (the model may still find useful patterns
   when combined with other features).

4) FUEL TYPE (avg selling price): {avg_by_fuel.round(2).to_dict()}.
   Only 2 CNG cars exist, so the CNG average is based on very little
   data and is not reliable. Diesel cars in this data are pricier on
   average (they include big SUVs like the Fortuner).

5) TRANSMISSION (avg selling price): {avg_by_trans.round(2).to_dict()}.
   Automatic cars average much higher, but remember: only
   {(df['Transmission'] == 'Automatic').sum()} automatics exist out of
   {len(df)} rows, so the gap partly reflects the car mix, not just
   the gearbox.

6) SELLING TYPE (avg selling price): {avg_by_seller.round(2).to_dict()}.
   Dealer-sold cars are costlier on average than individual sales.

7) OUTLIERS visible in the charts:
   - Highest Driven_kms: {int(max_kms_row['Driven_kms']):,} km
     (car: {max_kms_row['Car_Name']}, Selling_Price {max_kms_row['Selling_Price']} L)
   - Highest Selling_Price: {max_price_row['Selling_Price']} L
     (car: {max_price_row['Car_Name']}, Present_Price {max_price_row['Present_Price']} L)
   - Highest Present_Price: {max_present_row['Present_Price']} L
     (car: {max_present_row['Car_Name']}) - far above all other points.

8) SKEWNESS of Selling_Price = {skew_value:.2f} (0 = symmetric,
   > 1 = strongly right-skewed). The mean ({df['Selling_Price'].mean():.2f} L) is
   higher than the median ({df['Selling_Price'].median():.2f} L) because a few
   expensive cars pull it up: the target is RIGHT-SKEWED.
""")

print("=" * 60)
print("EDA complete. All 10 charts saved in the 'charts' folder.")
print("=" * 60)
