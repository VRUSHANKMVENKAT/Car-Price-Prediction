# Car Price Prediction with Machine Learning

A complete, beginner-friendly machine learning project that estimates the **selling price of a used vehicle** from its characteristics — from data cleaning to a deployed **Streamlit web application**.

---

## 1. Project Overview

This project uses machine learning regression techniques to estimate the selling price of a used vehicle based on vehicle characteristics.

The model uses:

- `Present_Price` – current showroom price (Lakhs)
- `Driven_kms` – kilometers driven
- `Owner` – number of previous owners
- `Car_Age` – vehicle age in years
- `Fuel_Type` – Petrol / Diesel / CNG
- `Selling_type` – Dealer / Individual
- `Transmission` – Manual / Automatic

**Target:** `Selling_Price` (measured in Indian Lakhs, where 1 Lakh = ₹100,000).

> ⚠️ The model produces a statistical **estimate**, not the exact market price.

---

## 2. Problem Statement

Selling a used car raises the same question for dealers, marketplaces and individual sellers: **"What is a fair price for this vehicle?"** Pricing too high deters buyers; pricing too low loses money. This project builds a regression model that learns pricing patterns from historical sales data and predicts a reasonable selling price for a vehicle described by its features.

---

## 3. Objectives

1. Understand and validate the dataset (inspection, missing values, duplicates, data types).
2. Clean the data without blindly deleting information.
3. Engineer a useful feature (`Car_Age`) without data leakage.
4. Explore the data visually (EDA) and document real relationships.
5. Train and compare multiple regression models with proper train/test discipline.
6. Evaluate with MAE, RMSE and R² and select the best model honestly.
7. Interpret the model (feature influence) and extract business insights.
8. Deploy the trained model as an interactive Streamlit application.

---

## 4. Dataset Description

| Stage | Rows | Columns |
|---|---|---|
| Original dataset (`car data.csv`) | 301 | 9 |
| After duplicate removal | 299 | 9 |
| After feature engineering (`df_features.csv`) | 299 | 10 |

Facts about the dataset (no invented statistics):

- **No missing values** were present in any column.
- **2 exact duplicate rows** were found and removed.
- `Car_Age` was added during feature engineering.
- The original `Year` column was retained in the engineered dataset but **excluded from the initial model** (age information is carried by `Car_Age`).
- `Selling_Price` remained the prediction target and was never altered.

---

## 5. Features Used

| Feature | Type | Description |
|---|---|---|
| Present_Price | Numerical | Current/showroom price in Lakhs |
| Driven_kms | Numerical | Kilometers driven |
| Owner | Numerical | Number of previous owners |
| Car_Age | Numerical | Vehicle age in years |
| Fuel_Type | Categorical | Petrol / Diesel / CNG |
| Selling_type | Categorical | Dealer / Individual |
| Transmission | Categorical | Manual / Automatic |
| **Selling_Price** | **Target** | Selling price in Lakhs |

The original dataset also contains **`Car_Name`** and **`Year`**. Why they are not model inputs:

- **`Selling_Price`** is the target — using it as an input would leak the answer into the model (**data leakage**).
- **`Car_Name`** was excluded from the initial model because of **high cardinality** (98 unique names, including motorcycles) — one-hot encoding it with only 299 rows would invite overfitting.
- **`Year`** was conceptually replaced by **`Car_Age`**, which expresses the same information more directly.
- **Horsepower, brand goodwill and advertising spend** were **not fabricated** — these columns do not exist in the dataset, so no such features were created.

---

## 6. Data Preprocessing

Performed in `data_preprocessing.py` (produces `df_clean`):

1. **Missing-value checking** — all columns checked with `isnull().sum()`; the dataset had **0 missing cells**. (The planned rule — median for numeric, mode for categorical — keeps every row instead of deleting data.)
2. **Duplicate checking** — 2 exact duplicate rows identified (with evidence printed).
3. **Duplicate removal** — only the exact copies were dropped (`drop_duplicates(keep="first")`); every remaining row is a real record.
4. **Data type handling** — numeric columns were already correct; the 4 text columns were converted to the `category` type.
5. **Invalid-value checks** — no negative prices/kms, no future years, all category labels valid.
6. **Outlier identification** — high-price and high-km vehicles were **reported, not deleted**: they are real (unusual but genuine) vehicles, and removing them blindly would discard true information.

The **target column was never altered** — verified by comparing every kept `Selling_Price` value against the original file.

---

## 7. Feature Engineering

Performed in `data_feature_engineering.py` (produces `df_features`):

```
Car_Age = 2026 - Year
```

**Why vehicle age is useful:** older vehicles generally tend to have lower resale values because of depreciation, although this relationship is not necessarily causal for every individual vehicle. Age is the concept buyers and sellers actually reason with ("a 12-year-old car"), so `Car_Age` makes the pattern easier for the model to learn and for humans to explain.

No feature was derived from `Selling_Price` (data-leakage prevention), and scaling was deliberately **not** applied at this stage — it is a model-specific requirement, to be handled inside a Pipeline if the chosen model needs it.

---

## 8. Exploratory Data Analysis

`data_analysis.py` generated the visualizations in `charts/`:

- Selling price distribution (histogram)
- Present price vs selling price (scatter)
- Car age vs selling price (scatter)
- Driven kilometers vs selling price (scatter)
- Fuel type / transmission / selling type price comparisons (boxplots)
- Correlation heatmap (numeric features)
- Average price by fuel type and by transmission (bar charts)
- Actual vs predicted prices (Step 5)
- Feature importance / coefficients (Step 6)
- Prediction error distribution (Step 6)

**Main observed relationships** (associations in *this* dataset, not causal claims):

| Feature | Relationship with Selling_Price |
|---|---|
| Present_Price | strongest, correlation ≈ **+0.88** |
| Car_Age | negative, ≈ **−0.23** (older → cheaper) |
| Driven_kms | weak, ≈ **+0.03** (mileage alone is a poor signal here) |

`Selling_Price` is **right-skewed** (skewness 2.54): most vehicles are budget cars, with a few expensive ones pulling the mean above the median.

---

## 9. Machine Learning Models

Three regression models were trained in `model_training.py`, each inside a scikit-learn **Pipeline** (preprocessing → model):

1. **Linear Regression** – fits a weighted linear combination of the features; simple, fast, and interpretable through its coefficients.
2. **Random Forest Regression** – an ensemble of 300 decision trees, each trained on random subsets; captures non-linear patterns and is robust to outliers.
3. **Gradient Boosting Regression** – builds trees sequentially, each correcting the previous one's errors; often strong on tabular data.

Preprocessing (`SimpleImputer` for gaps + `OneHotEncoder(handle_unknown="ignore")` for categories) lives **inside** the Pipeline and was fitted on **training data only** — preventing data leakage. Train/test split: `test_size=0.20`, `random_state=42` → **239 training / 60 testing rows**.

---

## 10. Model Evaluation

| Model | MAE (Lakhs) | RMSE (Lakhs) | R² |
|---|---|---|---|
| **Linear Regression** ⭐ | **1.4725** | **2.5245** | **0.7527** |
| Random Forest | 1.5320 | 3.6947 | 0.4703 |
| Gradient Boosting | 1.1762 | 2.6383 | 0.7299 |

*(Actual results from `model_comparison.csv`.)*

- **R² = 0.7527** means the model explains approximately **75.27%** of the variation in selling prices in the test set.
- **MAE = 1.4725 Lakhs** means the average absolute error was approximately **₹1.47 lakh** on the test set.
- **RMSE = 2.5245 Lakhs**; RMSE is more sensitive to **large errors** than MAE because errors are squared before averaging.

---

## 11. Best Model

**Linear Regression** was selected because it achieved the **highest test-set R²** (with lower RMSE as the tie-break) among the three tested models.

This does **not** mean Linear Regression is universally better than Random Forest or Gradient Boosting — model performance depends on the dataset, its size, and the train/test split. Here the Present_Price → Selling_Price relationship is nearly linear (correlation ≈ +0.88), which favors the linear model, while the tree ensembles had only 239 training rows and generalized worse.

---

## 12. Feature Importance

Because the selected model is Linear Regression (which has no `feature_importances_`), feature influence was analyzed using **absolute standardized coefficients** (|coefficient| × feature spread) in `model_analysis.py`:

1. **Present_Price** — the dominant factor
2. **Fuel_Type_Diesel** — associated with higher prices
3. **Car_Age** — associated with lower prices
4. **Fuel_Type_CNG** — ⚠️ based on only **2 CNG observations**, so this coefficient should **not be overinterpreted**
5. **Transmission_Manual** — associated with lower prices than Automatic

Full details: `feature_importance.csv` and `charts/12_feature_importance.png`.

---

## 13. Streamlit Application

`app.py` loads the saved pipeline (`car_price_model.pkl`) with **joblib** — no retraining happens.

The user enters in the sidebar:

- Present Price (₹ Lakhs)
- Kilometers Driven
- Previous Owners (0–3; the training data mainly observed 0, 1 and 3)
- Car Age (Years)
- Fuel Type, Selling Type, Transmission

The trained **Pipeline performs all preprocessing and prediction** on one raw pandas DataFrame with the exact trained feature names. The predicted selling price is displayed:

- in **Lakhs** (e.g. `₹ 6.39 Lakhs`), and
- in **approximate Indian rupee format** (e.g. `₹6,39,356`).

The application also shows: model performance metrics, key price factors, real-world use cases, an expandable EDA chart gallery (all 13 charts, each checked for existence before display), and the live model-comparison table with the best model highlighted. Invalid input, negative predictions and missing files are handled gracefully with warnings instead of crashes.

---

## 14. Real-World Applications

- **Used-car price estimation** when listing or valuing a vehicle
- **Dealer pricing support** for stocking and repricing decisions
- **Seller negotiation support** with an objective reference point
- **Comparing vehicle listings** against a data-driven benchmark
- **Identifying potentially overpriced or underpriced** vehicles
- **Data-driven pricing decisions** for marketplaces and individual sellers

> ⚠️ All predictions are **estimates** based on a small historical dataset — they are **not guaranteed market prices**.

---

## 15. Limitations

- Only **299 rows** after duplicate removal — a small sample.
- Very few **CNG** observations (2), so CNG-related results are unreliable.
- **Automatic** vehicles (39) are far outnumbered by Manual (260).
- The dataset contains **no horsepower** column.
- The dataset contains **no direct brand-goodwill** information.
- The dataset contains **no advertising expenditure**.
- `Car_Name` was excluded from the initial model because of **high cardinality** (98 labels, including motorcycles).
- Performance **depends on the train/test split**; a different split gives different metrics.
- **Extreme high-value vehicles** produce larger prediction errors (worst test error ≈ 11.4 Lakhs).
- **Correlation does not prove causation** — all findings are associations within this dataset.

---

## 16. Technologies Used

- **Python** 🐍
- **Pandas** – data manipulation
- **NumPy** – numerical computation
- **Scikit-learn** – modeling, pipelines, metrics
- **Matplotlib** – plotting
- **Seaborn** – statistical visualization
- **Streamlit** – web application
- **Joblib** – model serialization

---

## 17. Project Structure

```
Car-Price-Prediction/
│
├── car data.csv                      # Original dataset (never modified)
├── df_features.csv                   # Cleaned + engineered dataset
│
├── data_inspection.py                # Step 1: read-only data inspection
├── data_preprocessing.py             # Step 2: cleaning -> df_clean
├── data_feature_engineering.py       # Step 3: Car_Age -> df_features
├── data_analysis.py                  # Step 4: EDA + 10 charts
├── model_training.py                 # Step 5: train/evaluate 3 models
├── model_analysis.py                 # Step 6: analysis + business insights
├── app.py                            # Step 7: Streamlit web application
├── app_test.py                       # Headless sanity check of the model
│
├── model_comparison.csv              # MAE / RMSE / R2 for all models
├── predictions.csv                   # Actual vs predicted (test set)
├── feature_importance.csv            # Standardized coefficient importance
├── category_price_analysis.csv       # Average price by category
│
├── car_price_model.pkl               # Saved Pipeline (preprocessing + model)
│
├── business_insights.txt             # Business insights report
├── model_report.txt                  # Final model report
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
│
└── charts/
    ├── 01_price_distribution.png
    ├── 02_present_vs_selling_price.png
    ├── 03_car_age_vs_selling_price.png
    ├── 04_km_vs_selling_price.png
    ├── 05_fuel_type_price.png
    ├── 06_transmission_price.png
    ├── 07_selling_type_price.png
    ├── 08_correlation_heatmap.png
    ├── 09_fuel_average_price.png
    ├── 10_transmission_average_price.png
    ├── 11_actual_vs_predicted.png
    ├── 12_feature_importance.png
    ├── 13_prediction_error_distribution.png
    └── 14_actual_vs_predicted_best_model.png
```

---

## 18. How to Run

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Start the Streamlit application:**

   ```bash
   streamlit run app.py
   ```

3. **Open the displayed local URL** (typically `http://localhost:8501`) in your browser.

Optional checks (each script is independent and prints its own report):

```bash
python data_inspection.py          # Step 1: dataset inspection
python data_preprocessing.py       # Step 2: cleaning summary
python data_feature_engineering.py # Step 3: Car_Age + df_features.csv
python data_analysis.py            # Step 4: regenerate EDA charts
python model_training.py           # Step 5: retrain + compare models
python model_analysis.py           # Step 6: regenerate analysis reports
python app_test.py                 # headless model sanity check
```

*(The scripts regenerate the CSV/PNG/report outputs; the original `car data.csv` is never modified.)*

---

## 19. Example Prediction

Input entered in the app:

| Feature | Value |
|---|---|
| Present_Price | 6.0 Lakhs |
| Driven_kms | 30,000 |
| Owner | 0 |
| Car_Age | 5 |
| Fuel_Type | Petrol |
| Selling_type | Dealer |
| Transmission | Manual |

**Verified prediction: ≈ ₹6.39 Lakhs** (raw model output 6.3936 Lakhs, displayed as `₹6,39,356`).

> This is one worked example with this model — not a general expected price for every vehicle with those specifications.

---

## 20. Conclusion

This project demonstrates how regression-based machine learning can estimate used-vehicle selling prices from historical vehicle data. It covers the complete workflow — **preprocessing, feature engineering, exploratory data analysis, regression modeling, evaluation, model interpretation, and deployment through Streamlit** — with honest, reproducible results (every score in this README comes from actual runs, fixed with `random_state=42`).

The best model (Linear Regression, R² = 0.7527) shows that in this dataset a vehicle's current showroom price is the strongest factor associated with its resale value, followed by vehicle age — while mileage alone is a weak signal. Beyond the numbers, the project illustrates disciplined ML practice: preventing data leakage, not deleting unusual data blindly, reporting limitations, and never presenting an estimate as a guaranteed market price.

---

*Developed as a Machine Learning college project.*
