# ============================================================
# CAR PRICE PREDICTOR - Streamlit Web Application (UI v2)
# ------------------------------------------------------------
# Design pass: premium automotive analytics look & feel.
#
# FUNCTIONALITY IS UNCHANGED: this app LOADS the trained
# pipeline from Step 5 (car_price_model.pkl), never retrains,
# never modifies the dataset, and sends RAW feature values with
# the exact column names the pipeline was trained on. All
# metrics are read live from model_comparison.csv.
#
# Run with:  streamlit run app.py
# ============================================================

import os

import joblib
import pandas as pd
import streamlit as st

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
MODEL_FILE = "car_price_model.pkl"
COMPARISON_FILE = "model_comparison.csv"
CHARTS_DIR = "charts"

# Exact feature names/column order the trained pipeline expects
FEATURE_COLUMNS = ["Present_Price", "Driven_kms", "Owner", "Car_Age",
                   "Fuel_Type", "Selling_type", "Transmission"]

NAV_PAGES = ["Predict Price", "Market Insights", "Model Performance", "About"]

# ------------------------------------------------------------
# DESIGN SYSTEM (single style block - premium dark automotive)
# ------------------------------------------------------------
CSS = """
<style>
/* ---------- Design tokens ---------- */
:root {
  --bg: #0d1017;            /* near-black charcoal   */
  --surface: #151a23;       /* card surface          */
  --surface-2: #1b212c;     /* raised surface        */
  --border: #262e3b;
  --text: #f2f4f8;          /* off-white             */
  --text-dim: #9aa4b2;      /* neutral gray          */
  --accent: #e5484d;        /* racing red (CTA only) */
  --accent-soft: rgba(229, 72, 77, 0.12);
  --ok: #46c98b;
}

/* ---------- Base typography & spacing ---------- */
html, body, [class*="css"] {
  font-family: "Segoe UI", system-ui, -apple-system, "Helvetica Neue", sans-serif;
}
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1200px; }
h1, h2, h3 { color: var(--text); letter-spacing: -0.01em; }
p, li { color: var(--text-dim); }
a { color: var(--accent); }
hr { border-color: var(--border) !important; }

/* ---------- Hero ---------- */
.cp-hero { padding: 1.4rem 0 0.4rem 0; }
.cp-eyebrow {
  color: var(--accent); font-size: 0.8rem; font-weight: 700;
  letter-spacing: 0.28em; text-transform: uppercase; margin-bottom: 0.6rem;
}
.cp-hero-title {
  color: var(--text); font-size: clamp(2rem, 4.5vw, 3.2rem); font-weight: 800;
  line-height: 1.12; letter-spacing: -0.02em; margin: 0 0 0.7rem 0;
}
.cp-hero-sub { color: var(--text-dim); font-size: 1.05rem; max-width: 640px; line-height: 1.55; }

.cp-status {
  display: inline-flex; align-items: center; gap: 0.5rem;
  margin-top: 1.1rem; padding: 0.42rem 0.95rem;
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 999px; font-size: 0.8rem; color: var(--text-dim);
}
.cp-status .dot {
  width: 8px; height: 8px; border-radius: 50%;
  background: var(--ok); box-shadow: 0 0 8px rgba(70, 201, 139, 0.8);
}
.cp-status b { color: var(--text); font-weight: 600; }

/* ---------- Cards ---------- */
.cp-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 14px; padding: 1.25rem 1.35rem; margin-bottom: 1rem;
}
.cp-card-title {
  color: var(--text); font-weight: 700; font-size: 0.95rem; margin-bottom: 0.15rem;
}
.cp-card-sub { color: var(--text-dim); font-size: 0.85rem; }

/* ---------- Result card ---------- */
.cp-result {
  background:
    radial-gradient(1200px 300px at 20% -40%, rgba(229, 72, 77, 0.16), transparent 60%),
    linear-gradient(180deg, var(--surface-2), var(--surface));
  border: 1px solid var(--border); border-radius: 18px;
  padding: 2rem 2.2rem; text-align: center;
}
.cp-result .label {
  color: var(--text-dim); font-size: 0.78rem; font-weight: 700;
  letter-spacing: 0.24em; text-transform: uppercase;
}
.cp-result .price {
  color: var(--text); font-size: clamp(2.6rem, 6vw, 4rem);
  font-weight: 800; letter-spacing: -0.02em; line-height: 1.1; margin: 0.5rem 0 0.2rem 0;
}
.cp-result .price span { color: var(--accent); }
.cp-result .sub { color: var(--text-dim); font-size: 0.92rem; }

/* ---------- KPI cards ---------- */
.cp-kpi {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 14px; padding: 1.05rem 1.2rem; text-align: center; height: 100%;
}
.cp-kpi .k-label {
  color: var(--text-dim); font-size: 0.72rem; font-weight: 700;
  letter-spacing: 0.16em; text-transform: uppercase;
}
.cp-kpi .k-value { color: var(--text); font-size: 1.55rem; font-weight: 800; margin-top: 0.3rem; }
.cp-kpi .k-note { color: var(--text-dim); font-size: 0.78rem; margin-top: 0.15rem; }
.cp-kpi.accent .k-value { color: var(--accent); }

/* ---------- Steps (How it works) ---------- */
.cp-step { display: flex; gap: 1rem; align-items: flex-start; padding: 0.85rem 0; }
.cp-step .num {
  flex: 0 0 auto; width: 42px; height: 42px; border-radius: 12px;
  display: flex; align-items: center; justify-content: center;
  background: var(--accent-soft); color: var(--accent);
  font-weight: 800; font-size: 0.95rem; border: 1px solid rgba(229, 72, 77, 0.35);
}
.cp-step .t { color: var(--text); font-weight: 700; font-size: 0.98rem; }
.cp-step .d { color: var(--text-dim); font-size: 0.88rem; margin-top: 0.1rem; line-height: 1.45; }

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
  background: #0a0d13; border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .cp-brand {
  font-weight: 800; letter-spacing: 0.22em; font-size: 0.78rem;
  color: var(--text); text-transform: uppercase;
}
section[data-testid="stSidebar"] .cp-brand span { color: var(--accent); }
section[data-testid="stSidebar"] hr { margin: 0.4rem 0 0.9rem 0; }
section[data-testid="stSidebar"] .cp-group {
  color: var(--text-dim); font-size: 0.7rem; font-weight: 700;
  letter-spacing: 0.18em; text-transform: uppercase;
  margin: 1rem 0 0.15rem 0;
}
section[data-testid="stSidebar"] .block-container { padding-top: 1.4rem; }

/* Nav radio -> clean vertical pills */
section[data-testid="stSidebar"] div[role="radiogroup"] { gap: 0.25rem; }
section[data-testid="stSidebar"] div[role="radiogroup"] label {
  background: transparent; border: 1px solid transparent;
  border-radius: 10px; padding: 0.45rem 0.7rem; margin: 0;
  transition: background 0.15s ease, border-color 0.15s ease;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover { background: var(--surface); }
section[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"],
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
  background: var(--surface-2); border-color: var(--border);
}

/* ---------- CTA button ---------- */
.stButton > button {
  width: 100%; border: none; border-radius: 12px;
  background: var(--accent); color: #ffffff !important;
  font-weight: 700; font-size: 1rem; padding: 0.7rem 1rem;
  box-shadow: 0 6px 18px rgba(229, 72, 77, 0.28);
  transition: transform 0.12s ease, filter 0.12s ease;
}
.stButton > button:hover { filter: brightness(1.08); transform: translateY(-1px); }
.stButton > button:active { transform: translateY(0); }

/* ---------- Misc polish ---------- */
.cp-empty {
  border: 1px dashed var(--border); border-radius: 16px;
  padding: 3rem 1.5rem; text-align: center;
}
.cp-empty .big { color: var(--text); font-weight: 700; font-size: 1.1rem; }
.cp-empty .small { color: var(--text-dim); font-size: 0.9rem; margin-top: 0.3rem; }
.cp-note { color: var(--text-dim); font-size: 0.8rem; }
.cp-footer {
  margin-top: 2.5rem; padding-top: 1.2rem; border-top: 1px solid var(--border);
  color: var(--text-dim); font-size: 0.82rem;
  display: flex; justify-content: space-between; flex-wrap: wrap; gap: 0.5rem;
}
div[data-testid="stExpander"] {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 14px; overflow: hidden;
}
div[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 12px; }
</style>
"""

st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(CSS, unsafe_allow_html=True)


# ------------------------------------------------------------
# MODEL LOADING (unchanged behavior, cached)
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


@st.cache_data
def load_comparison():
    return pd.read_csv(COMPARISON_FILE)


try:
    model = load_model()
except FileNotFoundError:
    st.error(f"{MODEL_FILE} was not found. Please run model_training.py "
             "first, then restart this app.")
    st.stop()
except Exception as exc:
    st.error(f"The model file could not be loaded: {exc}")
    st.stop()

try:
    comparison = load_comparison()
except Exception:
    comparison = None          # pages degrade gracefully if this is missing


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def to_lakhs_rupees(price_in_lakhs, symbol="₹"):
    """Convert Lakhs to an Indian-formatted rupee string (₹X,XX,XXX)."""
    rupees = int(round(price_in_lakhs * 100000))
    if rupees >= 10000000:                       # 1 crore and above
        crores, rest = divmod(rupees, 10000000)
        text = f"{crores},{rest:07d}"
    else:
        text = str(rupees)
    if len(text) > 3:                            # Indian grouping
        head, tail = text[:-3], text[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        text = ",".join(groups) + "," + tail
    return f"{symbol}{text}"


def chart_path(filename):
    path = os.path.join(CHARTS_DIR, filename)
    return path if os.path.exists(path) else None


def show_chart(filename, caption, use_columns=False):
    """Display an existing chart if present; silent fallback otherwise."""
    path = chart_path(filename)
    if path:
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.caption(f"Chart not available: {filename}")


def best_model_row():
    if comparison is None or comparison.empty:
        return None
    return comparison.sort_values(by=["R2", "RMSE"],
                                  ascending=[False, True]).iloc[0]


# ------------------------------------------------------------
# SIDEBAR: brand, navigation, vehicle inputs, CTA
# ------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="cp-brand">CAR PRICE <span>PREDICTOR</span></div>',
                unsafe_allow_html=True)
    st.divider()

    page = st.radio("Navigation", NAV_PAGES, label_visibility="collapsed")

    st.markdown('<div class="cp-group">Vehicle Basics</div>',
                unsafe_allow_html=True)
    present_price = st.slider(
        "Present Price (₹ Lakhs)", min_value=0.5, max_value=50.0,
        value=6.0, step=0.1, key="pp",
        help="Current showroom price of a NEW version of this car. "
             "1 Lakh = ₹100,000.")
    driven_kms = st.slider(
        "Kilometers Driven", min_value=0, max_value=300000,
        value=30000, step=5000, key="kms",
        help="Total distance the vehicle has been driven.")

    st.markdown('<div class="cp-group">Ownership & Age</div>',
                unsafe_allow_html=True)
    owner = st.selectbox(
        "Previous Owners", options=[0, 1, 2, 3], index=0,
        help="Training data mainly observed 0, 1 and 3 previous owners; "
             "2 is offered as a user-friendly option but is an "
             "extrapolation beyond the observed categories.")
    car_age = st.slider("Car Age (Years)", min_value=0, max_value=30,
                        value=5, step=1, key="age")

    st.markdown('<div class="cp-group">Specifications</div>',
                unsafe_allow_html=True)
    fuel_type = st.selectbox("Fuel Type", options=["Petrol", "Diesel", "CNG"])
    selling_type = st.selectbox("Selling Type", options=["Dealer", "Individual"])
    transmission = st.selectbox("Transmission", options=["Manual", "Automatic"])

    st.divider()
    predict_clicked = st.button("Estimate Price", use_container_width=True)

    if page != "Predict Price":
        st.caption("Switch to “Predict Price” to review or change the "
                   "vehicle details used for the estimate.")

# ------------------------------------------------------------
# SHARED: hero
# ------------------------------------------------------------
def render_hero():
    st.markdown(
        """
        <div class="cp-hero">
          <div class="cp-eyebrow">Car Price Predictor</div>
          <div class="cp-hero-title">Know What Your Car<br>Is Really Worth.</div>
          <div class="cp-hero-sub">Estimate a vehicle's market value using a
          machine-learning model trained on historical car data.</div>
          <div class="cp-status"><span class="dot"></span>
            <b>ML MODEL READY</b>&nbsp;·&nbsp;AI-powered price estimation</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer():
    st.markdown(
        """
        <div class="cp-footer">
          <div><b style="color:#f2f4f8">Car Price Predictor</b><br>
          Machine Learning · Vehicle Analytics</div>
          <div>Built with Python, Streamlit &amp; Machine Learning</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# PAGE: PREDICT PRICE
# ============================================================
if page == "Predict Price":
    render_hero()
    st.write("")
    st.subheader("Estimate Your Car's Value")
    st.write("Enter the vehicle details in the sidebar to generate an "
             "estimated market price.")

    if predict_clicked:
        # ---- Input validation (friendly messages, details kept) ----
        valid = True
        if present_price <= 0:
            st.error("Present Price must be greater than 0. "
                     "Please check the vehicle details and try again.")
            valid = False
        if driven_kms < 0 or car_age < 0 or owner < 0:
            st.error("Vehicle details cannot be negative. "
                     "Please check the vehicle details and try again.")
            valid = False

        if valid:
            user_data = pd.DataFrame([{
                "Present_Price": float(present_price),
                "Driven_kms": int(driven_kms),
                "Owner": int(owner),
                "Car_Age": int(car_age),
                "Fuel_Type": fuel_type,
                "Selling_type": selling_type,
                "Transmission": transmission,
            }], columns=FEATURE_COLUMNS)

            try:
                with st.spinner("Analyzing vehicle details…"):
                    prediction = float(model.predict(user_data)[0])
                st.session_state["last_prediction"] = {
                    "value": prediction,
                    "inputs": {
                        "Present Price": f"{present_price:g} Lakhs",
                        "Kilometers Driven": f"{int(driven_kms):,} km",
                        "Car Age": f"{int(car_age)} years",
                        "Previous Owners": int(owner),
                        "Fuel Type": fuel_type,
                        "Selling Type": selling_type,
                        "Transmission": transmission,
                    },
                }
            except Exception as exc:
                st.error("Something went wrong while generating the estimate. "
                         "Please check the vehicle details and try again.")
                with st.expander("Technical details"):
                    st.code(repr(exc))

    # ---- Result / empty state ----
    result = st.session_state.get("last_prediction")

    if result:
        pred = result["value"]
        st.write("")
        if pred < 0:
            st.warning(
                "The model produced a value outside the realistic price "
                "range for these inputs. Try entering vehicle details "
                "closer to the dataset's observed range."
            )
            st.markdown(
                f'<div class="cp-result"><div class="label">'
                f'Estimated Market Value</div>'
                f'<div class="price">₹&nbsp;{pred:.2f}&nbsp;<span>Lakh</span></div>'
                f'<div class="sub">Raw model output (shown unmodified)</div></div>',
                unsafe_allow_html=True,
            )
        else:
            best = best_model_row()
            mae = float(best["MAE"]) if best is not None else None
            low, high = (pred - mae, pred + mae) if mae else (None, None)

            st.markdown(
                f"""
                <div class="cp-result">
                  <div class="label">Estimated Market Value</div>
                  <div class="price">₹&nbsp;{pred:.2f}&nbsp;<span>Lakh</span></div>
                  <div class="sub">Estimated resale value based on the
                  information provided · {to_lakhs_rupees(pred)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.write("")
            k1, k2, k3 = st.columns(3)
            with k1:
                st.markdown(
                    '<div class="cp-kpi accent"><div class="k-label">'
                    'Estimated Price</div>'
                    f'<div class="k-value">₹{pred:.2f}L</div>'
                    f'<div class="k-note">{to_lakhs_rupees(pred)}</div></div>',
                    unsafe_allow_html=True)
            with k2:
                st.markdown(
                    '<div class="cp-kpi"><div class="k-label">Vehicle Age</div>'
                    f'<div class="k-value">{int(car_age)} Years</div>'
                    f'<div class="k-note">as entered</div></div>',
                    unsafe_allow_html=True)
            with k3:
                st.markdown(
                    '<div class="cp-kpi"><div class="k-label">Mileage</div>'
                    f'<div class="k-value">{int(driven_kms):,} km</div>'
                    f'<div class="k-note">as entered</div></div>',
                    unsafe_allow_html=True)

            if low is not None:
                st.write("")
                st.markdown(
                    f'<div class="cp-note">Estimated range: '
                    f'<b style="color:#f2f4f8">₹{max(low, 0):.1f}L – ₹{high:.1f}L</b>'
                    f' &nbsp;·&nbsp; based on the model’s typical test-set '
                    f'error (MAE ±{mae:.2f} Lakh). Actual prices vary with '
                    f'condition, location and demand.</div>',
                    unsafe_allow_html=True,
                )

        # ---- Vehicle summary ----
        st.write("")
        st.markdown('<div class="cp-card-title">Vehicle Summary</div>',
                    unsafe_allow_html=True)
        summary = result["inputs"]
        s1, s2, s3, s4 = st.columns(4)
        cells = list(summary.items())
        for idx, (label, value) in enumerate(cells):
            with [s1, s2, s3, s4][idx % 4]:
                st.markdown(
                    f'<div class="cp-kpi"><div class="k-label">{label}</div>'
                    f'<div class="k-value" style="font-size:1.05rem">{value}</div>'
                    f'</div>',
                    unsafe_allow_html=True)

        st.write("")
        st.markdown(
            '<div class="cp-note">Note: This is an estimated value generated '
            'by a machine-learning model. Actual market prices may vary based '
            'on condition, location, demand, and other factors.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.write("")
        st.markdown(
            """
            <div class="cp-empty">
              <div class="big">Ready to find your car's estimated value?</div>
              <div class="small">Enter your vehicle details in the sidebar and
              click “Estimate Price”.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_footer()

# ============================================================
# PAGE: MARKET INSIGHTS
# ============================================================
elif page == "Market Insights":
    st.markdown('<div class="cp-eyebrow">Market Insights</div>',
                unsafe_allow_html=True)
    st.subheader("Explore the Market")
    st.write("What 299 real resale records say about used-car pricing.")

    tab_dist, tab_rel, tab_cat, tab_model = st.tabs(
        ["Price Distribution", "Feature Relationships",
         "Category Analysis", "Model Insights"])

    with tab_dist:
        show_chart("01_price_distribution.png",
                   "Distribution of Selling Price — most vehicles are "
                   "budget cars; a few expensive ones pull the average up "
                   "(right-skewed).")

    with tab_rel:
        c1, c2 = st.columns(2)
        with c1:
            show_chart("02_present_vs_selling_price.png",
                       "Present Price vs Selling Price — the strongest "
                       "relationship (correlation ≈ +0.88).")
            show_chart("04_km_vs_selling_price.png",
                       "Kilometers Driven vs Selling Price — a weak "
                       "relationship in this dataset.")
        with c2:
            show_chart("03_car_age_vs_selling_price.png",
                       "Car Age vs Selling Price — depreciation: older "
                       "cars sell for less.")
            show_chart("08_correlation_heatmap.png",
                       "Correlation heatmap of the numeric features.")

    with tab_cat:
        c1, c2 = st.columns(2)
        with c1:
            show_chart("05_fuel_type_price.png", "Selling Price by Fuel Type")
            show_chart("07_selling_type_price.png",
                       "Dealer vs Individual sales")
            show_chart("09_fuel_average_price.png",
                       "Average price by Fuel Type")
        with c2:
            show_chart("06_transmission_price.png",
                       "Selling Price by Transmission")
            show_chart("10_transmission_average_price.png",
                       "Average price by Transmission")

    with tab_model:
        c1, c2 = st.columns(2)
        with c1:
            show_chart("12_feature_importance.png",
                       "Feature importance of the best model "
                       "(standardized coefficients).")
            show_chart("13_prediction_error_distribution.png",
                       "Prediction error distribution on the test set.")
        with c2:
            show_chart("14_actual_vs_predicted_best_model.png",
                       "Actual vs Predicted — points near the red line "
                       "are accurate predictions.")
            show_chart("11_actual_vs_predicted.png",
                       "Actual vs Predicted (Step 5 run).")

    render_footer()

# ============================================================
# PAGE: MODEL PERFORMANCE
# ============================================================
elif page == "Model Performance":
    st.markdown('<div class="cp-eyebrow">Model Performance</div>',
                unsafe_allow_html=True)
    st.subheader("How Well Does the Model Predict?")
    st.write("Measured on held-out test data the model never saw during "
             "training (60 vehicles).")

    best = best_model_row()
    if best is None:
        st.warning("model_comparison.csv was not found, so the verified "
                   "performance numbers cannot be displayed. Run "
                   "model_training.py to generate it.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                '<div class="cp-kpi accent"><div class="k-label">Model</div>'
                f'<div class="k-value" style="font-size:1.15rem">'
                f'{best["Model"]}</div>'
                '<div class="k-note">best of 3 tested</div></div>',
                unsafe_allow_html=True)
        with c2:
            st.markdown(
                '<div class="cp-kpi"><div class="k-label">R²</div>'
                f'<div class="k-value">{best["R2"]:.4f}</div>'
                '<div class="k-note">variation in prices explained</div></div>',
                unsafe_allow_html=True)
        with c3:
            st.markdown(
                '<div class="cp-kpi"><div class="k-label">MAE</div>'
                f'<div class="k-value">{best["MAE"]:.4f}</div>'
                '<div class="k-note">Lakhs · typical absolute error</div></div>',
                unsafe_allow_html=True)
        with c4:
            st.markdown(
                '<div class="cp-kpi"><div class="k-label">RMSE</div>'
                f'<div class="k-value">{best["RMSE"]:.4f}</div>'
                '<div class="k-note">Lakhs · weighs large errors more</div></div>',
                unsafe_allow_html=True)

        st.write("")
        st.info(
            f"R² of {best['R2']:.4f} means the model explains approximately "
            f"{best['R2'] * 100:.2f}% of the variation in selling prices in "
            f"the test set. MAE of {best['MAE']:.4f} Lakhs means the average "
            f"absolute prediction error was approximately ₹{best['MAE']:.2f} "
            f"lakh. RMSE is {best['RMSE']:.4f} Lakhs and gives greater weight "
            f"to larger prediction errors."
        )

        st.write("")
        st.markdown("#### All models tested")
        if comparison is not None:
            def highlight_best(row):
                color = ("background-color: rgba(229, 72, 77, 0.14)"
                         if row["R2"] == comparison["R2"].max() else "")
                return [color] * len(row)

            st.dataframe(comparison.style.apply(highlight_best, axis=1),
                         use_container_width=True)
            st.caption("Metrics are loaded live from model_comparison.csv "
                       "— never hard-coded.")

    render_footer()

# ============================================================
# PAGE: ABOUT
# ============================================================
else:
    st.markdown('<div class="cp-eyebrow">About</div>',
                unsafe_allow_html=True)
    st.subheader("How It Works")

    steps = [
        ("01", "Enter Details",
         "Tell us about the vehicle — price, age, mileage, fuel, "
         "transmission and seller type."),
        ("02", "Analyze",
         "The machine-learning model processes the vehicle "
         "characteristics against historical sales data."),
        ("03", "Estimate",
         "Receive an estimated market price with the model's typical "
         "error range."),
        ("04", "Explore",
         "Review market visualizations and model performance to "
         "understand the estimate."),
    ]
    for num, title, desc in steps:
        st.markdown(
            f'<div class="cp-step"><div class="num">{num}</div><div>'
            f'<div class="t">{title}</div>'
            f'<div class="d">{desc}</div></div></div>',
            unsafe_allow_html=True)

    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="cp-card"><div class="cp-card-title">The Data</div>'
                    '<div class="cp-card-sub">299 historical resale records '
                    '(after removing 2 exact duplicates) with 9 attributes — '
                    'no missing values. Prices are in Indian Lakhs. '
                    'A Car_Age feature was engineered from the purchase year.'
                    '</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="cp-card"><div class="cp-card-title">The Model</div>'
                    '<div class="cp-card-sub">Three regressors were compared '
                    '(Linear Regression, Random Forest, Gradient Boosting). '
                    'Linear Regression performed best on unseen test data. '
                    'The full pipeline — preprocessing included — is saved as '
                    'car_price_model.pkl.</div></div>', unsafe_allow_html=True)

    st.write("")
    st.markdown("#### Keep in mind")
    st.markdown(
        "- Estimates are statistical — they are **not guaranteed market "
        "prices**.\n"
        "- The dataset is small (299 rows) and contains very few CNG "
        "vehicles and automatics.\n"
        "- The dataset has no horsepower, brand-goodwill or advertising "
        "information, so none is used.\n"
        "- Findings are associations within this dataset; correlation does "
        "not prove causation."
    )

    render_footer()
