# ============================================================
# CAR PRICE PREDICTOR - Streamlit Web Application (UI v3)
# ------------------------------------------------------------
# Design pass: warm neutral automotive look & feel with a
# 4-step guided estimate flow (Back moves exactly one step).
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

# Highest Present_Price observed in the training data (Lakhs).
# Used only for an honest "you are extrapolating" note - the
# model itself is untouched and will still receive any input.
TRAINING_MAX_PRESENT_PRICE = 92.6

WIZARD_STEPS = ["Vehicle Basics", "Usage & History", "Specifications",
                "Review & Estimate"]

# ------------------------------------------------------------
# DESIGN SYSTEM (single style block - warm neutral automotive)
# ------------------------------------------------------------
CSS = """
<style>
/* ---------- Design tokens ---------- */
:root {
  --bg: #f6f5f2;            /* warm off-white        */
  --surface: #ffffff;       /* card surface          */
  --surface-2: #fbfaf8;     /* raised/soft surface   */
  --border: #e6e3dc;        /* soft gray border      */
  --text: #23272f;          /* charcoal (not black)  */
  --text-dim: #5b6472;      /* medium neutral gray   */
  --accent: #b23a2f;        /* muted brick red       */
  --accent-strong: #972e24; /* hover state           */
  --accent-soft: rgba(178, 58, 47, 0.08);
  --ok: #2e7d5b;            /* muted green           */
}

/* ---------- Base typography & spacing ---------- */
html, body, [class*="css"] {
  font-family: "Segoe UI", system-ui, -apple-system, "Helvetica Neue", sans-serif;
}
body { background: var(--bg); color: var(--text); }
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1200px; }

/* Streamlit's real containers must inherit the warm light theme too,
   otherwise the app shows its default dark background behind cards. */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] section.main,
[data-testid="stHeader"] {
  background: var(--bg) !important;
  color: var(--text);
}
[data-testid="stHeader"] { border: none; box-shadow: none; }

/* Button labels: Streamlit wraps the label in a <p>, which would
   otherwise inherit the dim paragraph color and look faded. */
.stButton > button p { color: inherit; margin: 0; }

/* Explicit heading colors so headings can never inherit an
   invisible color on any page, step, or rerun state. */
h1, h2, h3, h4, h5, h6,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4 {
  color: var(--text) !important;
  letter-spacing: -0.01em;
}
p, li { color: var(--text-dim); }
b, strong { color: var(--text); }
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
.cp-status .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--ok); }
.cp-status b { color: var(--text); font-weight: 600; }

/* ---------- Cards ---------- */
.cp-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 14px; padding: 1.25rem 1.35rem; margin-bottom: 1rem;
}
.cp-card-title { color: var(--text); font-weight: 700; font-size: 0.95rem; margin-bottom: 0.15rem; }
.cp-card-sub { color: var(--text-dim); font-size: 0.85rem; }

/* ---------- Result card ---------- */
.cp-result {
  background: linear-gradient(180deg, var(--surface), var(--surface-2));
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
  font-weight: 800; font-size: 0.95rem; border: 1px solid rgba(178, 58, 47, 0.30);
}
.cp-step .t { color: var(--text); font-weight: 700; font-size: 0.98rem; }
.cp-step .d { color: var(--text-dim); font-size: 0.88rem; margin-top: 0.1rem; line-height: 1.45; }

/* ---------- Wizard step indicator ---------- */
.cp-wiz-steps { display: flex; gap: 0.5rem; flex-wrap: wrap; margin: 0.4rem 0 1.1rem 0; }
.cp-wchip {
  display: inline-flex; align-items: center; gap: 0.45rem;
  padding: 0.4rem 0.85rem; border: 1px solid var(--border);
  border-radius: 999px; font-size: 0.8rem; color: var(--text-dim);
  background: var(--surface-2);
}
.cp-wchip .n {
  width: 20px; height: 20px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-weight: 700; font-size: 0.72rem; background: var(--border); color: var(--text);
}
.cp-wchip.active { border-color: var(--accent); color: var(--text); background: var(--accent-soft); }
.cp-wchip.active .n { background: var(--accent); color: #ffffff; }
.cp-wchip.done .n { background: var(--ok); color: #ffffff; }
@media (max-width: 640px) { .cp-wchip { padding: 0.32rem 0.6rem; font-size: 0.72rem; } }

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
  background: var(--surface-2); border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .cp-brand {
  font-weight: 800; letter-spacing: 0.22em; font-size: 0.78rem;
  color: var(--text); text-transform: uppercase;
}
section[data-testid="stSidebar"] .cp-brand span { color: var(--accent); }
section[data-testid="stSidebar"] hr { margin: 0.4rem 0 0.9rem 0; }
section[data-testid="stSidebar"] .block-container { padding-top: 1.4rem; }

/* Nav radio -> clean vertical pills */
section[data-testid="stSidebar"] div[role="radiogroup"] { gap: 0.25rem; }
section[data-testid="stSidebar"] div[role="radiogroup"] label {
  background: transparent; border: 1px solid transparent;
  border-radius: 10px; padding: 0.45rem 0.7rem; margin: 0;
  color: var(--text-dim);
  transition: background 0.15s ease, border-color 0.15s ease;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover { background: var(--surface); }
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
  background: var(--surface); border-color: var(--border); color: var(--text);
}

/* ---------- Buttons (high contrast, non-neon) ---------- */
.stButton > button {
  width: 100%; border-radius: 12px; font-weight: 700; font-size: 1rem;
  padding: 0.7rem 1rem; transition: transform 0.12s ease, background 0.12s ease,
  border-color 0.12s ease;
}
.stButton > button[kind="primary"] {
  background: var(--accent); color: #ffffff !important; border: 1px solid var(--accent);
  box-shadow: 0 4px 14px rgba(178, 58, 47, 0.22);
}
.stButton > button[kind="primary"]:hover {
  background: var(--accent-strong); border-color: var(--accent-strong);
  color: #ffffff !important; transform: translateY(-1px);
}
.stButton > button[kind="primary"]:active { transform: translateY(0); }
.stButton > button[kind="primary"]:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.stButton > button[kind="secondary"], .stButton > button:not([kind="primary"]) {
  background: var(--surface); color: var(--text) !important; border: 1px solid var(--border);
}
.stButton > button[kind="secondary"]:hover, .stButton > button:not([kind="primary"]):hover {
  border-color: var(--text-dim); background: var(--surface-2); color: var(--text) !important;
}

/* ---------- Misc polish ---------- */
.cp-empty {
  border: 1px dashed var(--border); border-radius: 16px;
  padding: 3rem 1.5rem; text-align: center; background: var(--surface-2);
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


def format_price(price_in_lakhs):
    """Human price: ₹6.39 Lakh / ₹18.50 Lakh / ₹1.25 Crore."""
    if price_in_lakhs >= 100:
        return f"₹{price_in_lakhs / 100:.2f} Crore"
    return f"₹{price_in_lakhs:.2f} Lakh"


def price_input_caption(price_in_lakhs):
    """Readable caption under the price slider, in Indian units."""
    return (f"You selected {format_price(price_in_lakhs)} "
            f"({to_lakhs_rupees(price_in_lakhs)}) · "
            f"you can go up to {format_price(300.0)}")


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
# SIDEBAR: brand + navigation (vehicle inputs live in the wizard)
# ------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="cp-brand">CAR PRICE <span>PREDICTOR</span></div>',
                unsafe_allow_html=True)
    st.divider()

    page = st.radio("Navigation", NAV_PAGES, label_visibility="collapsed")

    st.divider()
    if page == "Predict Price":
        st.caption("Answer a few questions about the car and we'll "
                   "estimate its value.")
    else:
        st.caption("Open “Predict Price” to estimate a car's value.")

# ------------------------------------------------------------
# SHARED: hero, footer
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
          <div><b>Car Price Predictor</b><br>
          Machine Learning · Vehicle Analytics</div>
          <div>Built with Python, Streamlit &amp; Machine Learning</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------
# WIZARD STATE (survives reruns; can never leave 0..3)
# ------------------------------------------------------------
st.session_state.setdefault("wizard_step", 0)
st.session_state["wizard_step"] = max(0, min(int(st.session_state["wizard_step"]), 3))


def _persist(key):
    """on_change callback: mirror a widget's value into a 'saved_' key.

    Streamlit deletes a widget's session state whenever that widget is not
    rendered on a run (which happens on every other wizard step). Without
    this mirror, going Back or clicking Estimate after navigating would
    silently reset the form to defaults.
    """
    def _cb():
        st.session_state[f"saved_{key}"] = st.session_state[key]
    return _cb


def saved_value(key, default):
    """Latest known value for a wizard control (survives step changes)."""
    return st.session_state.get(f"saved_{key}", default)


def render_wizard_steps(current):
    chips = []
    for i, name in enumerate(WIZARD_STEPS):
        cls = "cp-wchip"
        if i == current:
            cls += " active"
        elif i < current:
            cls += " done"
        chips.append(f'<span class="{cls}"><span class="n">{i + 1}</span>{name}</span>')
    st.markdown('<div class="cp-wiz-steps">' + "".join(chips) + "</div>",
                unsafe_allow_html=True)


# ============================================================
# PAGE: PREDICT PRICE
# ============================================================
if page == "Predict Price":
    render_hero()
    st.write("")

    step = st.session_state["wizard_step"]
    render_wizard_steps(step)

    # --------------------------------------------------------
    # STEP 1 — Vehicle Basics
    # --------------------------------------------------------
    if step == 0:
        st.subheader("Tell us about your car")
        st.write("What is the current showroom price of a **new** version "
                 "of this car?")
        pp_kwargs = dict(step=0.5, key="pp", on_change=_persist("pp"))
        if "pp" not in st.session_state:            # restore after Back
            pp_kwargs["value"] = float(saved_value("pp", 6.0))
        present_price = st.slider(
            "Present Price (₹ Lakhs)", min_value=0.5, max_value=300.0,
            help="1 Lakh = ₹100,000 and 1 Crore = ₹10,000,000. "
                 "You can select up to ₹3 Crore.", **pp_kwargs)
        st.caption(price_input_caption(present_price))
        if present_price > TRAINING_MAX_PRESENT_PRICE:
            st.caption("Heads-up: the most expensive car in the training "
                       f"data was {format_price(TRAINING_MAX_PRESENT_PRICE)} — "
                       "above that, the estimate is an extrapolation.")

    # --------------------------------------------------------
    # STEP 2 — Usage & History
    # --------------------------------------------------------
    elif step == 1:
        st.subheader("How has it been used?")
        st.write("A few quick details about the car's history.")
        kms_kwargs = dict(step=5000, key="kms", on_change=_persist("kms"))
        if "kms" not in st.session_state:
            kms_kwargs["value"] = int(saved_value("kms", 30000))
        driven_kms = st.slider(
            "Kilometers Driven", min_value=0, max_value=300000,
            help="How far has it been driven in total?", **kms_kwargs)
        owner_kwargs = dict(key="owner", on_change=_persist("owner"))
        if "owner" not in st.session_state:
            owner_kwargs["index"] = [0, 1, 2, 3].index(int(saved_value("owner", 0)))
        owner = st.selectbox(
            "Previous Owners", options=[0, 1, 2, 3],
            help="How many people owned the car before you? The training "
                 "data mainly observed 0, 1 and 3; 2 is offered for "
                 "completeness but is an extrapolation.", **owner_kwargs)
        age_kwargs = dict(step=1, key="age", on_change=_persist("age"))
        if "age" not in st.session_state:
            age_kwargs["value"] = int(saved_value("age", 5))
        car_age = st.slider(
            "Car Age (Years)", min_value=0, max_value=30,
            help="How old is the car today?", **age_kwargs)

    # --------------------------------------------------------
    # STEP 3 — Specifications
    # --------------------------------------------------------
    elif step == 2:
        st.subheader("Which fuel does it use?")
        st.write("Fuel, transmission and how the car is being sold.")
        fuel_kwargs = dict(key="fuel_type", on_change=_persist("fuel_type"))
        if "fuel_type" not in st.session_state:
            fuel_kwargs["index"] = ["Petrol", "Diesel", "CNG"].index(
                saved_value("fuel_type", "Petrol"))
        fuel_type = st.selectbox("Fuel Type", options=["Petrol", "Diesel", "CNG"],
                                 **fuel_kwargs)
        sell_kwargs = dict(key="selling_type", on_change=_persist("selling_type"))
        if "selling_type" not in st.session_state:
            sell_kwargs["index"] = ["Dealer", "Individual"].index(
                saved_value("selling_type", "Dealer"))
        selling_type = st.selectbox(
            "Selling Type", options=["Dealer", "Individual"],
            help="Are you selling through a dealer or as an individual?",
            **sell_kwargs)
        trans_kwargs = dict(key="transmission", on_change=_persist("transmission"))
        if "transmission" not in st.session_state:
            trans_kwargs["index"] = ["Manual", "Automatic"].index(
                saved_value("transmission", "Manual"))
        transmission = st.selectbox("Transmission", options=["Manual", "Automatic"],
                                    **trans_kwargs)

    # --------------------------------------------------------
    # STEP 4 — Review & Estimate
    # --------------------------------------------------------
    else:
        st.subheader("Ready to estimate")
        st.write("Here's what you've told us about the car. Click "
                 "**Estimate Price** when you're happy with it.")

        present_price = float(saved_value("pp", 6.0))
        driven_kms = int(saved_value("kms", 30000))
        owner = int(saved_value("owner", 0))
        car_age = int(saved_value("age", 5))
        fuel_type = saved_value("fuel_type", "Petrol")
        selling_type = saved_value("selling_type", "Dealer")
        transmission = saved_value("transmission", "Manual")

        # ---- Review of entered values (read-only) ----
        review = [
            ("Present Price", format_price(present_price)),
            ("Kilometers Driven", f"{int(driven_kms):,} km"),
            ("Car Age", f"{int(car_age)} years"),
            ("Previous Owners", int(owner)),
            ("Fuel Type", fuel_type),
            ("Selling Type", selling_type),
            ("Transmission", transmission),
        ]
        r1, r2, r3, r4 = st.columns(4)
        containers = [r1, r2, r3, r4]
        for idx, (label, value) in enumerate(review):
            with containers[idx % 4]:
                st.markdown(
                    f'<div class="cp-kpi"><div class="k-label">{label}</div>'
                    f'<div class="k-value" style="font-size:1.05rem">{value}</div>'
                    f'</div>', unsafe_allow_html=True)

        # ---- Estimate button (high-contrast primary CTA) ----
        st.write("")
        estimate_clicked = st.button("Estimate Price", type="primary",
                                     use_container_width=True)

        if estimate_clicked:
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
                            "Present Price": format_price(present_price),
                            "Kilometers Driven": f"{int(driven_kms):,} km",
                            "Car Age": f"{int(car_age)} years",
                            "Previous Owners": int(owner),
                            "Fuel Type": fuel_type,
                            "Selling Type": selling_type,
                            "Transmission": transmission,
                        },
                    }
                except Exception as exc:
                    st.error("Something went wrong while generating the "
                             "estimate. Please check the vehicle details "
                             "and try again.")
                    with st.expander("Technical details"):
                        st.code(repr(exc))

        # ---- Result / empty state (persists across reruns) ----
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
                    f'<div class="price">{format_price(pred)}</div>'
                    f'<div class="sub">Raw model output (shown unmodified)'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )
            else:
                best = best_model_row()
                mae = float(best["MAE"]) if best is not None else None
                low, high = ((pred - mae, pred + mae) if mae else (None, None))

                st.markdown(
                    f"""
                    <div class="cp-result">
                      <div class="label">Your Estimated Car Value</div>
                      <div class="price">{format_price(pred)}</div>
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
                        f'<div class="k-value">{format_price(pred)}</div>'
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
                        f'<b>{format_price(max(low, 0))} – {format_price(high)}</b>'
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
                  <div class="small">Click “Estimate Price” above and the result
                  will appear here.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # --------------------------------------------------------
    # Wizard navigation: Back = exactly one step; Back hidden on step 1
    # --------------------------------------------------------
    st.write("")
    nav1, nav2 = st.columns(2)
    with nav1:
        if step > 0:
            if st.button("← Back", key="wiz_back", use_container_width=True):
                st.session_state["wizard_step"] = step - 1     # exactly one
                st.rerun()
    with nav2:
        if step < 3:
            if st.button("Continue →", type="primary", key="wiz_next",
                         use_container_width=True):
                st.session_state["wizard_step"] = step + 1     # exactly one
                st.rerun()

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
                color = ("background-color: rgba(178, 58, 47, 0.10)"
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
