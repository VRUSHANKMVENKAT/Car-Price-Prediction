# ============================================================
# app_test.py - HEADLESS sanity check of the trained model.
# Verifies the saved Pipeline accepts raw feature values in the
# exact format the web app will send. Does NOT modify anything.
# ============================================================
import joblib
import pandas as pd

model = joblib.load("car_price_model.pkl")

feature_columns = ["Present_Price", "Driven_kms", "Owner",
                   "Car_Age", "Fuel_Type", "Selling_type", "Transmission"]

def to_lakhs_rupees(price_in_lakhs, symbol="Rs "):
    """Convert Lakhs to an Indian-formatted rupee string.
    (Default symbol avoids Windows-console encoding issues; the
    Streamlit app passes the real rupee symbol for browser display.)"""
    rupees = int(round(price_in_lakhs * 100000))
    if rupees >= 10000000:
        s, rest = divmod(rupees, 10000000)
        text = f"{s},{rest:07d}"
    else:
        text = str(rupees)
    if len(text) > 3:
        head, tail = text[:-3], text[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        text = ",".join(groups) + "," + tail
    return f"{symbol}{text}"

user_input = {
    "Present_Price": 6.0,
    "Driven_kms": 30000,
    "Owner": 0,
    "Car_Age": 5,
    "Fuel_Type": "Petrol",
    "Selling_type": "Dealer",
    "Transmission": "Manual",
}

input_df = pd.DataFrame([user_input], columns=feature_columns)
print("Input DataFrame sent to model.predict():")
print(input_df.to_string(index=False))

prediction = model.predict(input_df)[0]
print(f"\nRaw prediction: {prediction:.4f} Lakhs")
print(f"Display value:  Rs {prediction:.2f} Lakhs")
print(f"Indian rupees:  {to_lakhs_rupees(prediction)}")

# A second spot check: an expensive car should predict higher.
expensive = dict(user_input, Present_Price=25.0, Fuel_Type="Diesel",
                 Transmission="Automatic")
pred2 = model.predict(pd.DataFrame([expensive], columns=feature_columns))[0]
print(f"\nSpot check (25 L diesel automatic, 5 yrs): {pred2:.2f} Lakhs "
      f"-> higher than petrol prediction: {pred2 > prediction}")
