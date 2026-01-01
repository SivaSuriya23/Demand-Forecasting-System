import gradio as gr
import pandas as pd
import joblib

# ===============================
# Load trained artifacts
# ===============================
model = joblib.load("xgb_demand_model.pkl")
features = joblib.load("model_features.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# ===============================
# Defaults from training
# ===============================
DEFAULTS = {
    "Store ID": label_encoders["Store ID"].classes_[0],
    "Product ID": label_encoders["Product ID"].classes_[0],
    "Inventory Level": 120,
    "Units Ordered": 100,
    "Price": 250.0,
    "Holiday/Promotion": 0,
    "Competitor Pricing": 250.0,

    "lag_1": 50,
    "lag_7": 50,
    "lag_14": 48,
    "lag_30": 48,
    "rolling_mean_7": 52,
    "rolling_mean_30": 50,

    # Neutralize leakage
    "Demand Forecast": 0
}

REFERENCE_DATE = pd.Timestamp("2022-01-01")

# ===============================
# Feature builder (SAFE)
# ===============================
def build_features(discount, category, region):
    date = REFERENCE_DATE

    data = {
        "Store ID": DEFAULTS["Store ID"],
        "Product ID": DEFAULTS["Product ID"],
        "Category": category,
        "Region": region,
        "Inventory Level": DEFAULTS["Inventory Level"],
        "Units Ordered": DEFAULTS["Units Ordered"],
        "Price": DEFAULTS["Price"],
        "Discount": float(discount),
        "Holiday/Promotion": DEFAULTS["Holiday/Promotion"],
        "Competitor Pricing": DEFAULTS["Competitor Pricing"],

        "year": int(date.year),
        "month": int(date.month),
        "day": int(date.day),
        "dayofweek": int(date.dayofweek),
        "weekofyear": int(date.isocalendar().week),

        "lag_1": DEFAULTS["lag_1"],
        "lag_7": DEFAULTS["lag_7"],
        "lag_14": DEFAULTS["lag_14"],
        "lag_30": DEFAULTS["lag_30"],
        "rolling_mean_7": DEFAULTS["rolling_mean_7"],
        "rolling_mean_30": DEFAULTS["rolling_mean_30"],

        "Demand Forecast": 0
    }

    df = pd.DataFrame([data])

    # Encode categoricals safely
    for col, le in label_encoders.items():
        if col in df.columns:
            value = df[col].iloc[0]
            if value in le.classes_:
                df[col] = le.transform([value])
            else:
                df[col] = le.transform([le.classes_[0]])

    # Ensure ALL expected features exist
    for col in features:
        if col not in df.columns:
            df[col] = 0

    return df[features]

# ===============================
# Prediction function (WITH ERROR DISPLAY)
# ===============================
def predict_all(discount, category, region):
    try:
        df = build_features(discount, category, region)
        prediction = float(model.predict(df)[0])

        return (
            f"📊 Demand Forecast: {int(round(prediction))} units\n\n"
            f"Inputs Used:\n"
            f"- Discount: {discount}%\n"
            f"- Category: {category}\n"
            f"- Region: {region}"
        )

    except Exception as e:
        # THIS WILL SHOW THE REAL ERROR IN UI
        return f"❌ Prediction Error:\n{str(e)}"

# ===============================
# Gradio UI
# ===============================
app = gr.Interface(
    fn=predict_all,
    inputs=[
        gr.Slider(0, 100, step=5, label="Discount (%)"),
        gr.Dropdown(label_encoders["Category"].classes_.tolist(), label="Category"),
        gr.Dropdown(label_encoders["Region"].classes_.tolist(), label="Region")
    ],
    outputs=gr.Textbox(
        label="Prediction Results / Errors",
        lines=8
    ),
    title="📊 Demand Forecasting System",
    description="Predicts demand using discount and market segmentation with ML."
)

app.launch()
