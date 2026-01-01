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

    "Demand Forecast": 0
}

REFERENCE_DATE = pd.Timestamp("2022-01-01")

# ===============================
# Feature builder
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
            df[col] = le.transform([df[col].iloc[0]])

    # Align with training schema
    for col in features:
        if col not in df.columns:
            df[col] = 0

    return df[features]

# ===============================
# Prediction function
# ===============================
def predict_all(discount, category, region):
    try:
        df = build_features(discount, category, region)
        prediction = float(model.predict(df)[0])

        return (
            f"📊 Demand Forecast (Units): {int(round(prediction))}\n\n"
            f"**What this means:**\n"
            f"- This is the expected number of units likely to be sold\n"
            f"- Based on selected discount, category, and region\n"
            f"- Learned from historical sales patterns"
        )

    except Exception as e:
        return f"❌ Error:\n{str(e)}"

# ===============================
# Gradio UI
# ===============================
with gr.Blocks() as app:

    gr.Markdown(
        """
        # 📊 Demand Forecasting System

        This app predicts **future product demand (in units)** using machine learning.

        ---
        ### 🔍 Input Explanations

        - **Discount (%)**  
          Higher discounts generally increase demand by reducing price.

        - **Category**  
          Different product categories behave differently (e.g., groceries vs electronics).

        - **Region**  
          Demand varies by geography due to customer behavior and market size.

        ---
        ### 📈 Prediction Explanation

        **Demand Forecast (Units)** represents the **expected number of units customers are likely to buy** under the selected conditions.
        """
    )

    discount = gr.Slider(0, 100, step=5, label="Discount (%)")
    category = gr.Dropdown(label_encoders["Category"].classes_.tolist(), label="Category")
    region = gr.Dropdown(label_encoders["Region"].classes_.tolist(), label="Region")

    output = gr.Textbox(label="Prediction Results", lines=8)

    submit = gr.Button("Submit")

    submit.click(
        fn=predict_all,
        inputs=[discount, category, region],
        outputs=output
    )

app.launch()
