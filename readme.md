# 📊 Demand Forecasting System (Machine Learning)

An end-to-end **Demand Forecasting application** built with **XGBoost** and deployed using **Gradio**.  
The system predicts **future product demand (in units)** based on **business-friendly inputs**, enabling better inventory and sales planning.

## 🚀 Project Highlights

- ✅ Predicts **future demand in units**
- ✅ Uses **XGBoost** for high accuracy on tabular data
- ✅ Simple, business-friendly UI (no technical inputs)
- ✅ Robust error handling (no silent failures)
- ✅ Clear explanations embedded in the app
- ✅ Production-style design

## 📌 Problem Statement

Retail and supply-chain teams need to estimate **how many units of a product will be sold** under different pricing and market conditions.

This project solves that problem by:
- Learning from historical sales data
- Modeling pricing and regional demand behavior
- Providing an interactive interface for demand simulation

## 🎯 Objective

- Forecast **expected units sold**
- Understand the impact of **discounts, category, and region**
- Provide a **decision-support tool** for business users

## 🧠 What Does the Model Predict?

### **Demand Forecast (Units)**

> The **expected number of units customers are likely to purchase** in the next time period under the selected conditions.

This value helps businesses:
- Plan inventory levels
- Avoid stock-outs or over-stocking
- Evaluate discount strategies

## 🧾 User Inputs

The app intentionally exposes **only high-impact, easy-to-understand inputs**.

| Input | Description |
|------|------------|
| **Discount (%)** | Percentage price reduction applied |
| **Category** | Product category (e.g., Grocery, Electronics) |
| **Region** | Sales region (e.g., North, South) |

> All technical features (lags, rolling averages, time features) are handled internally to keep the interface simple and stable.

## ⚙️ Technical Approach

### 🔹 Model
- **XGBoost Regressor**

### 🔹 Why XGBoost?
- Strong performance on structured/tabular data
- Handles non-linear relationships
- Widely used in real-world production systems
- Robust to feature interactions

## 🛠 Feature Engineering (Internal)

The model uses engineered features that are **not exposed to the user**:

- Lag features (`lag_1`, `lag_7`, `lag_14`, `lag_30`)
- Rolling averages (`rolling_mean_7`, `rolling_mean_30`)
- Date-based features (year, month, weekday)
- Encoded categorical variables

This ensures:
- Realistic predictions
- No data leakage
- Stable inference

## 🖥️ Application Features

- Interactive web interface using **Gradio**
- Dropdowns populated from training data (prevents invalid inputs)
- Embedded explanations for:
  - Input meaning
  - Prediction interpretation
- Defensive error handling with visible error messages
- Clean, professional UI suitable for demos

## 📊 Example Output

Demand Forecast (Units): 145

What this means:

* Expected number of units likely to be sold
* Based on selected discount, category, and region
* Learned from historical sales patterns

## 📁 Project Structure

demand-forecasting-system/
│
├── app.py                     # Gradio application
├── xgb_demand_model.pkl       # Trained XGBoost model
├── model_features.pkl         # Feature schema used during training
├── label_encoders.pkl         # Encoders for categorical variables
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation


## ▶️ Getting Started

### 1️⃣ Clone the repository

git clone <repository-url>
cd demand-forecasting-system

### 2️⃣ (Optional) Create a virtual environment

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

### 3️⃣ Install dependencies

pip install -r requirements.txt

### 4️⃣ Run the application

python app.py

### 5️⃣ Open the app

Gradio will display a local URL such as:

http://127.0.0.1:7860

## 🧠 Key Learnings

* Demand forecasting using machine learning
* Time-series feature engineering for tabular ML
* Handling categorical variables safely in production
* Avoiding data leakage
* Building business-friendly ML applications
* Deploying ML models with Gradio

## 🎤 Project Summary

“I built a demand forecasting system using XGBoost that predicts expected units sold based on discount, category, and region. The model uses internal time-series features and is deployed as a Gradio app with clear explanations and robust error handling.”

## 🚀 Future Enhancements

* Inventory order recommendations
* Confidence intervals for predictions
* Discount vs demand visualization
* Multi-store or multi-product selection
* Deployment to Hugging Face Spaces

## 📜 License

This project is intended for **educational and portfolio purposes**.

You are free to modify and extend it.
