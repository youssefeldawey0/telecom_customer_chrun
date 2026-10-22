from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "xgb_telco_churn_bundle.joblib"

st.set_page_config(page_title="Telco Churn Predictor", page_icon="📡", layout="centered")
st.title("📡 Telco Customer Churn Predictor")
st.write("Enter a customer's service and account details to estimate their churn risk.")


@st.cache_resource
def load_bundle(path: Path):
    return joblib.load(path)


if not MODEL_PATH.exists():
    st.error(
        "Saved model bundle not found. Run the final save cell in `tele cust project.ipynb` "
        "so `xgb_telco_churn_bundle.joblib` is created in this folder."
    )
    st.stop()

bundle = load_bundle(MODEL_PATH)
model = bundle["model"]
encoder = bundle["encoder"]
scaler = bundle["scaler"]
median_total_charges = bundle["median_total_charges"]
binary_maps = bundle["binary_maps"]
nominal_cols = bundle["nominal_cols"]
robust_cols = bundle["robust_cols"]
feature_columns = bundle["feature_columns"]


with st.form("customer_details"):
    st.subheader("Customer details")
    left, right = st.columns(2)

    with left:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior citizen", [0, 1], format_func=lambda v: "Yes" if v else "No")
        partner = st.selectbox("Has a partner?", ["No", "Yes"])
        dependents = st.selectbox("Has dependents?", ["No", "Yes"])
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12, step=1)
        phone_service = st.selectbox("Phone service", ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])

    with right:
        online_security = st.selectbox("Online security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming movies", ["No", "Yes", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless billing", ["No", "Yes"])
        payment_method = st.selectbox(
            "Payment method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )

    monthly_charges = st.number_input("Monthly charges", min_value=0.0, max_value=2000.0, value=70.0, step=1.0)
    unknown_total = st.checkbox("I don't know the customer's total charges")
    total_charges = None
    if not unknown_total:
        total_charges = st.number_input(
            "Total charges", min_value=0.0, max_value=100000.0, value=840.0, step=10.0
        )

    predict_clicked = st.form_submit_button(
        "Predict churn risk", type="primary", use_container_width=True
    )


def prepare_features(values: dict) -> pd.DataFrame:
    row = pd.DataFrame([values])
    for column, mapping in binary_maps.items():
        row[column] = row[column].map(mapping)

    encoded = encoder.transform(row[nominal_cols])
    prepared = pd.concat([row.drop(columns=nominal_cols), encoded], axis=1)

    # Match the notebook's preprocessing order: scale first, then fill missing TotalCharges.
    prepared.loc[:, robust_cols] = scaler.transform(prepared[robust_cols])
    prepared["TotalCharges"] = prepared["TotalCharges"].fillna(median_total_charges)
    return prepared.reindex(columns=feature_columns)


if predict_clicked:
    customer = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }
    features = prepare_features(customer)
    prediction = int(model.predict(features)[0])

    churn_probability = None
    if hasattr(model, "predict_proba"):
        class_index = list(model.classes_).index(1)
        churn_probability = float(model.predict_proba(features)[0, class_index])

    if prediction == 1:
        st.error("⚠️ This customer is at risk of leaving.")
    else:
        st.success("✅ This customer is likely to stay.")

    if churn_probability is not None:
        st.metric("Estimated churn probability", f"{churn_probability:.1%}")
