# Telco Customer Churn Predictor

## Run the app

1. Run the notebook cells through the final model-save cell to create `xgb_telco_churn_bundle.joblib` in this folder.
2. Install the app dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start Streamlit from this folder:

   ```bash
   streamlit run app.py
   ```

The app loads the tuned XGBoost model and its saved preprocessing objects, collects customer details, and reports the predicted churn status and probability.
pip install -r requirements.txt
to run project:
streamlit run app.py