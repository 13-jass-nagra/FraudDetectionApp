from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load model and scaler
model = joblib.load('model/xgb_model.pkl')
scaler = joblib.load('model/scaler.pkl')

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    step = float(request.form['step'])
    amount = float(request.form['amount'])
    oldbalanceOrg = float(request.form['oldbalanceOrg'])
    newbalanceOrig = float(request.form['newbalanceOrig'])
    oldbalanceDest = float(request.form['oldbalanceDest'])
    newbalanceDest = float(request.form['newbalanceDest'])
    isFlaggedFraud = int(request.form['isFlaggedFraud'])
    type_CASH_IN = int(request.form['type_CASH_IN'])
    type_CASH_OUT = int(request.form['type_CASH_OUT'])
    type_DEBIT = int(request.form['type_DEBIT'])
    type_PAYMENT = int(request.form['type_PAYMENT'])
    type_TRANSFER = int(request.form['type_TRANSFER'])

    # Create a dataframe for prediction
    input_data = pd.DataFrame({
        'step': [step],
        'amount': [amount],
        'oldbalanceOrg': [oldbalanceOrg],
        'newbalanceOrig': [newbalanceOrig],
        'oldbalanceDest': [oldbalanceDest],
        'newbalanceDest': [newbalanceDest],
        'isFlaggedFraud': [isFlaggedFraud],
        'type_CASH_IN': [type_CASH_IN],
        'type_CASH_OUT': [type_CASH_OUT],
        'type_DEBIT': [type_DEBIT],
        'type_PAYMENT': [type_PAYMENT],
        'type_TRANSFER': [type_TRANSFER]
    })

    # Scale the input data using the scaler
    input_data_scaled = scaler.transform(input_data)

    # Predict using the model
    prob = model.predict_proba(input_data_scaled)[:, 1]  # Get the probability for fraud
    prediction = 'Fraud' if prob[0] > 0.15 else 'Not Fraud'

    return render_template('index.html', prediction_text=f'Prediction: {prediction}', prob_text=f'Fraud Probability: {prob[0]:.2f}')

if __name__ == "__main__":
    app.run(debug=True)
