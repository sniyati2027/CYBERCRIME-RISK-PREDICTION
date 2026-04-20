from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os
import requests
import datetime
import random
import json

app = FastAPI(title="Cybercrime Risk Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE      = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "models")

risk_model    = joblib.load(os.path.join(MODEL_DIR, "risk_model.pkl"))
city_model    = joblib.load(os.path.join(MODEL_DIR, "city_model.pkl"))
scaler        = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
risk_features = joblib.load(os.path.join(MODEL_DIR, "risk_features.pkl"))
city_features = joblib.load(os.path.join(MODEL_DIR, "city_features.pkl"))

FRAUD_TYPES = ['Phishing','UPI Fraud','Credit Card Fraud','Fake Loan App','OTP Scam','Online Shopping Scam']
N8N_WEBHOOK_URL = "http://host.docker.internal:5678/webhook/cybercrime-alert"

FRONTEND_DIR = os.path.join(BASE, "..", "frontend")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# In-memory alert log
alert_log = []

@app.get("/")
def serve_map():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

@app.get("/predict-form")
def serve_form():
    return FileResponse(os.path.join(FRONTEND_DIR, "predict.html"))

@app.get("/lea")
def serve_lea():
    return FileResponse(os.path.join(FRONTEND_DIR, "lea.html"))

class ComplaintInput(BaseModel):
    fraud_type: str
    fraud_amount: float
    victim_city: str
    complaint_hour: int
    time_to_withdrawal: int

def send_alert_to_n8n(prediction: dict, input_data: ComplaintInput):
    alert_id  = f"ALT-{random.randint(1000,9999)}"
    timestamp = datetime.datetime.now().isoformat()

    if prediction["severity"] == "CRITICAL":
        alert_type = "ZONE_RED_FLAG"
        recipients = ["Cyber Cell HQ","Regional Banking Risk Unit","Local Control Room"]
    else:
        alert_type = "HIGH_PRIORITY_ALERT"
        recipients = ["Regional Banking Risk Unit","Local Cyber Cell"]

    payload = {
        "alert_id":           alert_id,
        "timestamp":          timestamp,
        "alert_type":         alert_type,
        "severity":           prediction["severity"],
        "fraud_type":         input_data.fraud_type,
        "fraud_amount":       input_data.fraud_amount,
        "victim_city":        input_data.victim_city,
        "predicted_city":     prediction["predicted_city"],
        "risk_score":         prediction["risk_score"],
        "is_high_risk":       prediction["is_high_risk"],
        "recipients":         recipients,
        "recommended_action": prediction["action"],
        "complaint_hour":     input_data.complaint_hour,
        "time_to_withdrawal": input_data.time_to_withdrawal,
    }

    # Log to memory
    alert_log.append(payload)

    try:
        response = requests.post(N8N_WEBHOOK_URL, json=payload, timeout=5)
        if response.status_code == 200:
            print(f"✅ Alert sent: [{prediction['severity']}] {input_data.fraud_type} → {prediction['predicted_city']}")
            return {"sent": True, "alert_id": alert_id}
        else:
            print(f"⚠️ n8n status {response.status_code}")
            return {"sent": False, "alert_id": alert_id}
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to n8n")
        return {"sent": False, "alert_id": alert_id, "error": "n8n not running"}
    except requests.exceptions.Timeout:
        return {"sent": False, "alert_id": alert_id, "error": "timeout"}

def build_features(data: ComplaintInput):
    row = {}
    row['fraud_amount']          = data.fraud_amount
    row['time_to_withdrawal']    = data.time_to_withdrawal
    row['complaint_hour']        = data.complaint_hour
    row['day_of_week']           = 0
    row['is_night']              = 1 if (data.complaint_hour >= 22 or data.complaint_hour <= 5) else 0
    row['log_fraud_amount']      = np.log1p(data.fraud_amount)
    if data.fraud_amount < 20000:
        row['amount_bucket_encoded'] = 0
    elif data.fraud_amount < 80000:
        row['amount_bucket_encoded'] = 1
    else:
        row['amount_bucket_encoded'] = 2
    for ft in FRAUD_TYPES:
        row[f'type_{ft}'] = 1 if data.fraud_type == ft else 0
    try:
        row['victim_city_encoded'] = int(label_encoder.transform([data.victim_city])[0])
    except:
        row['victim_city_encoded'] = 0
    return row

@app.post("/predict")
def predict(data: ComplaintInput):
    row = build_features(data)
    num_cols = ['fraud_amount','time_to_withdrawal','complaint_hour','day_of_week','log_fraud_amount']
    num_vals = pd.DataFrame([[row[c] for c in num_cols]], columns=num_cols)
    scaled   = scaler.transform(num_vals)[0]
    for i, c in enumerate(num_cols):
        row[c] = scaled[i]

    risk_row     = pd.DataFrame([[row.get(f,0) for f in risk_features]], columns=risk_features)
    risk_prob    = float(risk_model.predict_proba(risk_row)[0][1])
    is_high_risk = int(risk_prob > 0.4)

    row['is_high_risk'] = is_high_risk
    city_row  = pd.DataFrame([[row.get(f,0) for f in city_features]], columns=city_features)
    pred_city = city_model.predict(city_row)[0]

    if risk_prob > 0.40:
        severity = "CRITICAL"
        action   = f"Deploy task force immediately. Alert banks and ATMs in {pred_city}. Freeze transaction."
    elif risk_prob > 0.20:
        severity = "HIGH"
        action   = f"Enhanced monitoring. Coordinate with cyber cell in {pred_city}."
    else:
        severity = "LOW"
        action   = "Routine monitoring. Log complaint for records."

    prediction = {
        "risk_score":     round(risk_prob, 4),
        "is_high_risk":   is_high_risk,
        "severity":       severity,
        "predicted_city": pred_city,
        "action":         action,
        "fraud_type":     data.fraud_type,
        "fraud_amount":   data.fraud_amount,
        "victim_city":    data.victim_city,
    }

    alert_status = {"sent": False}
    if severity in ("CRITICAL","HIGH"):
        alert_status = send_alert_to_n8n(prediction, data)

    prediction["alert"] = alert_status
    return prediction

@app.get("/city-risk")
def city_risk():
    city_coords = {
        'Mumbai':    {"lat":19.0760,"lon":72.8777},
        'Kolkata':   {"lat":22.5726,"lon":88.3639},
        'Delhi':     {"lat":28.7041,"lon":77.1025},
        'Bengaluru': {"lat":12.9716,"lon":77.5946},
        'Lucknow':   {"lat":26.8467,"lon":80.9462},
        'Jaipur':    {"lat":26.9124,"lon":75.7873},
        'Indore':    {"lat":22.7196,"lon":75.8577},
        'Pune':      {"lat":18.5204,"lon":73.8567},
        'Hyderabad': {"lat":17.3850,"lon":78.4867},
        'Chennai':   {"lat":13.0827,"lon":80.2707},
    }
    df = pd.read_csv(os.path.join(BASE, "..", "city_risk_stats.csv"))
    result = []
    for _, row in df.iterrows():
        city   = row['withdrawal_city']
        coords = city_coords.get(city, {"lat":20.0,"lon":77.0})
        result.append({
            "city":      city,
            "lat":       coords["lat"],
            "lon":       coords["lon"],
            "total":     int(row['Total_Complaints']),
            "high_risk": int(row['High_Risk_Cases']),
            "score":     round(float(row['Avg_Risk_Score']),4),
            "priority":  row['Priority_Level'],
        })
    return result

@app.get("/alerts")
def get_alerts():
    static_alerts = []
    try:
        with open(os.path.join(BASE, "..", "generated_alerts.json")) as f:
            static_alerts = json.load(f)
    except:
        pass
    return {"session_alerts": alert_log, "static_alerts": static_alerts}

@app.get("/health")
def health():
    return {"status": "ok"}