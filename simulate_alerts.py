import pandas as pd
import json
import datetime
import random
import requests

# =============================================
# PASTE YOUR N8N PRODUCTION WEBHOOK URL HERE
# =============================================
N8N_WEBHOOK_URL = "http://localhost:5678/webhook/cybercrime-alert"
# =============================================

def send_to_n8n(alert: dict):
    """Sends a single alert to the n8n webhook."""
    try:
        payload = {
            "alert_id":           alert["alert_id"],
            "timestamp":          alert["timestamp"],
            "type":               alert["type"],
            "severity":           alert["severity"],
            "location":           alert["location"],
            "avg_risk_score":     alert["evidence"].get("avg_risk_score", "N/A"),
            "total_complaints":   alert["evidence"].get("total_complaints", "N/A"),
            "high_risk_cases":    alert["evidence"].get("high_risk_volume", "N/A"),
            "recommended_action": alert["recommended_action"]
        }

        response = requests.post(N8N_WEBHOOK_URL, json=payload, timeout=10)

        if response.status_code == 200:
            print(f"  ✅ Sent to n8n: [{alert['severity']}] {alert['location']}")
        else:
            print(f"  ⚠️  n8n returned status {response.status_code} for {alert['location']}")

    except requests.exceptions.ConnectionError:
        print("  ❌ Could not connect to n8n. Is it running? (run: npx n8n)")
    except requests.exceptions.Timeout:
        print("  ❌ Request timed out.")
    except Exception as e:
        print(f"  ❌ Unexpected error: {e}")


def simulate_alerts():
    print("Initializing Alert Simulation System...")

    # 1. Load Risk Intelligence
    try:
        df = pd.read_csv('city_risk_stats.csv')
    except FileNotFoundError:
        print("Error: city_risk_stats.csv not found.")
        return

    generated_alerts = []
    current_time = datetime.datetime.now().isoformat()

    print(f"Processing {len(df)} monitoring zones...")

    # 2. Risk Threshold Alerts (Static)
    RISK_THRESHOLD = 0.40

    for _, row in df.iterrows():
        if row['Avg_Risk_Score'] > RISK_THRESHOLD:
            alert = {
                "alert_id": f"ALT-{random.randint(1000, 9999)}",
                "timestamp": current_time,
                "type": "ZONE_RED_FLAG",
                "severity": "HIGH",
                "location": row['withdrawal_city'],
                "evidence": {
                    "avg_risk_score":  round(row['Avg_Risk_Score'], 4),
                    "total_complaints": int(row['Total_Complaints']),
                    "high_risk_volume": int(row['High_Risk_Cases'])
                },
                "recipients": ["Cyber Cell HQ", "Regional Banking Risk Unit"],
                "recommended_action": "Deploy specialized task force. Enable enhanced transaction monitoring."
            }
            generated_alerts.append(alert)

    # 3. Simulate Spike Alert (Dynamic)
    spike_city = "Jaipur"
    spike_alert = {
        "alert_id": f"ALT-{random.randint(1000, 9999)}",
        "timestamp": current_time,
        "type": "SURGE_WARNING",
        "severity": "CRITICAL",
        "location": spike_city,
        "evidence": {
            "trigger":       "Sudden Spike in High-Risk Predictions",
            "current_rate":  "50 events/hr",
            "baseline_rate": "5 events/hr",
            "deviation":     "+900%"
        },
        "recipients": ["Local Control Room", "Field Units"],
        "recommended_action": "Immediate on-ground verification. Intercept potential mule operations."
    }
    generated_alerts.append(spike_alert)

    # 4. Save locally
    with open('generated_alerts.json', 'w') as f:
        json.dump(generated_alerts, f, indent=4)

    print(f"\nSimulation Complete. Generated {len(generated_alerts)} alerts.")
    print("Alerts saved to: generated_alerts.json")

    # 5. Send HIGH and CRITICAL alerts to n8n
    print(f"\n--- Sending Alerts to n8n ---")
    for alert in generated_alerts:
        if alert["severity"] in ("HIGH", "CRITICAL"):
            send_to_n8n(alert)

    # 6. Preview
    print("\n--- Alert Log Preview ---")
    for alert in generated_alerts:
        print(f"[{alert['severity']}] {alert['type']} @ {alert['location']}")
        print(f"  Action: {alert['recommended_action']}")
        print("-" * 40)


if __name__ == "__main__":
    simulate_alerts()