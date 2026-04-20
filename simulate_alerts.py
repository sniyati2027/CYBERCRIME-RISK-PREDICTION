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

    # 5. Build one combined message and send as single POST to n8n
    print(f"\n--- Sending Combined Alert to n8n ---")

    high_alerts = [a for a in generated_alerts if a["severity"] in ("HIGH", "CRITICAL")]

    combined = "🚨 CYBERCRIME RISK ALERTS\n"
    combined += f"Generated: {current_time}\n"
    combined += "=" * 30 + "\n"

    for a in high_alerts:
        combined += f"\nAlert ID: {a['alert_id']}\n"
        combined += f"Type: {a['type']}\n"
        combined += f"Severity: {a['severity']}\n"
        combined += f"Location: {a['location']}\n"
        combined += f"Timestamp: {a['timestamp']}\n"
        combined += "Evidence\n"
        combined += f"Avg Risk Score: {a['evidence'].get('avg_risk_score', 'N/A')}\n"
        combined += f"Total Complaints: {a['evidence'].get('total_complaints', 'N/A')}\n"
        combined += f"High Risk Cases: {a['evidence'].get('high_risk_volume', 'N/A')}\n"
        combined += "Recommended Action\n"
        combined += f"{a['recommended_action']}\n"
        combined += "-" * 30 + "\n"

    try:
        payload = {"message": combined.strip()}
        response = requests.post(N8N_WEBHOOK_URL, json=payload, timeout=10)
        if response.status_code == 200:
            print("  ✅ Combined alert sent to n8n!")
        else:
            print(f"  ⚠️  n8n returned status {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("  ❌ Could not connect to n8n. Is it running? (run: npx n8n)")
    except requests.exceptions.Timeout:
        print("  ❌ Request timed out.")
    except Exception as e:
        print(f"  ❌ Unexpected error: {e}")

    # 6. Preview
    print("\n--- Alert Log Preview ---")
    for alert in generated_alerts:
        print(f"[{alert['severity']}] {alert['type']} @ {alert['location']}")
        print(f"  Action: {alert['recommended_action']}")
        print("-" * 40)


if __name__ == "__main__":
    simulate_alerts()