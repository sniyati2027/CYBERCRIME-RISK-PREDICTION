import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings('ignore')

def generate_risk_report():
    print("Loading Data...")
    try:
        X_train = pd.read_csv('X_train.csv')
        X_test = pd.read_csv('X_test.csv')
        y_train = pd.read_csv('y_train.csv')
        y_test = pd.read_csv('y_test.csv')
    except FileNotFoundError:
        print("Error: Train/Test files not found. Please run feature_engineering.py first.")
        return

    # 1. Prepare Data for Risk Modeling
    # We want to use the trained model to score ALL data (or just test data, but "each complaint" implies we can score everything we have)
    # Let's combine them to provide a comprehensive report on the dataset we have.
    
    # Concatenate inputs
    X_full = pd.concat([X_train, X_test], ignore_index=True)
    # Concatenate targets (withdrawal_city)
    # y files should contain 'withdrawal_city'
    y_full = pd.concat([y_train, y_test], ignore_index=True)
    
    # Verify alignment (shapes)
    if len(X_full) != len(y_full):
        print("Warning: X and y length mismatch after concatenation.")
    
    # 2. Train Risk Model (Logistic Regression)
    # We use the Training set to train the model, then apply it to the Full set.
    # Target: 'is_high_risk'
    
    if 'is_high_risk' not in X_train.columns:
        print("Error: 'is_high_risk' column not found in data.")
        return

    # Features for training: Drop 'is_high_risk'
    features_train = X_train.drop(columns=['is_high_risk'])
    target_train = X_train['is_high_risk']
    
    print("Training Logistic Regression Model for Risk Scoring...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(features_train, target_train)
    
    # 3. Compute Risk Scores
    print("Calculating Risk Scores...")
    # Features for full dataset
    features_full = X_full.drop(columns=['is_high_risk'])
    
    # predict_proba returns [prob_0, prob_1]
    risk_scores = lr_model.predict_proba(features_full)[:, 1]
    
    # 4. Create Analysis DataFrame
    analysis_df = pd.DataFrame()
    analysis_df['withdrawal_city'] = y_full.iloc[:, 0] # Assuming single column in y
    analysis_df['is_high_risk_actual'] = X_full['is_high_risk']
    analysis_df['risk_score'] = risk_scores
    
    # 5. Group and Aggregate
    print("Aggregating Risk Intelligence by City...")
    city_stats = analysis_df.groupby('withdrawal_city').agg(
        Total_Complaints=('risk_score', 'count'),
        High_Risk_Cases=('is_high_risk_actual', 'sum'),
        Avg_Risk_Score=('risk_score', 'mean')
    ).reset_index()
    
    # 6. Rank and Label
    city_stats = city_stats.sort_values(by='Avg_Risk_Score', ascending=False)
    
    # Label top cities (e.g., top 3 or Average score > threshold)
    # Let's define High Priority as top 25% of cities by risk score or just top N
    # For this output, we'll label the top 5 cities.
    top_n = 5
    city_stats['Priority_Level'] = 'Standard Monitoring'
    city_stats.iloc[:top_n, city_stats.columns.get_loc('Priority_Level')] = 'High Priority Monitoring Zone'
    
    # 7. Output
    # Save CSV for visualization
    city_stats.to_csv('city_risk_stats.csv', index=False)
    print("Stats saved: city_risk_stats.csv")

    with open('risk_intelligence_report.txt', 'w', encoding='utf-8') as f:
        f.write("=== City-Level Risk Intelligence ===\n")
        f.write(city_stats.to_string(index=False) + "\n")
        
        f.write("\n\n=== Actionable Intelligence Explanation ===\n")
        f.write("1. **Risk Score**: The probability (0-1) that a transaction/complaint related to a city is high-risk.\n")
        f.write("2. **High Priority Monitoring Zones**: Cities with the highest average risk scores. These locations are hotspots for fraudulent withdrawals.\n")
        f.write("   - Action: Deploy focused cyber-cells or coordinate with local law enforcement in these specific cities.\n")
        f.write("   - Action: Scrutinize transactions with 'time_to_withdrawal' anomalies originating or ending in these cities.\n")
        f.write("3. **Resource Allocation**: Allocate more investigative resources to cities at the top of this list.\n")
    
    print("Report generated: risk_intelligence_report.txt")

if __name__ == "__main__":
    generate_risk_report()
