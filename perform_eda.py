import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create folder for plots if not exists
if not os.path.exists('eda_plots'):
    os.makedirs('eda_plots')

# Load Data
print("Loading dataset...")
df = pd.read_csv('cybercrime_dataset.csv')
df['complaint_time'] = pd.to_datetime(df['complaint_time'])

# 1. Basic Info
print("\n--- Shape and Types ---")
print(df.info())

print("\n--- Summary Statistics ---")
print(df[['fraud_amount', 'time_to_withdrawal']].describe())

# 2. Visualizations

# Set style
sns.set(style="whitegrid")

# a. Distribution of fraud_amount
plt.figure(figsize=(10, 6))
sns.histplot(df['fraud_amount'], bins=50, kde=True, color='red')
plt.title('Distribution of Fraud Amount')
plt.xlabel('Amount (INR)')
plt.savefig('eda_plots/dist_fraud_amount.png')
plt.close()

# b. Distribution of time_to_withdrawal
plt.figure(figsize=(10, 6))
sns.histplot(df['time_to_withdrawal'], bins=50, kde=True, color='blue')
plt.title('Distribution of Time to Withdrawal')
plt.xlabel('Time (Minutes)')
plt.savefig('eda_plots/dist_time_withdrawal.png')
plt.close()

# c. Fraud Type Frequency
plt.figure(figsize=(12, 6))
sns.countplot(y='fraud_type', data=df, order=df['fraud_type'].value_counts().index, palette='viridis')
plt.title('Frequency of Fraud Types')
plt.savefig('eda_plots/freq_fraud_type.png')
plt.close()

# d. Victim City vs Withdrawal City Frequency (Heatmap - Top Cities)
# Create a cross tab
city_crosstab = pd.crosstab(df['victim_city'], df['withdrawal_city'])
plt.figure(figsize=(12, 10))
sns.heatmap(city_crosstab, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Victim City vs Withdrawal City Frequency')
plt.savefig('eda_plots/heatmap_cities.png')
plt.close()

# e. Fraud Type vs Average Amount
plt.figure(figsize=(12, 6))
sns.barplot(x='fraud_amount', y='fraud_type', data=df, estimator=np.mean, ci=None, palette='magma')
plt.title('Average Fraud Amount by Type')
plt.savefig('eda_plots/avg_amount_by_type.png')
plt.close()

# f. Fraud Type vs Time to Withdrawal
plt.figure(figsize=(12, 6))
sns.boxplot(x='time_to_withdrawal', y='fraud_type', data=df, palette='coolwarm')
plt.title('Time to Withdrawal by Fraud Type')
plt.savefig('eda_plots/time_withdrawal_by_type.png')
plt.close()

# g. Hourly High-Risk Cases
df['hour'] = df['complaint_time'].dt.hour
high_risk_hourly = df[df['is_high_risk'] == 1].groupby('hour').size()
plt.figure(figsize=(12, 6))
high_risk_hourly.plot(kind='bar', color='darkred')
plt.title('High-Risk Cases by Hour of Day')
plt.xlabel('Hour (0-23)')
plt.ylabel('Count of High-Risk Cases')
plt.savefig('eda_plots/hourly_high_risk.png')
plt.close()

print("\n--- Visualizations saved to 'eda_plots/' ---")

# 3. Pattern Analysis

# Percentage of High-Risk Cases
high_risk_pct = (df['is_high_risk'].sum() / len(df)) * 100
print(f"\nPercentage of High-Risk Cases: {high_risk_pct:.2f}%")

# Top 5 Withdrawal Cities for High-Risk Cases
top_high_risk_cities = df[df['is_high_risk'] == 1]['withdrawal_city'].value_counts().head(5)
print("\nTop 5 Withdrawal Cities for High-Risk Cases:")
print(top_high_risk_cities)

# Anomalies/Biases Check
print("\n--- Anomalies/Biases ---")
# Check for extremely short withdrawal times
short_withdrawals = df[df['time_to_withdrawal'] < 10]
print(f"Number of cases with < 10 min withdrawal: {len(short_withdrawals)}")

# Check for correlation between amount and time
corr = df[['fraud_amount', 'time_to_withdrawal']].corr().iloc[0,1]
print(f"Correlation between Amount and Time to Withdrawal: {corr:.4f}")
