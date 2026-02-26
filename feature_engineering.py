import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load Data
print("Loading dataset...")
df = pd.read_csv('cybercrime_dataset.csv')
df['complaint_time'] = pd.to_datetime(df['complaint_time'])

# --- 1. Time-Based Features ---
print("Extracting time features...")
df['complaint_hour'] = df['complaint_time'].dt.hour
df['day_of_week'] = df['complaint_time'].dt.dayofweek # 0=Monday, 6=Sunday
df['is_night'] = df['complaint_hour'].apply(lambda x: 1 if (x >= 22 or x <= 5) else 0)

# --- 2. Amount-Based Features ---
print("Creating amount features...")
df['log_fraud_amount'] = np.log1p(df['fraud_amount'])
# Bucketing: Low, Medium, High based on quantiles
df['amount_bucket'] = pd.qcut(df['fraud_amount'], q=3, labels=['Low', 'Medium', 'High'])
# Encode bucket immediately for ML readiness (0, 1, 2)
bucket_map = {'Low': 0, 'Medium': 1, 'High': 2}
df['amount_bucket_encoded'] = df['amount_bucket'].map(bucket_map)

# --- 3. Encoding Categorical Variables ---
print("Encoding categorical variables...")
# One-Hot Encoding for fraud_type
df = pd.get_dummies(df, columns=['fraud_type'], prefix='type', drop_first=False)

# Label Encoding for victim_city
le_city = LabelEncoder()
df['victim_city_encoded'] = le_city.fit_transform(df['victim_city'])

# --- 4. Define Target and Features ---

target_col = 'withdrawal_city'

# Drop non-ML columns
drop_cols = ['complaint_id', 'complaint_time', 'victim_city', 'amount_bucket', 'withdrawal_city']

feature_cols = [col for col in df.columns if col not in drop_cols]

X = df[feature_cols]
y = df[target_col]

# Identify numerical columns for scaling
num_cols = ['fraud_amount', 'time_to_withdrawal', 'complaint_hour', 'day_of_week', 'log_fraud_amount']
# Scale them
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

print(f"Features developed: {list(X.columns)}")

# --- 5. Data Splitting ---
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Save
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

print("\n--- Summary ---")
print(f"Train Shape: {X_train.shape}")
print(f"Test Shape: {X_test.shape}")
print("Files saved: X_train.csv, X_test.csv, y_train.csv, y_test.csv")
