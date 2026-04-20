FROM python:3.11-slim

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python -c "\
import pandas as pd, numpy as np, joblib, os, warnings; \
warnings.filterwarnings('ignore'); \
from sklearn.linear_model import LogisticRegression; \
from sklearn.ensemble import RandomForestClassifier; \
from sklearn.preprocessing import LabelEncoder, StandardScaler; \
from sklearn.model_selection import train_test_split; \
df = pd.read_csv('cybercrime_dataset.csv'); \
df['complaint_time'] = pd.to_datetime(df['complaint_time']); \
df['complaint_hour'] = df['complaint_time'].dt.hour; \
df['day_of_week'] = df['complaint_time'].dt.dayofweek; \
df['is_night'] = df['complaint_hour'].apply(lambda x: 1 if (x >= 22 or x <= 5) else 0); \
df['log_fraud_amount'] = np.log1p(df['fraud_amount']); \
df['amount_bucket'] = pd.qcut(df['fraud_amount'], q=3, labels=['Low','Medium','High']); \
df['amount_bucket_encoded'] = df['amount_bucket'].map({'Low':0,'Medium':1,'High':2}); \
df = pd.get_dummies(df, columns=['fraud_type'], prefix='type', drop_first=False); \
le = LabelEncoder(); \
df['victim_city_encoded'] = le.fit_transform(df['victim_city']); \
drop_cols = ['complaint_id','complaint_time','victim_city','amount_bucket','withdrawal_city']; \
feature_cols = [c for c in df.columns if c not in drop_cols]; \
X = df[feature_cols].copy(); y = df['withdrawal_city']; \
num_cols = ['fraud_amount','time_to_withdrawal','complaint_hour','day_of_week','log_fraud_amount']; \
sc = StandardScaler(); X[num_cols] = sc.fit_transform(X[num_cols]); \
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y); \
y_risk_tr = X_tr['is_high_risk']; X_risk_tr = X_tr.drop(columns=['is_high_risk']); \
lr = LogisticRegression(max_iter=1000); lr.fit(X_risk_tr, y_risk_tr); \
rf = RandomForestClassifier(n_estimators=100, random_state=42); rf.fit(X_tr, y_tr); \
os.makedirs('backend/models', exist_ok=True); \
joblib.dump(lr, 'backend/models/risk_model.pkl'); \
joblib.dump(rf, 'backend/models/city_model.pkl'); \
joblib.dump(sc, 'backend/models/scaler.pkl'); \
joblib.dump(le, 'backend/models/label_encoder.pkl'); \
joblib.dump(list(X_risk_tr.columns), 'backend/models/risk_features.pkl'); \
joblib.dump(list(X_tr.columns), 'backend/models/city_features.pkl'); \
print('Models trained!') \
"

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]