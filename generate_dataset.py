import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta

# Initialize Faker and seed for reproducibility
fake = Faker('en_IN')
np.random.seed(42)
random.seed(42)

# Configuration
NUM_ROWS = 10000
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2024, 12, 31)

# Categorical data
FRAUD_TYPES = [
    'Phishing', 'UPI Fraud', 'Credit Card Fraud', 
    'Fake Loan App', 'OTP Scam', 'Online Shopping Scam'
]

# Cities (Victim and Withdrawal)
CITIES = [
    'Delhi', 'Mumbai', 'Bengaluru', 'Hyderabad', 'Chennai', 
    'Pune', 'Kolkata', 'Jaipur', 'Indore', 'Lucknow'
]

# Helper function to generate random date
def random_date(start, end):
    delta = end - start
    int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
    random_second = random.randrange(int_delta)
    return start + timedelta(seconds=random_second)

def generate_dataset():
    data = []
    
    for i in range(1, NUM_ROWS + 1):
        # 1. Complaint ID
        complaint_id = f"CMP_{100000 + i}"
        
        # 2. Complaint Time (2023-2024)
        complaint_time = random_date(START_DATE, END_DATE)
        
        # 3. Fraud Type weighted selection
        # UPI and OTP might be more common
        fraud_type = np.random.choice(FRAUD_TYPES, p=[0.15, 0.30, 0.15, 0.10, 0.20, 0.10])
        
        # 4. Fraud Amount
        # Higher amounts for Credit Card and Fake Loan
        if fraud_type in ['Credit Card Fraud', 'Fake Loan App']:
            amount = round(np.random.uniform(10000, 500000), 2)
        else:
            amount = round(np.random.uniform(1000, 100000), 2)
        
        # 5. Victim City
        victim_city = np.random.choice(CITIES)
        
        # 6. Behavioral Pattern: Night-time complaints (10 PM - 5 AM)
        # Adjust time slightly to match pattern for some cases
        if random.random() < 0.3: # 30% chance to force into night time for higher risk scenarios
            # Set hour to be between 22 and 5
            hour = random.choice([22, 23, 0, 1, 2, 3, 4, 5])
            complaint_time = complaint_time.replace(hour=hour, minute=random.randint(0, 59))
        
        # 7. Time to Withdrawal
        # UPI/OTP scams usually faster
        if fraud_type in ['UPI Fraud', 'OTP Scam']:
            time_to_withdrawal = random.randint(5, 120) # Minutes
        else:
            time_to_withdrawal = random.randint(60, 1440) # Minutes
            
        # 8. Withdrawal City
        # High chance of same city or nearby major metro
        if random.random() < 0.7:
             withdrawal_city = victim_city
        else:
             withdrawal_city = np.random.choice(CITIES)
             
        # 9. High Risk Calculation
        is_high_risk = 1 if (amount > 50000 and time_to_withdrawal < 120) else 0
        
        # Behavioral Check: High risk cases often cluster in specific cities
        if is_high_risk == 1 and random.random() < 0.6:
            # Force withdrawal city to be a major hotspot
            withdrawal_city = np.random.choice(['Delhi', 'Mumbai', 'Kolkata'])

        data.append([
            complaint_id, 
            complaint_time, 
            fraud_type, 
            amount, 
            victim_city, 
            time_to_withdrawal, 
            withdrawal_city, 
            is_high_risk
        ])
        
    # Create DataFrame
    columns = [
        'complaint_id', 'complaint_time', 'fraud_type', 'fraud_amount', 
        'victim_city', 'time_to_withdrawal', 'withdrawal_city', 'is_high_risk'
    ]
    df = pd.DataFrame(data, columns=columns)
    
    return df

if __name__ == "__main__":
    print("Generating dataset...")
    df = generate_dataset()
    
    # Validation
    print(f"Generated {len(df)} rows.")
    print("Null values:\n", df.isnull().sum())
    print("Risk distribution:\n", df['is_high_risk'].value_counts())
    
    output_file = "cybercrime_dataset.csv"
    df.to_csv(output_file, index=False)
    print(f"Dataset saved to {output_file}")
