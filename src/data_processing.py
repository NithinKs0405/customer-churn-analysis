import os
import random
import numpy as np
import pandas as pd

np.random.seed(42)
random.seed(42)

def generate_and_clean_churn_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_path = os.path.join(base_dir, 'data', 'raw', 'telco_customer_churn_raw.csv')
    processed_path = os.path.join(base_dir, 'data', 'processed', 'cleaned_churn_data.csv')
    
    print("Generating Telco Customer Churn dataset (7,043 records)...")
    n_records = 7043
    
    genders = ['Male', 'Female']
    yes_no = ['Yes', 'No']
    internet_services = ['DSL', 'Fiber optic', 'No']
    contracts = ['Month-to-month', 'One year', 'Two year']
    payment_methods = [
        'Electronic check', 
        'Mailed check', 
        'Bank transfer (automatic)', 
        'Credit card (automatic)'
    ]
    
    records = []
    for i in range(1, n_records + 1):
        cid = f"{random.randint(1000, 9999)}-{chr(random.randint(65, 90))}{chr(random.randint(65, 90))}{chr(random.randint(65, 90))}{chr(random.randint(65, 90))}"
        gender = random.choice(genders)
        senior = random.choices([0, 1], weights=[0.84, 0.16])[0]
        partner = random.choice(yes_no)
        dependents = 'Yes' if partner == 'Yes' and random.random() < 0.5 else 'No'
        
        # Tenure
        if random.random() < 0.15:
            tenure = random.randint(0, 3) # very new customer
        elif random.random() < 0.40:
            tenure = random.randint(4, 24)
        else:
            tenure = random.randint(25, 72)
            
        phone = random.choices(yes_no, weights=[0.90, 0.10])[0]
        multilines = random.choice(['Yes', 'No']) if phone == 'Yes' else 'No phone service'
        
        internet = random.choices(internet_services, weights=[0.34, 0.44, 0.22])[0]
        
        if internet != 'No':
            online_sec = random.choices(yes_no, weights=[0.35, 0.65])[0]
            online_bkp = random.choices(yes_no, weights=[0.40, 0.60])[0]
            dev_prot = random.choices(yes_no, weights=[0.40, 0.60])[0]
            tech_sup = random.choices(yes_no, weights=[0.35, 0.65])[0]
            str_tv = random.choices(yes_no, weights=[0.45, 0.55])[0]
            str_mov = random.choices(yes_no, weights=[0.45, 0.55])[0]
        else:
            online_sec = online_bkp = dev_prot = tech_sup = str_tv = str_mov = 'No internet service'
            
        # Contract type correlates with tenure
        if tenure > 36:
            contract = random.choices(contracts, weights=[0.25, 0.35, 0.40])[0]
        elif tenure > 12:
            contract = random.choices(contracts, weights=[0.45, 0.35, 0.20])[0]
        else:
            contract = random.choices(contracts, weights=[0.80, 0.15, 0.05])[0]
            
        paperless = random.choices(yes_no, weights=[0.60, 0.40])[0]
        pay_method = random.choices(payment_methods, weights=[0.35, 0.22, 0.22, 0.21])[0]
        
        # Monthly Charges
        base_charge = 20.0
        if phone == 'Yes': base_charge += 10.0
        if multilines == 'Yes': base_charge += 5.0
        if internet == 'DSL': base_charge += 25.0
        elif internet == 'Fiber optic': base_charge += 45.0
        if online_sec == 'Yes': base_charge += 5.0
        if online_bkp == 'Yes': base_charge += 5.0
        if dev_prot == 'Yes': base_charge += 5.0
        if tech_sup == 'Yes': base_charge += 5.0
        if str_tv == 'Yes': base_charge += 10.0
        if str_mov == 'Yes': base_charge += 10.0
        
        monthly_charge = round(base_charge + random.uniform(-3, 3), 2)
        
        # Total Charges (with whitespace edge case for tenure == 0)
        if tenure == 0:
            total_charges_str = " " # real-world dirty data anomaly!
            total_charge_num = 0.0
        else:
            total_charge_num = round(monthly_charge * tenure * random.uniform(0.96, 1.04), 2)
            total_charges_str = str(total_charge_num)
            
        # Realistic Churn Probability Model matching canonical Telco distribution
        # Base churn rate ~ 26.5%
        churn_score = -2.15 # base log-odds
        
        # Contract impact
        if contract == 'Month-to-month': churn_score += 1.45
        elif contract == 'One year': churn_score -= 0.60
        elif contract == 'Two year': churn_score -= 1.60
        
        # Tenure impact
        if tenure < 6: churn_score += 1.10
        elif tenure < 12: churn_score += 0.65
        elif tenure > 48: churn_score -= 1.30
        
        # Services impact
        if internet == 'Fiber optic': churn_score += 0.75
        elif internet == 'No': churn_score -= 0.80
        
        if tech_sup == 'No' and internet != 'No': churn_score += 0.55
        if online_sec == 'No' and internet != 'No': churn_score += 0.45
        if pay_method == 'Electronic check': churn_score += 0.50
        if senior == 1: churn_score += 0.25
        
        # Convert log-odds to probability
        churn_prob = 1.0 / (1.0 + np.exp(-churn_score))
        churn = 'Yes' if random.random() < churn_prob else 'No'
        
        records.append({
            'customerID': cid,
            'gender': gender,
            'SeniorCitizen': senior,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone,
            'MultipleLines': multilines,
            'InternetService': internet,
            'OnlineSecurity': online_sec,
            'OnlineBackup': online_bkp,
            'DeviceProtection': dev_prot,
            'TechSupport': tech_sup,
            'StreamingTV': str_tv,
            'StreamingMovies': str_mov,
            'Contract': contract,
            'PaperlessBilling': paperless,
            'PaymentMethod': pay_method,
            'MonthlyCharges': monthly_charge,
            'TotalCharges': total_charges_str,
            'Churn': churn
        })
        
    df_raw = pd.DataFrame(records)
    df_raw.to_csv(raw_path, index=False)
    print(f"Raw data saved with {len(df_raw)} records to {raw_path}")
    
    # Cleaning Phase
    print("Performing data cleaning: handling whitespace strings in TotalCharges, type casting...")
    df_clean = df_raw.copy()
    
    # 1. Fix whitespace TotalCharges to float
    df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'].replace(r'^\s*$', np.nan, regex=True), errors='coerce')
    imputed_count = df_clean['TotalCharges'].isna().sum()
    print(f"Detected {imputed_count} records with empty TotalCharges (tenure=0). Imputing with 0.0.")
    df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(0.0)
    
    # 2. Add tenure group feature
    df_clean['TenureGroup'] = pd.cut(
        df_clean['tenure'], 
        bins=[-1, 12, 24, 48, 72], 
        labels=['0-12 Months', '13-24 Months', '25-48 Months', '49-72 Months']
    )
    
    # 3. Churn binary
    df_clean['ChurnBinary'] = df_clean['Churn'].map({'Yes': 1, 'No': 0})
    
    df_clean.to_csv(processed_path, index=False)
    print(f"Cleaned dataset saved to {processed_path}")
    print(f"Overall Churn Rate: {df_clean['ChurnBinary'].mean() * 100:.2f}%")
    
    return df_clean

if __name__ == '__main__':
    generate_and_clean_churn_data()
