import os
import joblib
import pandas as pd

def predict_single_customer(customer_data: dict):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'models', 'churn_logistic_regression.pkl')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Run train_model.py first.")
        
    pipeline = joblib.load(model_path)
    df_input = pd.DataFrame([customer_data])
    
    churn_prob = pipeline.predict_proba(df_input)[0, 1]
    churn_pred = pipeline.predict(df_input)[0]
    
    risk_level = "High" if churn_prob >= 0.65 else ("Medium" if churn_prob >= 0.35 else "Low")
    
    return {
        'churn_prediction': "Yes (Will Churn)" if churn_pred == 1 else "No (Will Stay)",
        'churn_probability': round(float(churn_prob), 4),
        'risk_level': risk_level
    }

if __name__ == '__main__':
    # Sample Test Customer 1: High Risk (Month-to-month, new tenure, Fiber Optic, No Tech Support)
    test_cust_high_risk = {
        'gender': 'Female',
        'SeniorCitizen': 0,
        'Partner': 'No',
        'Dependents': 'No',
        'tenure': 2,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'Fiber optic',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'No',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'Yes',
        'StreamingMovies': 'Yes',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 89.50,
        'TotalCharges': 179.00
    }
    
    # Sample Test Customer 2: Low Risk (Two year, 60 months tenure, DSL, Tech Support)
    test_cust_low_risk = {
        'gender': 'Male',
        'SeniorCitizen': 0,
        'Partner': 'Yes',
        'Dependents': 'Yes',
        'tenure': 60,
        'PhoneService': 'Yes',
        'MultipleLines': 'Yes',
        'InternetService': 'DSL',
        'OnlineSecurity': 'Yes',
        'OnlineBackup': 'Yes',
        'DeviceProtection': 'Yes',
        'TechSupport': 'Yes',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'Contract': 'Two year',
        'PaperlessBilling': 'No',
        'PaymentMethod': 'Credit card (automatic)',
        'MonthlyCharges': 65.00,
        'TotalCharges': 3900.00
    }
    
    print("Testing Customer 1 (High Risk Profile):")
    res1 = predict_single_customer(test_cust_high_risk)
    print(res1)
    
    print("\nTesting Customer 2 (Low Risk Profile):")
    res2 = predict_single_customer(test_cust_low_risk)
    print(res2)
