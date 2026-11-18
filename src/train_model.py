import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

def train_churn_model():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'processed', 'cleaned_churn_data.csv')
    models_dir = os.path.join(base_dir, 'models')
    viz_dir = os.path.join(base_dir, 'visualizations')
    reports_dir = os.path.join(base_dir, 'reports')
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    print("Loading cleaned dataset...")
    df = pd.read_csv(data_path)
    
    # 1. Exploratory Visualizations
    sns.set_theme(style="whitegrid")
    
    # Viz A: Churn by Contract Type
    plt.figure(figsize=(8, 5))
    contract_churn = df.groupby(['Contract', 'Churn']).size().unstack(fill_value=0)
    contract_pct = contract_churn.div(contract_churn.sum(axis=1), axis=0) * 100
    ax = contract_pct.plot(kind='bar', stacked=True, color=['#2ca02c', '#d62728'], figsize=(8, 5), width=0.6)
    plt.title('Customer Churn Rate by Contract Type', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Contract Type', fontweight='bold')
    plt.ylabel('Percentage (%)', fontweight='bold')
    plt.xticks(rotation=0)
    plt.legend(title='Churned', loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'churn_by_contract_type.png'), dpi=300)
    plt.close()
    
    # Viz B: Tenure vs Monthly Charges by Churn
    plt.figure(figsize=(9, 5))
    sns.scatterplot(
        data=df.sample(2000, random_state=42), 
        x='tenure', 
        y='MonthlyCharges', 
        hue='Churn', 
        palette={'No': '#1f77b4', 'Yes': '#d62728'}, 
        alpha=0.65,
        s=35
    )
    plt.title('Customer Tenure vs Monthly Charges (Red = Churned)', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Tenure (Months)', fontweight='bold')
    plt.ylabel('Monthly Charges ($)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'tenure_vs_monthly_charges.png'), dpi=300)
    plt.close()
    
    # 2. Features and Target Preparation
    drop_cols = ['customerID', 'Churn', 'ChurnBinary', 'TenureGroup']
    X = df.drop(columns=drop_cols)
    y = df['ChurnBinary']
    
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"Features: {len(num_cols)} numerical, {len(cat_cols)} categorical.")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_cols)
        ]
    )
    
    # 3. Model Pipeline - Logistic Regression
    print("Training Logistic Regression Model...")
    lr_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, C=1.0, random_state=42))
    ])
    
    lr_pipeline.fit(X_train, y_train)
    y_pred = lr_pipeline.predict(X_test)
    y_prob = lr_pipeline.predict_proba(X_test)[:, 1]
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    print("\n--- Model Performance Results ---")
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall:    {rec * 100:.2f}%")
    print(f"F1-Score:  {f1 * 100:.2f}%")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    
    metrics = {
        'model_name': 'Logistic Regression',
        'accuracy': round(acc, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1_score': round(f1, 4),
        'roc_auc': round(roc_auc, 4),
        'test_set_size': len(y_test)
    }
    with open(os.path.join(reports_dir, 'model_performance_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
        
    # 4. Save Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Retained (0)', 'Churned (1)'],
                yticklabels=['Retained (0)', 'Churned (1)'])
    plt.title('Logistic Regression Confusion Matrix', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Predicted Label', fontweight='bold')
    plt.ylabel('True Label', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()
    
    # 5. ROC Curve Plot
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color='#1f77b4', linewidth=2.5, label=f'Logistic Regression (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1.5, label='Random Guess')
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('False Positive Rate', fontweight='bold')
    plt.ylabel('True Positive Rate (Recall)', fontweight='bold')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'roc_curve.png'), dpi=300)
    plt.close()
    
    # 6. Feature Importance (Coefficients)
    model = lr_pipeline.named_steps['classifier']
    feature_names = (
        num_cols + 
        list(lr_pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(cat_cols))
    )
    coefs = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_[0]
    }).sort_values(by='Coefficient', ascending=False)
    
    plt.figure(figsize=(10, 8))
    top_pos_neg = pd.concat([coefs.head(8), coefs.tail(8)]).sort_values(by='Coefficient')
    colors = ['#d62728' if c > 0 else '#2ca02c' for c in top_pos_neg['Coefficient']]
    plt.barh(top_pos_neg['Feature'], top_pos_neg['Coefficient'], color=colors, height=0.6)
    plt.axvline(0, color='black', linestyle='--', linewidth=1)
    plt.title('Key Drivers of Churn: Logistic Regression Coefficients', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Log-Odds Coefficient (Red = Increases Churn, Green = Reduces Churn)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'feature_importance.png'), dpi=300)
    plt.close()
    
    # 7. Save Model Artifact
    model_path = os.path.join(models_dir, 'churn_logistic_regression.pkl')
    joblib.dump(lr_pipeline, model_path)
    print(f"Trained pipeline model saved to {model_path}")
    
    return lr_pipeline

if __name__ == '__main__':
    train_churn_model()
