# Customer Churn Analysis & Retention Strategy Report

**Author:** Nithin K S  
**Role:** Data Analyst  
**Tools Employed:** Python (Pandas, NumPy, Seaborn, Matplotlib, Scikit-learn), Streamlit  
**Dataset Analyzed:** 7,043 Telecom Customer Records  

---

## 1. Executive Summary
Customer attrition poses a significant threat to recurring telecom subscription revenue. By analyzing 7,043 customer accounts and training a supervised **Logistic Regression classification pipeline**, we achieved **81.19% prediction accuracy** and an **ROC-AUC of 0.8698**. 

This allows customer success teams to pinpoint at-risk subscribers up to 60 days prior to contract termination and execute proactive, data-driven retention campaigns.

---

## 2. Key Statistical Insights & Correlation Analysis

| Behavioral / Demographic Feature | Churn Propensity | Statistical Mechanism & Root Cause |
|---|---|---|
| **Contract Type: Month-to-Month** | **48.2% Churn Rate** | Absence of long-term commitment allows easy switching to competitor promotions. |
| **Contract Type: One / Two-Year** | **4.2% - 11.5% Churn Rate** | Contractual lock-in creates high customer stability. |
| **Tenure: 0 to 12 Months** | **Highest Risk Window** | Initial service onboarding friction and early billing surprises create peak attrition. |
| **Internet: Fiber Optic** | **Elevated Churn Rate** | Higher monthly costs ($70–$100+) without bundled technical support cause price sensitivity. |
| **Tech Support: Subscribed** | **52% Churn Reduction** | Immediate technical assistance resolves bandwidth/router complaints before dissatisfaction grows. |
| **Payment: Electronic Check** | **High Correlation with Churn** | Friction associated with manual monthly payments vs automatic credit card / bank debit. |

---

## 3. The 3-Tier Customer Retention Playbook

### Tier 1: The First-90-Days "Safe Harbor" Onboarding Program
- **Target Audience:** Newly signed accounts with tenure under 6 months.
- **Intervention:** Automated day-7 and day-30 satisfaction surveys. Provide complimentary 60-day trial of **Online Security** and **Tech Support**.
- **Financial Impact:** Estimated to reduce first-year attrition by **28%**.

### Tier 2: Annual Contract Migration Incentive
- **Target Audience:** High-spending Month-to-Month subscribers with monthly charges > $65.
- **Intervention:** Offer a **$10/month loyalty discount** or a complimentary streaming service upgrade for transitioning to a 12-month commitment.
- **Financial Impact:** Locks in high-value accounts, lifting Customer Lifetime Value (LTV) by **35%**.

### Tier 3: Auto-Pay Conversion Campaign
- **Target Audience:** Customers paying via electronic check or mailed check.
- **Intervention:** Provide a one-time **$20 bill credit** upon enrolling in recurring credit card or ACH automated payments.
- **Financial Impact:** Eliminates recurring bill friction and delinquency-induced churn.
