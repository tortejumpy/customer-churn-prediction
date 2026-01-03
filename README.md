# 📉 Customer Churn Prediction Using Machine Learning

## 📌 Project Overview
Customer churn is a critical challenge for subscription-based businesses, as retaining existing customers is often more cost-effective than acquiring new ones.  
This project focuses on building an **end-to-end machine learning pipeline** to **predict customer churn** and **identify the key factors driving customer attrition**, enabling businesses to take **proactive retention actions**.

The solution combines **data exploration, feature engineering, model development, hyperparameter tuning, and interpretability**, ensuring that results are not only accurate but also actionable for business stakeholders.

---

## 🎯 Business Problem
The company observed a consistent decline in active customers month over month.  
Leadership required:
- Early identification of customers likely to churn  
- Clear explanations of *why* customers are leaving  
- Insights that could directly inform retention strategies  

---

## 🧠 Objectives
- Build a reliable **binary classification model** to predict churn
- Handle **class imbalance** effectively
- Identify and interpret **key churn drivers**
- Translate technical findings into **business recommendations**

---

## 📂 Dataset Description
The dataset consists of customer-level data across three categories:

- **Demographics**: Gender, SeniorCitizen, Partner, Dependents  
- **Account Information**: Tenure, Contract type, Payment Method  
- **Service Details**: Internet Service, Tech Support, Monthly & Total Charges  

**Target Variable**:  
- `Churn` → Yes / No

---

## 🛠️ Tools & Technologies
- **Programming Language**: Python  
- **Libraries**:
  - Data Handling: `pandas`, `numpy`
  - Visualization: `matplotlib`, `seaborn`
  - Modeling: `scikit-learn`
  - Imbalance Handling: `SMOTE`
- **Environment**: Jupyter Notebook

---

## 🔍 Exploratory Data Analysis (EDA)
Key insights uncovered during EDA:

- **Churn Rate**: ~26.5% of customers churned
- **Contract Type**:
  - Month-to-month customers churn significantly more than long-term contracts
- **Tenure**:
  - New customers (first 6 months) are at the highest churn risk
- **Monthly Charges**:
  - Higher charges correlate with increased churn
- **Service Add-ons**:
  - Customers without Tech Support churn more
- **Payment Method**:
  - Electronic check users show higher churn behavior

EDA transformed raw data into meaningful business insights that guided modeling decisions.

---

## 🧹 Data Preprocessing & Feature Engineering
Key preprocessing steps included:

- Handling missing values in `TotalCharges`
- One-hot encoding of categorical variables
- Feature scaling for numeric variables
- Addressing **class imbalance** using **SMOTE**
- Grouping sparse categories to reduce noise

Result: A clean, fully model-ready dataset.

---

## 🤖 Model Development
A layered modeling approach was followed:

1. **Logistic Regression**
   - Used as a baseline for interpretability
2. **Random Forest**
   - Captured nonlinear patterns but showed signs of overfitting
3. **Gradient Boosting (Final Model)**
   - Best balance of performance and generalization
   - Tuned using **GridSearchCV**

---

## ⚙️ Hyperparameter Tuning
Gradient Boosting parameters optimized:
- `n_estimators`
- `learning_rate`
- `max_depth`
- `subsample`

**Scoring Metric**: ROC-AUC  
**Cross-validation**: 5-fold

---

## 📊 Model Evaluation
Evaluation focused on metrics suitable for imbalanced data:

- **ROC-AUC**
- **Recall (Churn class)** – prioritized to catch at-risk customers
- **Precision**
- **F1-Score**
- **Confusion Matrix**
- **ROC Curve**
- **Precision-Recall Curve**

The final model demonstrated:
- Strong discriminatory power (high ROC-AUC)
- High recall with acceptable precision
- Reliable probability estimates for real-world use

---

## 🔎 Model Interpretability
Feature importance analysis revealed:

Top churn drivers:
- Contract type
- Tenure
- Monthly charges
- Tech support availability
- Payment method (Electronic Check)

Model explanations aligned strongly with EDA insights, increasing stakeholder trust.

---

## 💡 Business Recommendations
Based on model insights, the following strategies were proposed:

- Incentivize **month-to-month customers** to move to long-term contracts
- Improve **onboarding experience** for new customers
- Offer **retention discounts** for high-paying customers
- Bundle **Tech Support and Security services**
- Investigate friction in **Electronic Check payment method**

---

## 🏁 Conclusion
This project delivered more than a predictive model—it provided a **data-driven retention strategy**.

By combining:
- Strong EDA
- Thoughtful preprocessing
- Robust machine learning models
- Interpretability and business translation  

the solution enables organizations to **predict churn early**, **understand its causes**, and **act decisively** to improve customer retention.

---

## 📈 Future Enhancements
- Deploy model via REST API
- Real-time churn scoring
- Cost-sensitive modeling
- Survival analysis for churn timing
- Integration with CRM systems

---

## 👤 Author
**Harsh Pandey**  
Aspiring Data Scientist | Machine Learning Enthusiast  

---

⭐ *If you find this project helpful, feel free to star the repository!*
