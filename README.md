# 🏦 Bank Term Deposit Subscription Prediction
🔍 *Machine Learning • Scikit-learn • XGBoost • Streamlit • Streamlit Cloud *

## 🚀 Tech Stack & Domains
![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Library-Scikit--learn-orange?logo=scikitlearn)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-brightgreen?logo=xgboost)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-blueviolet)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red?logo=streamlit)
![Domain](https://img.shields.io/badge/Domain-Financial%20Services%20%26%20Predictive%20Analytics-navy)

---

## 📘 Overview
This project predicts whether a bank client will subscribe to a **term deposit** based on demographic, financial, and campaign behavior data.

The system is designed to:
- Improve marketing ROI via targeted outreach  
- Reduce unnecessary campaign spending  
- Improve customer experience with personalized offers  

This project demonstrates **end-to-end ML lifecycle capability**, including feature engineering, model stacking, interpretability with SHAP, and real-time deployment with Streamlit.

---

## 🎯 Problem Statement
Banks struggle to identify customers likely to subscribe to term deposits, leading to:
- Low conversion rates  
- High marketing expenditure  
- Poor campaign decision-making  

This project builds a binary classification model to score customer likelihood and deliver **predictive insights for next-best action**.

---

## 💼 Business Use Cases
| Use Case | Description |
|---------|-------------|
| 🎯 Targeted Marketing | Focus on customers with high deposit subscription probability |
| 💰 Cost Optimization | Reduce outreach to low-probability leads |
| 🤝 Customer Retention | Tailor offers based on readiness and interest |
| 🧭 Decision Support | Use predictions to support sales and marketing workflows |

---

## 🧠 Model Performance Comparison
| Model | Accuracy | Precision | Recall | F1-Score |
|--------|----------|------------|--------|----------|
| 🌳 DecisionTreeClassifier | **0.90** | 0.60 | 0.61 | 0.60 |
| ⚡ XGBClassifier | **0.93** | 0.76 | 0.63 | 0.69 |
| 👥 KNeighborsClassifier | **0.91** | 0.67 | 0.55 | 0.60 |

🔹 **XGBClassifier achieved the best overall performance** based on accuracy and F1-score.  
🔹 SHAP plots were used to validate interpretability and feature contribution.

---

## 🗺️ Project Workflow

### 🧾 1 — Data Preprocessing
- Cleaned and validated dataset  
- Missing value treatment & outlier removal  
- Ordered + one-hot encoding for categorical variables  
- Stratified train-test split for stable validation  

### 🧮 2 — Feature Engineering
- Interaction features (age × balance, duration × education)  
- Numerical feature scaling  
- SMOTE & undersampling for class imbalance  

### 🤖 3 — Modeling
- Baseline models: Logistic Regression, Random Forest  
- Advanced models: XGBoost, LightGBM, CatBoost, Gradient Boost, SVM, Naive Bayes, ANN  
- Stacking / blending ensembles for performance boost  

### 📊 4 — Evaluation
- Accuracy, Precision, Recall, F1-Score, AUC-ROC  
- SHAP for global + local interpretability  

### 🌐 5 — Deployment
- Streamlit web application for real-time customer scoring  
- AWS deployment for business usability  

---

<summary>📸 Click to view Streamlit UI screenshots</summary>

#### Home Page  
![Home Page](https://github.com/user-attachments/assets/cd739586-e3aa-4852-8496-89147d4e676e)


#### Results Page  
![Result Page](https://github.com/user-attachments/assets/85db0c8a-20e1-441f-b728-47299de56e73)


---


## 📁 Project Structure
```
Bank-Term-Deposit-Prediction/  
│  
├── EDA/  
│   └── bank-term-eda.ipynb  
│  
├── Training/  
│   └── bank-term.ipynb  
│  
├── models/  
│   ├── Bank Random Forest model.pkl  
│   ├── Bank New model.pkl  
│  
├── app/  
│   └── app.py  
│  
├── requirements.txt  
└── README.md  

```
---

## 🛠️ Installation & Execution

Clone repository:
```
git clone https://github.com/sarankumar74/Bank-Term-Deposit-Subscription-Prediction.git
cd Bank-Term-Deposit-Prediction
```

Install dependencies:
```
pip install -r requirements.txt
```

Run Streamlit app:
```
streamlit run app/app.py
```
