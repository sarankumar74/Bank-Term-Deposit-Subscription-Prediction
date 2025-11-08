# 🏦 Bank-Term Deposit Subscription Prediction

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Library-Scikit--learn-orange?logo=scikitlearn)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-brightgreen?logo=xgboost)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-blueviolet)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red?logo=streamlit)
![Domain](https://img.shields.io/badge/Domain-Financial%20Services%20%26%20Predictive%20Analytics-navy)

---

## 📘 Overview
**Bank-Term Deposit Subscription Prediction** is a **machine learning project** that predicts whether a bank client will subscribe to a term deposit offer based on demographic, financial, and campaign data.  
Using **advanced ensemble models** and **feature engineering**, this project helps banks optimize marketing efforts, improve customer targeting, and enhance conversion rates.

It demonstrates strong skills in **tabular ML**, **imbalanced data handling**, **model interpretability**, and **Streamlit-based deployment**.

---

## 🎯 Problem Statement
Banks often face challenges in predicting which clients will subscribe to term deposits, resulting in inefficient marketing campaigns and resource waste.

This project aims to build a **binary classification model** that predicts **term deposit subscription likelihood** using historical client and campaign data.

The system focuses on:
- 🧩 Clean, modular ML pipelines using Scikit-learn  
- 🧠 Robust model training with XGBoost, LightGBM, and CatBoost  
- ⚖️ Handling imbalanced data and advanced encoding strategies  
- 📊 Model interpretability with SHAP  
- 🌐 Real-time deployment using Streamlit and AWS  

---

## 💼 Business Use Cases

### 🎯 Targeted Marketing
- Focus marketing efforts on clients with **high conversion probability**  
- Improve campaign ROI through **data-driven prioritization**

### 💰 Cost Optimization
- Reduce marketing costs by eliminating **low-potential leads**  
- Allocate outreach resources effectively  

### 🤝 Customer Retention
- Identify **receptive customers** and tailor personalized offers  
- Improve customer satisfaction and long-term loyalty  

### 🧭 Strategic Decision-Making
- Enable predictive insights for **next-best-action** decisions  
- Support marketing and product teams with **AI-powered recommendations**  

---

## 🧠 Skills Takeaway
- **Python** – Core scripting and ML development  
- **Pandas / NumPy** – Data preprocessing and manipulation  
- **Scikit-learn** – Pipeline creation, model training, and validation  
- **XGBoost / LightGBM / CatBoost** – Advanced tree-based ensemble methods  
- **Model Stacking & Blending** – Ensemble optimization  
- **SHAP Explainability** – Model interpretation and feature insights  
- **Streamlit + AWS** – Interactive web deployment  
- **Feature Engineering** – Encoding, scaling, interaction features  
- **Model Evaluation** – Precision, Recall, F1, ROC-AUC  

---

## 🗺️ Key Development Steps

### 🧾 Step 1: Data Preprocessing & Exploration
- Processed **tabular client and campaign data**  
- Addressed missing values, outliers, and inconsistent entries  
- Encoded categorical variables with **ordered and one-hot encoding**  
- Split data using **Stratified K-Fold Cross Validation**  

### 🧮 Step 2: Feature Engineering
- Created interaction features from key variables (e.g., age × balance, duration × education)  
- Scaled numerical features using StandardScaler  
- Balanced class distribution using **SMOTE / undersampling techniques**  

### 🤖 Step 3: Modeling
#### Baseline Models
- Logistic Regression and Random Forest  

#### Advanced Models
- XGBoost, LightGBM, CatBoost, Gradient Boosting, SVM, Naive Bayes, and Neural Network (Deep Learning)  
- Applied **model stacking and blending** for improved performance  

### 📊 Step 4: Model Evaluation
- Evaluated models using:
  - Accuracy  
  - Precision, Recall, F1-Score  
  - ROC-AUC (primary metric)  
- Used SHAP for **feature interpretability and impact visualization**  

### 🧪 Step 5: Pipeline & Validation
- Built reusable **Scikit-learn pipelines** to prevent data leakage  
- Modularized preprocessing, modeling, and evaluation scripts  
- Ensured reproducibility through **configurable training scripts**  

### 🌐 Step 6: Deployment
- Built **Streamlit web application** for live predictions  
- Integrated with **AWS** for production deployment  
- Enabled real-time input and prediction for business users  

---

<summary>📸 Click to view Streamlit UI screenshots</summary>

#### Home Page  
![Home Page](https://github.com/user-attachments/assets/cd739586-e3aa-4852-8496-89147d4e676e)


#### Results Page  
![Result Page](https://github.com/user-attachments/assets/85db0c8a-20e1-441f-b728-47299de56e73)


---

## 🧩 Project Structure
```bash

Bank-Term-Deposit-Prediction/
│
├── EDA/        
│   └── bank-term-eda.ipynb
│
├── Traning/
│   ├── bank-term.ipynb
│
├── Bank Random Forest model.pkl
│
├── Bank New model.pkl
│ 
├── app/
│   ├── app.py            
│
├── requirements.txt                
└── README.md                      
