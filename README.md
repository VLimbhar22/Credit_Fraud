# 💳 Credit Card Fraud Detection using Logistic Regression and Sampling Techniques

## 📌 Project Overview

This project tackles the real-world challenge of detecting fraudulent credit card transactions using machine learning. We worked with a synthetic dataset containing over **339,000 transactions**, aiming to identify patterns that distinguish legitimate from fraudulent transactions.

Explored multiple logistic regression models, applied **class imbalance solutions** like **oversampling**, **undersampling**, **SMOTE**, and **ROSE**, and evaluated model performance using AUC, accuracy, sensitivity, and specificity.

## 🧠 Key Objectives

- Preprocess and engineer features from raw transaction data
- Address extreme class imbalance with sampling techniques
- Build and evaluate logistic regression models
- Compare models using cross-validation and ROC curves

## 📂 Dataset

- Source: Synthetic dataset from Kaggle/credit card fraud simulation
- Records: 339,607 transactions
- Target variable: `is_fraud` (0 = Not Fraud, 1 = Fraud)
- Features include: transaction amount, location, job type, time of transaction, and merchant distance

## ⚙️ Tools & Technologies

- **Language**: R
- **Libraries**: `tidyverse`, `ggplot2`, `caret`, `ROSE`, `pROC`, `rpart`, `xgboost`, `themis`, `gridExtra`
- **Sampling Techniques**: Normal, Over, Under, Mixed, ROSE, SMOTE
- **Model Evaluation**: Confusion matrix, ROC curve, AUC score

## 📊 Workflow

1. **Data Preprocessing**
   - Converted datetime, extracted transaction hour & day type
   - Created new features like `cc_holder_age`, `merchant_distance`, `merchant_type`
   - Categorized cities, jobs, and states

2. **Exploratory Data Analysis**
   - Histograms, bar plots, violin plots
   - Correlation matrix using `ggcorrplot`

3. **Modeling Techniques**
   - Baseline Logistic Regression
   - Logistic Regression with:
     - Oversampling
     - Undersampling
     - Mixed Sampling
     - ROSE
     - SMOTE
   - Cross-validation (10-fold)

4. **Performance Comparison**
   - Best AUC from ROSE & SMOTE sampling (~0.90+)
   - Trade-offs between sensitivity and specificity

## 📈 Results Summary

| Sampling Method | AUC    | Accuracy | Sensitivity | Specificity |
|-----------------|--------|----------|-------------|-------------|
| Normal          | 0.85   | 99.4%    | 0.0%        | 99.98%      |
| Over Sampling   | 0.85   | 87.9%    | 70.8%       | 88.0%       |
| Under Sampling  | 0.89   | 87.1%    | 71.3%       | 87.2%       |
| Mixed Sampling  | 0.90   | 87.8%    | 87.9%       | 70.8%       |
| ROSE Sampling   | 0.90   | 86.5%    | 71.6%       | 86.6%       |
| SMOTE (CV)      | 0.90   | 86.5%    | 71.9%       | 86.6%       |

## ✅ Conclusion

Our analysis showed that handling class imbalance is critical in fraud detection. Techniques like **SMOTE** and **ROSE** significantly improve model performance, especially in correctly identifying fraudulent transactions.

## 👩‍💻 Contributors

- **Vedant Shamling Limbhare**
- **Prathyusha Elipay**

## 🔗 Project Link

[GitHub Repository](https://github.com/VLimbhar22/Credit_Fraud/tree/main)

---

Feel free to clone, explore, and adapt this work for learning or production use! 
