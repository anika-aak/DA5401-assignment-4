# Credit Card Fraud Detection using GMM-based Synthetic Sampling

## Overview
This project addresses the **credit card fraud detection problem** where the dataset is highly imbalanced (fraud cases <0.2%).  
A baseline Logistic Regression model trained on such data tends to ignore the minority class, leading to poor recall.  

To improve detection, we apply:
- **Gaussian Mixture Model (GMM)-based synthetic sampling** to generate realistic fraud samples.  
- **Clustering-Based Undersampling (CBU)** to reduce the majority class while retaining diversity.  

Both strategies are compared against the baseline model on an unchanged, imbalanced test set.

---

## Dataset
- **Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)  
- **Instances:** 284,807 transactions  
- **Features:** 30 numerical features (PCA-transformed, plus `Time` and `Amount`)  
- **Target:** `Class`  
  - `0` → Non-fraud  
  - `1` → Fraud  

---

## Project Structure

### Part A: Data Exploration and Baseline
- Load and inspect dataset  
- Handle duplicates and missing values (none found)  
- Visualize class imbalance (bar and pie charts)  
- Stratified train-test split (75/25)  
- Baseline Logistic Regression without resampling  
- Evaluation: precision, recall, F1, ROC-AUC, PR-AUC  

### Part B: GMM-based Oversampling
- Explain differences: **SMOTE vs GMM**  
- Fit GMM on minority class only  
- Determine number of components (k) using **BIC/AIC**  
- Generate synthetic minority samples  
- Balance dataset by oversampling to match majority  

### Part C: CBU + GMM
- Apply **Clustering-Based Undersampling (CBU)** to reduce majority class  
- Generate additional minority samples with GMM to match reduced majority  
- Create a balanced dataset with both strategies combined  

### Part D: Model Training & Evaluation
- Train Logistic Regression on:
  - GMM-balanced data  
  - CBU+GMM-balanced data  
- Evaluate on **original imbalanced test set**  
- Compare results with baseline  

---

## Key Findings
- **Baseline:** High precision (~0.85) but limited recall (~0.64) → misses many fraud cases.  
- **GMM-balanced:** Recall improved dramatically (~0.86) but precision dropped (~0.08).  
- **CBU+GMM-balanced:** Similar pattern, with slightly higher recall but lower precision.  
- **ROC-AUC remained strong (~0.97)** and **PR-AUC competitive**, confirming separation ability.  

---

## Recommendation
GMM-based oversampling is effective for fraud detection when the goal is to **maximize recall**.  
Although precision decreases due to more false positives, this is an acceptable trade-off in fraud detection where catching fraud is more critical than avoiding additional investigations.  
CBU+GMM offers a more compact and computationally efficient balanced dataset compared to full oversampling.  

---

## Requirements
- Python 3.8+  
- Libraries: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `imblearn`, `kaggle`  

---

## How to Run
1. Upload your `kaggle.json` file to Colab or local environment.  
2. Install requirements:  
   ```bash
   pip install -U kaggle scikit-learn imbalanced-learn matplotlib pandas
