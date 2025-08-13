# Credit Card Fraud Detection

This project detects fraudulent credit card transactions using **machine learning** techniques. It leverages **data preprocessing, exploratory data analysis (EDA), class imbalance handling (SMOTE)**, and multiple classification algorithms (**Logistic Regression**, **XGBoost**, and **Random Forest**). The aim is to build a high-recall model to identify fraudulent transactions effectively.

---

## 📂 Dataset

The dataset used is **[Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)**, which contains transactions made by European cardholders in September 2013.

- **Rows:** 284,807 transactions  
- **Columns:** 31 features (V1–V28 from PCA, Amount, Time, and Class)  
- **Class Distribution:**  
  - 0 → Non-fraudulent (99.83%)  
  - 1 → Fraudulent (0.17%)  

---

## 🛠️ Project Workflow

### 1️⃣ Data Loading & EDA
- Checked missing values and data distribution.
- Visualized fraud vs non-fraud transaction counts & percentages.
- Analyzed distributions of `Amount` and `Time` for fraud/non-fraud classes.

### 2️⃣ Data Preprocessing
- Dropped `Time` column (not informative).
- Standardized `Amount` using **StandardScaler**.
- Applied **PowerTransformer (Yeo-Johnson)** to reduce skewness.
- Split data into **train (80%)** and **test (20%)** sets.

### 3️⃣ Handling Class Imbalance
- Used **SMOTE (Synthetic Minority Oversampling Technique)** to balance the dataset.

### 4️⃣ Model Training & Evaluation
#### **Logistic Regression**
- Hyperparameter tuning using `GridSearchCV` (parameter: C).
- Optimized for **ROC-AUC** score.
- Evaluated using Accuracy, Sensitivity (Recall), Specificity, F1-score, and ROC Curve.

#### **XGBoost**
- Tuned hyperparameters: `learning_rate`, `subsample`, `max_depth`, `n_estimators`.
- Focused on maximizing ROC-AUC.
- Evaluated on train/test sets with confusion matrix and ROC curve.

#### **Random Forest**
- Tuned hyperparameters: `max_depth`, `min_samples_leaf`, `min_samples_split`, `n_estimators`, `max_features`.
- Evaluated similar to other models.

---

## 📊 Results Summary

| Model               | Accuracy | Sensitivity (Recall) | Specificity | ROC-AUC |
|---------------------|----------|----------------------|-------------|---------|
| Logistic Regression | ~99%     | High                 | High        | High    |
| XGBoost             | ~99%     | High                 | High        | High    |
| Random Forest       | ~99%     | High                 | High        | High    |

> **Note:** Sensitivity is prioritized to catch as many fraudulent transactions as possible.

---

## 📦 Requirements

Install the dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost
```

---

## ▶️ Usage

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/credit-card-fraud-detection.git
   ```
2. Navigate to the project folder:
   ```bash
   cd credit-card-fraud-detection
   ```
3. Open and run the Jupyter Notebook:
   ```bash
   jupyter notebook "credit_card_fraud.ipynb"
   ```

---

## 💾 Model Saving
The best **XGBoost model** is saved as:
```bash
xgb_fraud_model.pkl
```
You can load it in future predictions:
```python
import pickle
with open('xgb_fraud_model.pkl', 'rb') as f:
    model = pickle.load(f)
```

---

## 📌 Key Takeaways
- **SMOTE** significantly improved fraud detection recall.
- **XGBoost** provided the best ROC-AUC performance.
- **Data transformation** reduced skewness, improving model performance.

---


