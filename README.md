# Customer Churn Prediction Pipeline

This repository implements an end-to-end machine learning pipeline to predict enterprise telecommunications customer churn using the Telco Churn dataset. The project addresses key real-world data science challenges, including structural data type parsing and severe class imbalance optimization.

---

## Model Evolution and Trade-off Analysis

The pipeline utilizes a Random Forest Classifier. Optimization was heavily focused on maximizing minority class detection (Recall for churned customers) rather than optimizing purely for raw global accuracy. 

| Pipeline Phase | Global Accuracy | Class 1 (Churn) Precision | Class 1 (Churn) Recall | Key Optimization Strategy |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline Model** | `80.91%` | `0.69` | `0.51` | Default configuration; heavily biased toward the majority class due to data distribution. |
| **Optimized Model** | `79.28%` | `0.59` | `0.70` | Synthetic Minority Over-sampling Technique (SMOTE) balancing applied to training features. |

### Business Rationale
While SMOTE implementation introduced a minor drop in global accuracy, it increased the capture rate of churned customers from 51% to 70%. In a production environment, minimizing false negatives (missing customers who intend to leave) directly saves subscription revenue, validating this trade-off.

---

## Feature Engineering and Data Preprocessing

* **Data Cleaning:** Transformed the `TotalCharges` column from object strings to float numerical data. Handled structural blank spaces (`" "`) caused by new customer accounts with 0-month tenures by forcing conversion via numeric coercion and filling resulting nulls with `0`.
* **Dimensionality Reduction:** Dropped high-cardinality unique identifiers (`customerID`) that yield no statistical predictive power.
* **Categorical Encoding:** Applied explicit binary mapping to the target variable (`Churn`) and executed One-Hot Encoding via dummy variables for multi-class categorical features.
* **Resampling:** Used `imblearn` SMOTE to synthetically scale minority sample distributions during the training stage to ensure balanced classification bounds.

---

## Project Architecture

```text
customer_churn_predictor/
│
├── churn.csv            # Telco Customer Churn source dataset
├── explore.py           # Initial data structure evaluation and schema script
├── train.py             # Preprocessing, SMOTE execution, training, and evaluation pipeline
└── README.md            # Technical project documentation
