
# Student Success Analysis using Machine Learning

This project analyzes student success (Dropout, Graduate, Enrolled) using demographic, academic, and economic features. The workflow follows the CRISP-DM methodology and uses modern ensemble learning techniques for predictive modeling.

---

## Objective

To predict student academic outcomes and identify key factors contributing to dropouts using classification models and ensemble techniques.

---

## Dataset

- **Source:** [UCI Machine Learning Repository / Kaggle](https://www.kaggle.com/datasets/alexandressm/student-academic-performance)
- **Shape:** 4,424 students × 37 features
- **Target Classes:**
  - `Dropout`
  - `Enrolled`
  - `Graduate`

---

## Tech Stack

- **Python**
- **Pandas, NumPy**
- **Scikit-learn**
- **XGBoost**
- **CatBoost**
- **Seaborn & Matplotlib**

---

## 🧠 ML Workflow (CRISP-DM)

### 1. Data Understanding
- Dataset exploration
- Target distribution
- Feature types and value ranges

### 2. Data Preparation
- Label encoding & normalization
- Feature engineering: average grades, delta, consistency

### 3. Modeling & Evaluation
- Base models: `RandomForest`, `XGBoost`, `CatBoost`
- Advanced:
  - `VotingClassifier` (soft voting)
  - `StackingClassifier` (meta-learning via `LogisticRegression`)
- Evaluation metrics:
  - Accuracy, F1 Score, Confusion Matrix
  - Cross-validation scores

### 4. Interpretability
- Feature importance plots

> “The goal is to turn data into information, and information into insight — and insight into action.”  

