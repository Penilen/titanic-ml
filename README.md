# 🚢 Titanic Survival Prediction (Kaggle)

## 📌 Project Overview

This project predicts whether a passenger survived the Titanic disaster
using machine learning.

The goal of this project is to build a **clean, reproducible ML
workflow** while incrementally improving performance through controlled
experimentation.

------------------------------------------------------------------------

## 🗂 Project Structure

    titanic-ml/
    │
    ├── src/
    │   └── train.py        # Model training + validation + submission generation
    │
    ├── README.md
    └── .gitignore

------------------------------------------------------------------------

## 🧠 Problem Type

Binary Classification:

-   **1 → Survived**
-   **0 → Did Not Survive**

Dataset: Kaggle Titanic Competition

------------------------------------------------------------------------

## ⚙️ Current ML Pipeline

### Model

-   `RandomForestClassifier`
    -   `n_estimators=100`
    -   `max_depth=5`
    -   `random_state=1`

### Preprocessing

-   **Age** → Median imputation (computed from training data)
-   **Embarked** → Mode imputation (computed from training data)
-   One-hot encoding for categorical variables (`Sex`, `Embarked`)

### Features Used

-   `Pclass`
-   `Sex`
-   `Age`
-   `SibSp`
-   `Parch`
-   `Embarked`

------------------------------------------------------------------------

## 📊 Validation Strategy

-   80/20 Train--Validation Split
-   Stratified sampling (`stratify=y`) to preserve survival ratio
-   Metric: **Accuracy**

------------------------------------------------------------------------

## 📈 Results (Validation Accuracy)

  Experiment    Features                           Accuracy
  ------------- ---------------------------------- ----------
  Baseline      Pclass, Sex, SibSp, Parch          \~0.821
  \+ Age        Added Age (median imputation)      \~0.832
  \+ Embarked   Added Embarked (mode imputation)   \~0.838

------------------------------------------------------------------------

## 🚀 How to Run

1.  Download the Titanic dataset from Kaggle.
2.  Place `train.csv` and `test.csv` in the project root.
3.  Run:

``` bash
python src/train.py
```

This will: - Print validation accuracy - Generate `submission.csv`

------------------------------------------------------------------------

## 🎯 Learning Focus

This project emphasizes:

-   Proper validation and generalization
-   Avoiding data leakage
-   Incremental feature experimentation
-   Clean repository structure
-   Reproducibility

------------------------------------------------------------------------

## 🔮 Planned Improvements

-   Add `Fare` and feature engineering (FamilySize, IsAlone, Title
    extraction)
-   Replace manual preprocessing with `Pipeline` + `ColumnTransformer`
-   Add cross-validation
-   Hyperparameter tuning
-   Feature importance visualization
-   Error analysis

------------------------------------------------------------------------

> This project is part of a structured learning path in applied machine
> learning.
