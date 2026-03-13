# Titanic ML Experiments

This file tracks the main experiments, feature changes, and model results during the project.

---

## Experiment 1 — Pipeline + model comparison baseline

### Features
- Pclass
- Sex
- Age
- Fare
- FamilySize
- IsAlone

### Models tested
- Logistic Regression
- Random Forest
- Gradient Boosting

### Cross-validation
- 5-fold StratifiedKFold
- Scoring metric: accuracy

### Results
- Logistic Regression: 0.7957
- Random Forest: 0.8272
- Gradient Boosting: 0.8216

### Best model
- Random Forest

### Best mean CV accuracy
- 0.8272

### Notes
- Adding FamilySize and IsAlone improved performance over the simpler feature set.
- Fare also added useful signal.

---

## Experiment 2 — Add Title and Embarked

### New features added
- Title (extracted from Name)
- Embarked

### Title processing
- Extracted titles from passenger names
- Grouped rare titles into "Rare"
- Normalized equivalent titles:
  - Mlle -> Miss
  - Ms -> Miss
  - Mme -> Mrs

### Features
- Pclass
- Sex
- Age
- Fare
- Embarked
- Title
- FamilySize
- IsAlone

### Models tested
- Logistic Regression
- Random Forest
- Gradient Boosting

### Cross-validation
- 5-fold StratifiedKFold
- Scoring metric: accuracy

### Results
- Logistic Regression: 0.8260
- Random Forest: 0.8339
- Gradient Boosting: 0.8339

### Best model
- Gradient Boosting
  - Note: Gradient Boosting and Random Forest had the same displayed score (0.8339), but the script selected Gradient Boosting as best.

### Best mean CV accuracy
- 0.8339

### Improvement over previous best
- Previous best: 0.8272
- New best: 0.8339
- Improvement: +0.0067

### Notes
- Title was a strong feature addition.
- Logistic Regression improved noticeably after adding stronger categorical features.

---

## Experiment 3 — Feature Importance Analysis

### Goal
Understand which features the model relies on most.

### Method
Used permutation feature importance on the trained model.

### Results
Feature importance (accuracy drop when shuffled):

Title         0.203  
Pclass        0.093  
Fare          0.069  
Age           0.043  
FamilySize    0.028  
Sex           0.011  
Embarked      0.003  
IsAlone      -0.0003  

### Observations
- **Title** is by far the most important feature.
- **Pclass** and **Fare** are also strong predictors.
- **Sex** importance decreased because **Title already encodes gender information**.
- **IsAlone** contributes almost no signal.