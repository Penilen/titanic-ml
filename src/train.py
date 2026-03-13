import pandas as pd
import matplotlib.pyplot as plt

# Tools for cross-validation
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Tools for building a machine learning pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Tools for preprocessing data
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Models we want to compare
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


# -------------------------------------------------------
# 1. LOAD THE DATA
# -------------------------------------------------------

# Read the Titanic datasets
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


# -------------------------------------------------------
# 2. CREATE NEW FEATURES
# -------------------------------------------------------

# FamilySize counts how many family members are traveling together
# +1 includes the passenger themselves
train["FamilySize"] = train["SibSp"] + train["Parch"] + 1
test["FamilySize"] = test["SibSp"] + test["Parch"] + 1

# IsAlone is 1 if the passenger is traveling alone, otherwise 0
train["IsAlone"] = (train["FamilySize"] == 1).astype(int)
test["IsAlone"] = (test["FamilySize"] == 1).astype(int)

# Extract title from passenger names
# Example: "Braund, Mr. Owen Harris" -> "Mr"
train["Title"] = train["Name"].str.extract(r",\s*([^\.]+)\.")
test["Title"] = test["Name"].str.extract(r",\s*([^\.]+)\.")

# Group uncommon titles into one category to reduce noise
rare_titles = [
    "Lady", "Countess", "Capt", "Col", "Don", "Dr",
    "Major", "Rev", "Sir", "Jonkheer", "Dona"
]

train["Title"] = train["Title"].replace(rare_titles, "Rare")
test["Title"] = test["Title"].replace(rare_titles, "Rare")

# Normalize equivalent titles into the same category
train["Title"] = train["Title"].replace({
    "Mlle": "Miss",
    "Ms": "Miss",
    "Mme": "Mrs"
})

test["Title"] = test["Title"].replace({
    "Mlle": "Miss",
    "Ms": "Miss",
    "Mme": "Mrs"
})


# -------------------------------------------------------
# 3. DEFINE TARGET AND FEATURES
# -------------------------------------------------------

# The target is what we want to predict
y = train["Survived"]

# These are the columns we will use as input features
features = [
    "Pclass",
    "Sex",
    "Age",
    "Fare",
    "Embarked",
    "Title",
    "FamilySize",
    "IsAlone"
]

# Training features
X = train[features]

# Test features (Kaggle test set)
X_test = test[features]


# -------------------------------------------------------
# 4. DEFINE FEATURE TYPES
# -------------------------------------------------------

# Numeric columns contain numbers
numeric_features = ["Pclass", "Age", "Fare", "FamilySize", "IsAlone"]

# Categorical columns contain categories or labels
categorical_features = ["Sex", "Embarked", "Title"]


# -------------------------------------------------------
# 5. DEFINE PREPROCESSING FOR NUMERIC DATA
# -------------------------------------------------------

# For numeric columns, replace missing values with the median
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)


# -------------------------------------------------------
# 6. DEFINE PREPROCESSING FOR CATEGORICAL DATA
# -------------------------------------------------------

# For categorical columns:
# 1. Fill missing values with the most common value
# 2. Convert categories into numbers
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)


# -------------------------------------------------------
# 7. COMBINE PREPROCESSING STEPS
# -------------------------------------------------------

# Apply numeric preprocessing to numeric columns
# and categorical preprocessing to categorical columns
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)


# -------------------------------------------------------
# 8. DEFINE THE MODELS WE WANT TO COMPARE
# -------------------------------------------------------

# We will test three different models using the same preprocessing
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=1),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=1
    ),
    "Gradient Boosting": GradientBoostingClassifier(random_state=1)
}


# -------------------------------------------------------
# 9. DEFINE CROSS-VALIDATION STRATEGY
# -------------------------------------------------------

# StratifiedKFold keeps the same class balance in each fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)


# -------------------------------------------------------
# 10. EVALUATE EACH MODEL WITH CROSS-VALIDATION
# -------------------------------------------------------

results = {}

for name, model in models.items():
    # Build a full pipeline for this model
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    # Run cross-validation
    scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=cv,
        scoring="accuracy"
    )

    # Save the average score
    results[name] = scores.mean()

    # Print detailed results
    print(f"{name} fold scores: {scores}")
    print(f"{name} mean CV accuracy: {scores.mean():.4f}")
    print("-" * 50)


# -------------------------------------------------------
# 11. CHOOSE THE BEST MODEL
# -------------------------------------------------------

best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

print(f"Best model: {best_model_name}")
print(f"Best mean CV accuracy: {results[best_model_name]:.4f}")


# -------------------------------------------------------
# 12. TRAIN THE BEST MODEL ON ALL TRAINING DATA
# -------------------------------------------------------

best_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", best_model),
    ]
)

best_pipeline.fit(X, y)

from sklearn.inspection import permutation_importance

# -------------------------------------------------------
# 13. PERMUTATION FEATURE IMPORTANCE
# -------------------------------------------------------

# Measure how much each feature affects model accuracy
result = permutation_importance(
    best_pipeline,
    X,
    y,
    n_repeats=10,
    random_state=1,
    scoring="accuracy"
)

# Create a sorted list of feature importances
importance = pd.Series(result.importances_mean, index=X.columns)
importance = importance.sort_values(ascending=False)

print("\nPermutation Feature Importance:")
print(importance)

# -------------------------------------------------------
# 14. VISUALIZE FEATURE IMPORTANCE
# -------------------------------------------------------

# Create a horizontal bar plot
plt.figure(figsize=(8, 5))
importance.plot(kind="barh")

# Labels and title
plt.title("Permutation Feature Importance")
plt.xlabel("Accuracy Decrease When Feature Is Shuffled")
plt.ylabel("Feature")

# Make the most important feature appear on top
plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()


# -------------------------------------------------------
# 15. MAKE PREDICTIONS FOR KAGGLE TEST SET
# -------------------------------------------------------

test_preds = best_pipeline.predict(X_test)


# -------------------------------------------------------
# 16. CREATE KAGGLE SUBMISSION FILE
# -------------------------------------------------------

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": test_preds
})

submission.to_csv("submission.csv", index=False)

print("submission.csv created successfully.")