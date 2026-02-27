# Import pandas for working with tabular data (DataFrames)
import pandas as pd

# Import RandomForestClassifier (ensemble tree-based classification model)
from sklearn.ensemble import RandomForestClassifier

# Used to split dataset into training and validation sets
from sklearn.model_selection import train_test_split

# Used to evaluate how accurate the model predictions are
from sklearn.metrics import accuracy_score


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

# Load the training dataset.
# This contains both features and the target column "Survived".
train = pd.read_csv("train.csv")

# Load the test dataset.
# This does NOT contain the target column (Survived).
# We will generate predictions for this dataset.
test = pd.read_csv("test.csv")


# --------------------------------------------------
# DEFINE TARGET VARIABLE
# --------------------------------------------------

# The target variable is what we want to predict.
# Survived = 1 (passenger survived)
# Survived = 0 (passenger did not survive)
y = train["Survived"]


# --------------------------------------------------
# HANDLE MISSING VALUES (Age)
# --------------------------------------------------

# Many passengers have missing Age values.
# Machine learning models cannot handle NaN values directly.
# We compute the median Age from the training data.
age_median = train["Age"].median()

# Replace missing Age values in training data with the median.
train["Age"] = train["Age"].fillna(age_median)

# IMPORTANT:
# We use the training median to fill test data as well.
# This prevents data leakage (we do not compute statistics from test data).
test["Age"] = test["Age"].fillna(age_median)


# --------------------------------------------------
# SELECT FEATURES
# --------------------------------------------------

# Define which columns we want to use as model inputs.
# These were chosen based on EDA and domain understanding.
features = ["Pclass", "Sex", "Age", "SibSp", "Parch"]

# Extract selected feature columns from training data.
# pd.get_dummies converts categorical columns (like "Sex")
# into numeric columns using one-hot encoding.
X = pd.get_dummies(train[features])

# Apply the same transformation to the test data.
X_test = pd.get_dummies(test[features])


# --------------------------------------------------
# ENSURE TRAIN AND TEST HAVE SAME COLUMNS
# --------------------------------------------------

# Sometimes a category may appear in training but not in test
# (or vice versa). This can cause mismatched columns.
# Reindex ensures test data has the exact same columns as training.
# Any missing columns are filled with 0.
X_test = X_test.reindex(columns=X.columns, fill_value=0)


# --------------------------------------------------
# DEFINE MODEL
# --------------------------------------------------

# Create the Random Forest classifier.
# n_estimators = number of trees in the forest.
# max_depth = maximum depth of each tree (controls complexity).
# random_state = ensures reproducible results.
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=1
)


# --------------------------------------------------
# TRAIN / VALIDATION SPLIT
# --------------------------------------------------

# Split the dataset into:
# 80% training data
# 20% validation data
# stratify=y ensures the survival ratio remains balanced in both sets.
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=1,
    stratify=y
)


# --------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------

# Fit the model using only the training portion.
model.fit(X_train, y_train)


# --------------------------------------------------
# VALIDATE MODEL
# --------------------------------------------------

# Predict survival on validation data (which the model has never seen).
val_predictions = model.predict(X_val)

# Measure how accurate predictions are.
val_accuracy = accuracy_score(y_val, val_predictions)

print("Validation Accuracy:", val_accuracy)


# --------------------------------------------------
# FINAL TRAINING FOR KAGGLE SUBMISSION
# --------------------------------------------------

# After validation, retrain the model on the full training dataset
# so it can learn from all available labeled data.
model.fit(X, y)

# Generate predictions for the Kaggle test dataset.
predictions = model.predict(X_test)


# --------------------------------------------------
# CREATE SUBMISSION FILE
# --------------------------------------------------

# Kaggle requires a CSV file with exactly two columns:
# PassengerId and Survived.
output = pd.DataFrame({
    "PassengerId": test.PassengerId,
    "Survived": predictions
})

# Save predictions to submission.csv
output.to_csv("submission.csv", index=False)

print("Saved submission.csv")