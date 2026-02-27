import pandas as pd
# import the Random Forest model for classification (predicting categories, not numbers)
from sklearn.ensemble import RandomForestClassifier

# load the training data from the csv file into a dataframe
train = pd.read_csv('train.csv')

# load the test data (passengers without survival answers) from the csv file into a dataframe
test = pd.read_csv('test.csv')

# show how many rows and columns the dataset has
print(train.shape)

# show the first 5 rows so you can see what the data looks like
print(train.head())

# show column names, data types, and how many non-null values each column has
print(train.info())

# show how many missing values each column has
print(train.isnull().sum())

# filter the dataset to only rows where Sex is female, then get the Survived column
women = train.loc[train.Sex == 'female']["Survived"]

# calculate the survival rate by dividing number who survived (sum) by total women (len)
# sum works because Survived is 1 for survived and 0 for died
rate_women = sum(women)/len(women)

print("% of women who survived:", rate_women)

# filter the dataset to only rows where Sex is male, then get the Survived column
men = train.loc[train.Sex == 'male']["Survived"]

# calculate the survival rate by dividing number who survived (sum) by total women (len)
# sum works because Survived is 1 for survived and 0 for died
rate_men = sum(men)/len(men)

print("% of men who survived:", rate_men)

# set y to the column we want to predict — survived (1) or died (0)
y = train["Survived"]


# fill missing Age values first
age_median = train["Age"].median()
train["Age"] = train["Age"].fillna(age_median)
test["Age"] = test["Age"].fillna(age_median)


# choose which columns to use as features for prediction
features = ["Pclass", "Sex", "Age", "SibSp", "Parch"]

# pd.get_dummies converts text columns into numbers
# for example Sex becomes two columns: Sex_male (1/0) and Sex_female (1/0)
# because ML models can't work with text, only numbers
X = pd.get_dummies(train[features])
X_test = pd.get_dummies(test[features])

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# create the model
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

# split data BEFORE training
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y
)

# train only on training split
model.fit(X_train, y_train)

# validate
val_predictions = model.predict(X_val)
val_accuracy = accuracy_score(y_val, val_predictions)

print("Validation Accuracy:", val_accuracy)

# NOW retrain on full data before final submission
model.fit(X, y)

# predict on Kaggle test set
predictions = model.predict(X_test)

# create a dataframe with passenger IDs and their predicted survival
# this is the format Kaggle expects for submission
output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})

# save it to a csv file
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")