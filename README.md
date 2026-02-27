\# Titanic Survival Prediction (Kaggle)



Predict whether a passenger survived the Titanic disaster using classic tabular ML.  

This project is focused on learning a correct ML workflow: preprocessing, validation, iterative feature improvements, and reproducibility.



\## Project Structure





.

├── src/

│ └── train.py # trains model, prints validation accuracy, writes submission.csv

├── README.md

└── .gitignore





\## Approach



\### Model

\- `RandomForestClassifier` (scikit-learn)



\### Features (current)

\- `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Embarked`



\### Preprocessing (current)

\- `Age`: median imputation (median computed on training data)

\- `Embarked`: mode imputation (mode computed on training data)

\- One-hot encoding for categorical variables (`Sex`, `Embarked`)



\### Validation

\- 80/20 train/validation split (`stratify=y`)

\- Metric: accuracy



\## Results (validation accuracy)



| Experiment | Features | Validation Accuracy |

|---|---|---|

| Baseline | Pclass, Sex, SibSp, Parch | ~0.821 |

| + Age | + Age (median imputation) | ~0.832 |

| + Embarked | + Embarked (mode imputation) | ~0.838 |



\## How to Run



1\. Download the Titanic dataset from Kaggle.

2\. Place `train.csv` and `test.csv` in the project root.

3\. Run:



```bash

python src/train.py

