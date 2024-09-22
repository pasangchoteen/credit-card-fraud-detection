# Credit Card Fraud Detection

## Project Overview

This project aims to detect fraudulent credit card transactions using a Logistic Regression model. The dataset contains transactions labeled as either legitimate (0) or fraudulent (1). By training on this data, we aim to create a model that accurately identifies fraudulent transactions.

## Project Structure

1. **Data Loading and Exploration**:
   - Load the dataset from `creditcard.csv` into a Pandas DataFrame.
   - Display the first five rows of the data using `data.head()`.
   - Summarize statistics with `data.describe()` and check for null values using `data.isnull().sum()`.
   - The dataset contains **284,807 transactions**, with **492 labeled as fraudulent**.

2. **Data Preparation**:
   - Separate the data into legitimate transactions (`legit`) and fraudulent transactions (`fraud`).
   - The shapes of the two datasets are:
     - Legitimate Transactions: **(284,315, 31)**
     - Fraudulent Transactions: **(492, 31)**
   - To balance the dataset, take a random sample of **492 legitimate transactions** to match the number of fraudulent transactions, resulting in a new dataset with **984 transactions**.

3. **Model Training**:
   - Split the balanced dataset into features (`X`) and target (`Y`), where `Y` is the 'Class' column.
   - Further split the data into training (80%) and testing (20%) sets using `train_test_split()`.
   - Train a Logistic Regression model on the training set.

4. **Model Evaluation**:
   - Calculate the accuracy on the training data:
     - Training Accuracy: **0.944 (94.4%)**
   - Calculate the accuracy on the test data:
     - Testing Accuracy: **0.925 (92.5%)**

## Requirements

To run this project, you need to have the following packages installed:

- Python 3.x
- pandas
- numpy
- scikit-learn

You can install the required packages using:


## Data Sources

**Credit Card Transactions Dataset**: The dataset used is `creditcard.csv`, which contains **284,807 transactions** with features such as transaction amount and a binary label indicating legitimacy.

## Tools

- **Python**: Utilized for data manipulation, statistical modeling, and machine learning.
- **Libraries**: pandas, numpy, scikit-learn

## Detailed Analysis Steps

1. **Data Exploration**:
   - Load the data and display:
     ```python
     data = pd.read_csv('creditcard.csv')
     print(data.head())
     print(data.describe())
     print(data.info())
     ```
   - Check for nulls:
     ```python
     print(data.isnull().sum())
     ```
   - Count transactions:
     ```python
     print(data['Class'].value_counts())
     ```

2. **Data Preparation**:
   - Separate data:
     ```python
     legit = data[data.Class == 0]
     fraud = data[data.Class == 1]
     ```
   - Under-sample legitimate transactions:
     ```python
     legit_sample = legit.sample(n=492)
     new_dataset = pd.concat([legit_sample, fraud], axis=0)
     ```

3. **Model Training**:
   - Define features and targets:
     ```python
     X = new_dataset.drop(columns='Class', axis=1)
     Y = new_dataset['Class']
     ```
   - Split the data:
     ```python
     X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
     ```
   - Train the model:
     ```python
     model = LogisticRegression()
     model.fit(X_train, Y_train)
     ```

4. **Model Evaluation**:
   - Calculate accuracy:
     ```python
     X_train_prediction = model.predict(X_train)
     training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
     print('Accuracy on Training data : ', training_data_accuracy)

     X_test_prediction = model.predict(X_test)
     test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
     print('Accuracy score on Test Data : ', test_data_accuracy)
     ```

## Limitations
- The model may not capture all fraudulent patterns due to data limitations.
- The dataset is imbalanced, with fraudulent transactions being significantly fewer.

## References
- [Kaggle - Credit Card Fraud Detection Dataset](https://www.kaggle.com/dalpozz/creditcard-fraud)


