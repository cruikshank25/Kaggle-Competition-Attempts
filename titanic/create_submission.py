import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


td_fp = 'train.csv'
td = pd.read_csv(td_fp)

# replace missing age values with the mean
td['Age'].fillna(td['Age'].mean(), inplace=True)

# covert Sex to binary value
sx = [
    (td.Sex == 'male'),
    (td.Sex == 'female'),
]

binary = ['1', '0']
td['BinSex'] = np.select(sx, binary)

# drop NaN columns
td1 = td.drop('Cabin', 1)
td2 = td1.drop('Embarked', 1)

# define y (prediction target)
y_unshaped = td2.Survived
y = y_unshaped.values.reshape(891, 1)

# define x (features)
features = ['BinSex', 'Pclass', 'Age', 'Fare']
X = td2[features]

# define model and fit
model = LinearDiscriminantAnalysis()
model.fit(X, y)

# load test data 
test_data_path = 'test.csv'
test_data = pd.read_csv(test_data_path)

# add sex to test data
sx = [
    (test_data.Sex == 'male'),
    (test_data.Sex == 'female'),
]
binary = ['1', '0']
test_data['BinSex'] = np.select(sx, binary)


# replace missing Age & Fare values with the mean
test_data['Age'].fillna(test_data['Age'].mean(), inplace=True)
test_data['Fare'].fillna(test_data['Fare'].mean(), inplace=True)

# feature selection for test data
test_X = test_data[features]

# predict test X
test_preds = model.predict(test_X)

# round prediction values
rounded_preds = np.round(test_preds, 0).astype(int)

# Run the code to save predictions in the format used for competition scoring
output = pd.DataFrame({'PassengerId': test_data.PassengerId,
                       'Survived': rounded_preds})
output.to_csv('submission.csv', index=False)

