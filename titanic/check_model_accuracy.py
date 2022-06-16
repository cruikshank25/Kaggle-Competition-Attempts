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

# split into validation and training data
train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=1)

# predict on test_X
predict = model.predict(test_X)
print(accuracy_score(test_y,predict))
print(precision_score(test_y,predict))
print(recall_score(test_y,predict))
