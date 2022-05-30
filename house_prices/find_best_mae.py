import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


# get mean absolute error function
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = RandomForestRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)


# import data
home_data_fp = 'train.csv'
home_data = pd.read_csv(home_data_fp)

# define y (prediction target)
y = home_data.SalePrice

# define x (features)
features = ['LotArea', 'BedroomAbvGr', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'OverallQual', 'OverallCond', 'MSSubClass', 'FullBath', 'HalfBath', 'TotRmsAbvGrd', 'Fireplaces', 'MiscVal']
X = home_data[features]

# split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# define model 
model = RandomForestRegressor(random_state=1)
model.fit(train_X, train_y)

# make prediction with training data
predictions = model.predict(X)

# calculate mean error for training data
val_predictions = model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))

# find best max_leaf_nodes setting
for max_leaf_nodes in [2, 5, 25, 50, 100, 250, 500, 750, 1000, 1500, 2000]:
    mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, mae))

