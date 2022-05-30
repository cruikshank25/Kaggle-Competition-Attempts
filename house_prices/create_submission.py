import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

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

# best number for max leaf nodes
best_tree_size = 750

# run model and get final predictions
rf_model_on_full_data = RandomForestRegressor(max_leaf_nodes=best_tree_size)
rf_model_on_full_data.fit(train_X, train_y)
test_data_path = 'test.csv'
test_data = pd.read_csv(test_data_path)
test_X = test_data[features]
test_preds = rf_model_on_full_data.predict(test_X)

# Run the code to save predictions in the format used for competition scoring
output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_preds})
output.to_csv('submission.csv', index=False)
