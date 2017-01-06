import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np
from sklearn.model_selection import GridSearchCV

np.random.seed(42)

def train_val_split(df):
    '''Makes a train/val split based on time series.  Uses the first 1000 observations for training.  This will provide somewhat more training data for San Juan, which may lead to revisions later.
    Input: pandas dataframe
    Output: 2 pandas dataframes
    '''
    train = df.head(1000)
    val = df.tail(df.shape[0] - 1000)
    train_y = train.pop('total_cases').values
    train_X = train.values
    val_y = val.pop('total_cases').values
    val_X = val.values
    return train_X, train_y, val_X, val_y

def random_forest_regressor(train_X, train_y):
    '''Fits a random forest regressor to the data
    Input: 2 pandas dataframes
    Output: sklearn random forest regrssor model
    '''

    rf = RandomForestRegressor()
    rf.fit(train_X, train_y)
    return rf

def score_model(model, X, y):
    preds = model.predict(X)
    score = mean_absolute_error(y, preds)
    print score
    return score

def rf_grid_search(model, X, y):
    parameters = {'n_estimators':[5, 10, 20], 'max_depth':[None, 5, 10], 'criterion':['mae']}
    grid = GridSearchCV(model, parameters)
    grid.fit(X, y)
    print grid.score(X, y)
    print grid.get_params()
    score_model(grid, val_X, val_y)
    test_preds = grid.predict(test_df.values)

    return test_preds

if __name__ == '__main__':
    train_df = pd.read_csv('../../data/processed/train_clean.csv')
    test_df = pd.read_csv('../../data/processed/test_features_clean.csv')
    train_X, train_y, val_X, val_y = train_val_split(train_df)

    test_preds = rf_grid_search(RandomForestRegressor(), train_X, train_y)

    # rf = random_forest_regressor(train_X, train_y)
    # score_model(rf, val_X, val_y)
    # score_model(rf, train_X, train_y)
    #
    # test_df['total_cases'] = rf.predict(test_df.values)
    # test_df['total_cases'] = test_df['total_cases'].astype(int)

    test_df['total_cases'] = test_preds
    test_df['total_cases'] = test_df['total_cases'].astype(int)
    test_df = test_df[['sj','year','weekofyear','total_cases']]
    test_df['city'] = test_df['sj'].map(lambda x: 'sj' if x == 1.0 else 'iq')
    test_df.drop(['sj'], axis=1, inplace=True)
    test_df = test_df[['city', 'year', 'weekofyear', 'total_cases']]
    test_df.to_csv('../../models/random_forest_submission.csv', index=False)
