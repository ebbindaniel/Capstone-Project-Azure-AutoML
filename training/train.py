import sklearn
from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
import azureml.core
import pandas as pd
from azureml.core.run import Run
from sklearn.ensemble import RandomForestRegressor

def clean_data(data):
    data = data.dropna()
    x_df = data.drop(columns = {'mpg'})
    y_df = data['mpg']
    return x_df, y_df

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_estimators', type=int, default=200, help="number of trees in the foreset")
    parser.add_argument('--max_depth', type=int, default=10, help="max number of levels in each decision tree")
    parser.add_argument('--bootstrap', type=bool, default=1, help="method for sampling data points (with or without replacement)")

    args = parser.parse_args()
    
    
    run = Run.get_context()

    run.log("number of trees in the foreset:",args.n_estimators)
    run.log("max number of levels in each decision tree:", int(args.max_depth))
    run.log("method for sampling data points (with or without replacement)", bool(args.bootstrap))

    #Dataset:
    
    ds = pd.read_csv('auto-mpg.csv')
    x, y = clean_data(ds)

    # Split data into train and test sets.

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2) #80 - 20 train test split 

    
    model = RandomForestRegressor(n_estimators = args.n_estimators,max_depth = args.max_depth,bootstrap = args.bootstrap ).fit(x_train, y_train)

    r2_score = model.score(x_test, y_test)
    run.log("r2_score", float(r2_score))

    #save the model to folder outputs
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(model, "outputs/model.joblib")
    
if __name__ == '__main__':
    main()