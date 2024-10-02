import argparse
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error


def load_data(csv_file):
    # Load the CSV file into a pandas DataFrame and drop any rows without a value for return
    data = pd.read_csv(csv_file)
    data = data.dropna(subset=['RET'])

    # Make an X and y set
    X = data.drop(columns=['RET', 'DATE'])
    y = data['RET']

    # Fill missing values with the average for that column
    try:
        X.fillna(X.mean(), inplace=True)
        X = X.select_dtypes(include='number')
        return X, y
    except:
        non_numeric_columns = X.select_dtypes(exclude='number').columns
        print("Non-numeric columns:", non_numeric_columns.tolist())

def create_pipeline(alpha=1.0):
    # Create a pipeline that standardizes features, then applies Ridge regression
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=alpha))
    ])
    return pipeline


def get_scores(pipeline, X, y):
    y_pred = pipeline.predict(X)
    mse = mean_squared_error(y, y_pred)
    r_squared = r2_score(y, y_pred)
    return mse, r_squared

def run_pipeline(train_file, test_file, scoring_metric):
    # Load the data
    X_train, y_train = load_data(train_file)
    X_test, y_test = load_data(test_file)
    X = pd.concat([X_train, X_test], axis=0) 
    y = pd.concat([y_train, y_test], axis=0)

    # Create pipeline
    pipeline = create_pipeline(alpha=1.0)

    pipeline.fit(X_train, y_train)

    train_mse, train_r2 = get_scores(pipeline, X_train, y_train)
    test_mse, test_r2 = get_scores(pipeline, X_test, y_test)
    print(f'Train MSE:\t{train_mse}\nTrain r2:\t{train_r2}\nTest MSE:\t{test_mse}\nTest r2:\t{test_r2}')

    # Cross-validation with mse
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=kf, scoring=scoring_metric)

    if scoring_metric == 'mse':
        # Print cross-validation scores
        print(f"Cross-validation MSE scores: {-scores}")
        print(f"Average MSE: {-scores.mean()}")
    else:
        # Print R^2 scores from cross-validation
        print(f"Cross-validation R² scores: {scores}")
        print(f"Average R² score: {scores.mean()}")
