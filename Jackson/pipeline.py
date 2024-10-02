from pipes import run_pipeline
import argparse
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

if __name__ == "__main__":
    # Create argument parser for command-line argument (CSV file location)
    parser = argparse.ArgumentParser(description='Train a Ridge regression model with cross-validation.')
    parser.add_argument('train_file', type=str, help='Path to the training CSV file')
    parser.add_argument('test_file', type=str, help='Path to the testing CSV file')
    parser.add_argument('--scoring', type=str, default='r2', help="Scoring metric: 'r2' (default) or 'mse'")

    # Parse arguments
    args = parser.parse_args()

    scoring_metric = 'neg_mean_squared_error' if args.scoring == 'mse' else 'r2'

    # Run the pipeline with the specified CSV file
    run_pipeline(args.train_file, args.test_file, scoring_metric)