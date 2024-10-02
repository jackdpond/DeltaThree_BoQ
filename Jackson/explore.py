import pandas as pd
import pyarrow as pq
import matplotlib.pyplot as plt
import numpy as np
import threading
import time


def find_correlation(X, Y):
    # Remove any rows where X or Y is NaN
    mask = ~np.isnan(X) & ~np.isnan(Y)
    X_clean = X[mask]
    Y_clean = Y[mask]

    # Check if either X or Y has only constant values
    if np.std(X_clean) == 0 or np.std(Y_clean) == 0:
        return np.nan  # Return NaN if correlation is undefined due to constant values

    # Calculate and return the correlation
    correlation_matrix = np.corrcoef(X_clean, Y_clean)
    return correlation_matrix[0, 1]

def calculate_correlation(X, Y, result):
    """Function to calculate correlation and store the result."""
    try:
        result.append(find_correlation(X, Y))
    except Exception as e:
        print(e) # Handle any errors

def graph_correlation(feature, df):
    X = df[feature]
    Y = df['RET']
    correlation = find_correlation(X, Y)

    # Add labels and title
    plt.title(f'Return vs. {feature}')
    plt.xlabel(feature)
    plt.ylabel(f"Return, Correlation: {correlation:.2f}")

    # Show the plot
    plt.show()

# df = pd.read_parquet("/Users/jackson/competitive/BattleofQuants/data/BOQ_data.parquet")
# mdf = pd.read_parquet("/Users/jackson/competitive/BattleofQuants/data/msf.parquet")

# df.to_csv("/Users/jackson/competitive/BattleofQuants/data/BOQ_data.csv")
# mdf.to_csv("/Users/jackson/competitive/BattleofQuants/data/msf.csv")
    
def main():
    df = pd.read_csv("/Users/jackson/competitive/BattleofQuants/data/BOQ_data.csv")
    df.drop('DATE', axis=1, inplace=True)
    print("read file")
    correlations = []
    Y = df["RET"]
    print("Initialized Y")
    counter = 0
    timeout_threshold = 30
    for col in df.columns:
        X = df[col]
        result = []
        thread = threading.Thread(target=calculate_correlation, args=(X, Y, correlations))

        print(f"start thread {col}")
        start = time.time()
        thread.start()

        thread.join(timeout_threshold)
        print("time taken: ", time.time()-start)

        if thread.is_alive():
            correlations.append("timeout")
            # Optionally, you could terminate the thread here if you have control (not recommended in Python)
            thread.join()  # Ensure the thread is cleaned up

        counter += 1
        print(counter)

    table = {"feature" : [str(a) for a in df.columns],
             "correlation" : correlations}
    
    print(table)
    print(len(df.columns))
    print(len(correlations))
    
    cdf = pd.DataFrame(table)
    cdf_sorted = cdf.reindex(cdf['correlation'].abs().sort_values(ascending=False).index)
    cdf_sorted.reset_index(drop=True, inplace=True)

    cdf_sorted.to_csv("/Users/jackson/competitive/BattleofQuants/data/correlations.csv", index=False)





if __name__ == "__main__":
    #main()

    # df = pd.read_csv('/Users/jackson/competitive/BattleofQuants/data/BOQ_data.csv')
    # print('df loaded')
    # df['DATE'] = pd.to_datetime(df['DATE'])
    # print('dates converted')

    # # Apply a filter to remove rows with dates more recent than 2014
    # df_train = df[df['DATE'] < '2015-01-01']
    # print('df_train made')
    # df_test = df[df['DATE'] > '2015-01-01']
    # print('df_test made')

    # df_train.to_csv('/Users/jackson/competitive/BattleofQuants/data/train.csv', index=False)
    # print('train printed')
    # df_test.to_csv('/Users/jackson/competitive/BattleofQuants/data/test.csv', index=False)
    # print('test printed')

    df = pd.read_csv('/Users/jackson/competitive/BattleofQuants/data/train.csv')
    df.drop('DATE', axis=1, inplace=True)
    for col in df.columns:
        graph_correlation(col, df)
