import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta

def two_year_counter(og_df, permno_df, permno, end_date):
    """
    Calculates the amount of positive returns in the last two years.

    Parameters:
        og_df(pd.DataFrame): The original dataframe
        permno_df(pd.DataFrame): The filtered dataframe containing only rows of type permno, sorted by date
        permno(str): The identifier we are paying attention to
        end_date(datetime object): The date of the row we are adding a value for; we are looking at the two years previous to this date

    Returns:
        Directly alters the og_df dataframe by adding a value for the new column '2YCOUNT' for the right row
    """
    # Set two_years_df to be a copy of permno_df containing only rows with dates within the 2 year range
    two_years_df = permno_df[(permno_df['DATE'] >= (end_date - relativedelta(years=2))) &
                 (permno_df['DATE'] <= end_date)]
    # Only continue if there are 12 or more rows in two_years_df
    if len(two_years_df) >= 12:
        # Get the count of positive return values within the range of two years
        positive_count = sum(1 for i in two_years_df['RET'] if i > 0)
        # Get the index of the row we want to change, then add in a value for column '2YCOUNT' for that row
        row_index = og_df[(og_df['PERMNO'] == permno) & (og_df['DATE'] == end_date)].index
        og_df.loc[row_index, '2YCOUNT'] = positive_count / 20


def assemble(df):
    """
    Iterates through each unique permno identifier, creating and sorting a view of the original df filtered by permno, then iterates
    through dates, applying two_year_counter to each one.

    Parameters:
        df(pd.DataFrame): The original dataframe containing all the data

    Returns:
        Directly alters the dataframe by adding all new values for the '2YCOUNT' column
    """
    # Get the list of unique permno identifiers
    list_of_permnos = df['PERMNO'].unique().tolist()

    # Iterate through each permno identifier
    for permno in list_of_permnos:
        # Set permno_df to be a filtered df containing only rows with the given permno id, then sort by date and reset the index
        permno_df = df[df['PERMNO'] == permno]
        permno_df.sort_values(by='DATE', inplace=True)
        permno_df.reset_index(drop=True, inplace=True)

        # Set target_date to be the date two years after the first date in permno_df, and set end_date to be the same
        target_date = permno_df.loc[0, 'DATE'] + relativedelta(years=2)
        end_date = target_date
        # Starting from index 1, check each date until at the target_date
        index = 0
        while end_date < target_date:
            index += 1
            end_date = permno_df.loc[1, 'DATE']

        # Starting at the index of the first row with two years of previous data, call two_year_counter on each row
        for date in list(df['DATE'])[index: ]:
            two_year_counter(df, permno_df, permno, date)


def main():
    """
    Test the above functions on 1000 row version of the og df.
    """
    df = pd.read_csv('data/BOQ_data.csv')
    print('read in data')
    df = df.head(1000)
    df = df.dropna(subset=['RET'])
    df = df.dropna(subset=['DATE'])
    print('prepped data')

    df['DATE'] = pd.to_datetime(df['DATE'])

    df['2YCOUNT'] = np.nan

    assemble(df)

    df.to_csv('Jackson/data/tester.csv', index=False)


if __name__ == "__main__":
    main()
