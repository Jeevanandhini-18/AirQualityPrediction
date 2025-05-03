import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_clean_data(file_path='data/AirQuality.csv', verbose=False):
    """
    Loads, cleans, and preprocesses air quality data from the specified CSV file.

    Parameters:
        file_path (str): Path to the Air Quality dataset.
        verbose (bool): If True, prints intermediate results for debugging.

    Returns:
        pd.DataFrame: The cleaned and preprocessed DataFrame.
    """
    # Load CSV with appropriate separators
    df = pd.read_csv(file_path, sep=';', decimal=',')

    if verbose:
        print("Initial data shape:", df.shape)
        print(df.head())

    # Remove last two unnamed columns and keep only first 9357 rows
    df = df.iloc[:, :-2]
    df = df.head(9357)

    if verbose:
        print("After removing last two columns and trimming rows:", df.shape)

    # Replace -200 with NaN (representing missing values)
    df.replace(to_replace=-200, value=np.nan, inplace=True)

    if verbose:
        print("Count of -200 values per column:")
        print(df.isin([-200]).sum())

    # Fill NaN values with column mean (only for numeric columns)
    df.fillna(df.select_dtypes(include='number').mean(), inplace=True)

    if verbose:
        print("After handling missing values:")
        print(df.isnull().sum())

    # Drop duplicate rows
    duplicate_count = df.duplicated().sum()
    if verbose:
        print(f"Duplicate rows found: {duplicate_count}")
    df.drop_duplicates(inplace=True)

    # Fix time formatting from 'HH.MM.SS' to 'HH:MM:SS'
    df['Time'] = df['Time'].str.replace('.', ':', regex=False)

    # Combine Date and Time into a single DateTime column
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)
    df.drop(columns=['Date', 'Time'], inplace=True)

    if verbose:
        print("After creating DateTime column:")
        print(df[['DateTime']].head())

    # Outlier detection and handling using IQR method
    num_cols = df.select_dtypes(include='number').columns
    for column in num_cols:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower_limit = q1 - 1.5 * iqr
        upper_limit = q3 + 1.5 * iqr
        median = df[column].median()
        outlier_count = df[(df[column] < lower_limit) | (df[column] > upper_limit)].shape[0]

        if verbose:
            print(f"{column} - Outliers detected: {outlier_count}")

        df[column] = df[column].apply(lambda x: median if x < lower_limit or x > upper_limit else x)

    # Standardize numeric data
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    if verbose:
        print("Final cleaned data preview:")
        print(df.head())

    return df
# Calling to display
if __name__ == "__main__":
    df_cleaned = load_and_clean_data(verbose=True)
    print("Final cleaned data shape:", df_cleaned.shape)
