# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Data Preprocessing
# Load data from the CSV file (separated by ';' and ',' as decimal)
air_quality_data = pd.read_csv('data/AirQuality.csv', sep=';', decimal=',')

# Show the first 5 rows of the dataset
print(air_quality_data.head())

# Remove the last two columns which have NaN values (using slicing)
air_quality_data = air_quality_data.iloc[:, :-2]

# Show the last 5 rows of the dataset
print(air_quality_data.tail())

# Check the number of rows and columns
print("Shape of the data:", air_quality_data.shape)

# Show the last row of the dataset
print(air_quality_data.loc[[9356]])

# Display the first 9357 rows (removes NaN rows)
air_quality_data = air_quality_data.head(9357)
print(air_quality_data)

# Show first few rows again
print(air_quality_data.head())
print(air_quality_data.tail())
print("Shape after cleanup:", air_quality_data.shape)

# Checking the info of the data
air_quality_data.info()

# Checking the number of missing values in the dataset
print("Missing values count:")
print(air_quality_data.isnull().sum())

# Missing value analysis: Check for -200 values (which indicate missing data)
print("Counting -200 values (representing missing values):")
print(air_quality_data.isin([-200]).sum(axis=0))

# Handling missing values
# Convert all -200 values to NaN and replace them with the column mean
air_quality_data = air_quality_data.replace(to_replace=-200, value=np.nan)
print("After replacing -200 with NaN:")
print(air_quality_data.tail())

# Finding the mean of each column
print("Mean of each column:")
print(air_quality_data.select_dtypes(include='number').mean())

# Replace missing values with the mean of the column
air_quality_data = air_quality_data.fillna(air_quality_data.select_dtypes(include='number').mean())
print("After filling NaN values with mean:")
print(air_quality_data.tail())

# Check again for missing values
print("Missing values count after imputation:")
print(air_quality_data.isnull().sum())

# Check for duplicate rows
duplicate_count = air_quality_data.duplicated().sum()
print(f"Duplicate Rows: {duplicate_count}")

# Remove duplicates if any
air_quality_data.drop_duplicates(inplace=True)

# Convert time from 'HH.MM.SS' to 'HH:MM:SS'
air_quality_data['Time'] = air_quality_data['Time'].str.replace('.', ':', regex=False)
print("After converting time format:")
print(air_quality_data.head())

# Combine Date and Time into a single DateTime column
air_quality_data['DateTime'] = pd.to_datetime(air_quality_data['Date'] + ' ' + air_quality_data['Time'], dayfirst=True)
air_quality_data = air_quality_data.drop(columns=['Date', 'Time'])
print("After combining Date and Time into DateTime:")
print(air_quality_data.head())

# Outlier Detection and Handling
# Selecting only the numeric columns for outlier detection
num_cols = air_quality_data.select_dtypes(include='number').columns

# Identifying and handling outliers using IQR
for column in num_cols:
    q1 = air_quality_data[column].quantile(0.25)
    q3 = air_quality_data[column].quantile(0.75)
    iqr = q3 - q1

    # Setting thresholds to detect outliers
    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr

    # Finding outliers
    outlier_rows = air_quality_data[(air_quality_data[column] < lower_limit) | (air_quality_data[column] > upper_limit)]
    print(f"{column} - Total outliers: {len(outlier_rows)}")

    # Replacing outliers with the median of the column
    col_median = air_quality_data[column].median()
    air_quality_data[column] = air_quality_data[column].apply(lambda x: col_median if x < lower_limit or x > upper_limit else x)

# Data Scaling
# Scale the numeric data using StandardScaler
numeric_cols = air_quality_data.select_dtypes(include='number').columns
scaler = StandardScaler()

# Fit and transform the data
air_quality_data[numeric_cols] = scaler.fit_transform(air_quality_data[numeric_cols])

# Show the scaled data
print("Data after scaling:")
print(air_quality_data.head())
