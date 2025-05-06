import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from cleaning import load_and_clean_data

sns.set(style='whitegrid', palette='Set2')

def univariate_analysis(df):
    print("\nUnivariate Analysis")
    numeric_cols = df.select_dtypes(include='number').columns
    cat_cols = df.select_dtypes(include='object').columns

    # Histograms
    num_features = len(numeric_cols)
    rows = math.ceil(num_features / 3)
    cols = 3

    fig, axes = plt.subplots(rows, cols, figsize=(18, 14), constrained_layout=True)
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        df[col].hist(ax=axes[i], bins=20, edgecolor='black')
        axes[i].set_title(col, fontsize=10)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Histograms of Numeric Features", fontsize=16)
    plt.show()

    # Boxplots
    n = len(numeric_cols)
    cols = 2
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4), constrained_layout=True)
    axes = axes.flatten()
    for i, col in enumerate(numeric_cols):
        sns.boxplot(x=df[col], ax=axes[i])
        axes[i].set_title(f"Boxplot of {col}", fontsize=10, pad=10)
        axes[i].tick_params(axis='x', labelsize=8)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    fig.suptitle("Boxplots of Numeric Features", fontsize=16)
    plt.show()

    # Countplots for categorical columns
    if len(cat_cols) > 0:
        rows, cols = math.ceil(len(cat_cols) / 2), 2
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5), constrained_layout=True)
        axes = axes.flatten()
        for i, col in enumerate(cat_cols):
            sns.countplot(x=df[col], ax=axes[i])
            axes[i].set_title(f"Countplot of {col}", fontsize=10, pad=10)
            axes[i].tick_params(axis='x', labelrotation=45)
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        fig.suptitle("Countplots of Categorical Features", fontsize=16)
        plt.show()

def bivariate_multivariate_analysis(df):
    print("\nBivariate / Multivariate Analysis")

    # Correlation Heatmap
    plt.figure(figsize=(12, 8), constrained_layout=True)
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Correlation Heatmap", fontsize=16, pad=12)
    plt.show()

    # Pairplot
    selected_cols = df.select_dtypes(include='number').columns[:4]
    sns.pairplot(df[selected_cols], height=2.5)
    plt.suptitle("Pairplot of Selected Features", y=1.02)
    plt.show()

    # Hourly Average Line Plot
    if 'DateTime' in df.columns:
        df['Hour'] = df['DateTime'].dt.hour
        hourly_avg = df.groupby('Hour').mean(numeric_only=True)
        selected = [col for col in ['CO(GT)', 'NOx(GT)', 'NO2(GT)', 'C6H6(GT)'] if col in hourly_avg.columns]

        plt.figure(figsize=(12, 6), constrained_layout=True)
        sns.lineplot(data=hourly_avg[selected])
        plt.title("Hourly Average Trends", fontsize=16, pad=12)
        plt.xlabel("Hour of Day")
        plt.ylabel("Average Pollutant Level")
        plt.legend(selected, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()

    # Scatterplots: T vs other numeric features
    if 'T' in df.columns:
        num_cols = [col for col in df.select_dtypes(include='number').columns if col != 'T']
        rows, cols = math.ceil(len(num_cols) / 2), 2
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4), constrained_layout=True)
        axes = axes.flatten()
        for i, col in enumerate(num_cols):
            sns.scatterplot(x=df['T'], y=df[col], ax=axes[i])
            axes[i].set_title(f"T vs {col}", fontsize=10, pad=10)
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        fig.suptitle("Scatterplots: Temperature vs Other Features", fontsize=16)
        plt.show()

def analyze_target_relationship(df, target='CO(GT)'):
    print(f"\nRelationship with Target Variable: {target}")
    features = [col for col in df.select_dtypes(include='number').columns if col != target]
    rows, cols = math.ceil(len(features) / 2), 2
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4), constrained_layout=True)
    axes = axes.flatten()
    for i, col in enumerate(features):
        sns.scatterplot(x=df[col], y=df[target], ax=axes[i])
        axes[i].set_title(f"{col} vs {target}", fontsize=10, pad=10)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    fig.suptitle(f"Feature Relationships with Target: {target}", fontsize=16)
    plt.show()

if __name__ == "__main__":
    df = load_and_clean_data(verbose=False)
    univariate_analysis(df)
    bivariate_multivariate_analysis(df)
    analyze_target_relationship(df)