import pandas as pd
import numpy as np

crimes_dict = {0: 'BATTERY', 1: 'THEFT', 2: 'CRIMINAL DAMAGE',
               3: 'DECEPTIVE PRACTICE', 4: 'ASSAULT'}


def create_4_df_splits_processed():
    """
    Creates and returns 4 data splits (original_full, train, validation, test)
    after preprocessing
    """
    original_full = preprocess_data("Dataset/dataset_crimes.csv")
    train = preprocess_data("Dataset/train.csv")
    validation = preprocess_data("Dataset/validate.csv")
    test = preprocess_data("Dataset/test.csv")
    return original_full, train, validation, test


def create_4_df_splits_raw():
    """
    Creates and returns 4 data splits (original_full, train, validation, test)
    using raw data *without* preprocessing
    """
    original_full = pd.read_csv("Dataset/dataset_crimes.csv", index_col=0)
    train = pd.read_csv("Dataset/train.csv", index_col=0)
    validation = pd.read_csv("Dataset/validate.csv", index_col=0)
    test = pd.read_csv("Dataset/test.csv", index_col=0)
    return original_full, train, validation, test


def preprocess_data(path):
    """
    Creates and returns a pandas dataframe from given file path, after
    preprocessing
    """
    df = pd.read_csv(path, index_col=0)
    add_primary_code_col(df)
    return df


def add_primary_code_col(df):
    """
    Adds a primary code column to given df.
    """
    conditions = [
        (df["Primary Type"] == "BATTERY"),
        (df["Primary Type"] == "THEFT"),
        (df["Primary Type"] == "CRIMINAL DAMAGE"),
        (df["Primary Type"] == "DECEPTIVE PRACTICE"),
        (df["Primary Type"] == "ASSAULT")
    ]
    values = crimes_dict.keys()
    df["Primary Code"] = np.select(conditions, values)
