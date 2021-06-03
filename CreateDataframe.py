import pandas as pd
import numpy as np
from _datetime import datetime

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
    date_process(df)
    df = dummies(df)
    drop(df)
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


def date_process(df):
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['day_of_week'] = df['Date'].dt.day_name()
    df['Date'] = ((df['Date'] - datetime(2021, 1, 1)).dt.total_seconds()) / (24 * 60 * 60)



def dummies(df):
    mapping = {'TRUE': 1, 'FALSE': 0}
    df = pd.get_dummies(df, columns=['Location Description'])
    df = pd.get_dummies(df, columns=['day_of_week'])
    df.replace({'TRUE': mapping, 'FALSE': mapping})
    df['Arrest'] = df['Arrest'].astype(np.int32)
    df['Domestic'] = df['Domestic'].astype(np.int32)
    return df


def drop(df):
    """
    Drops unnecessary columns.
    """
    df.drop(df.columns[df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
    df.drop('Primary Type', axis=1, inplace=True)
    df.drop('hour', axis=1, inplace=True)
    df.drop('Case Number', axis=1, inplace=True)
    df.drop('Updated On', axis=1, inplace=True)
    df.drop('Block', axis=1, inplace=True)
    df.drop('Location', axis=1, inplace=True)
    df.dropna(inplace=True)


def date_process_task_2(df):
    df['minute'] = df['Date'].dt.minute

    conditions = []
    for i in range(24):
        conditions.append(((df["hour"] == i) & (df["minute"] <= 30)))
        conditions.append(((df["hour"] == i) & (df["minute"] > 30)))

    values = [i for i in range(48)]
    df["minute Code"] = np.select(conditions, values)