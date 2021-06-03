import numpy as np
import pandas as pd
from _datetime import datetime
import CreateDataframe as CD

DATA_PATH_TRAIN = "Dataset/train.csv"


def date_process(df):
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['day_of_week'] = df['Date'].dt.day_name()
    df['Date'] = ((df['Date'] - datetime(2021, 1, 1)).dt.total_seconds()) / (24 * 60 * 60)
    print(df['day_of_week'])
    df = pd.get_dummies(df, columns=['day_of_week'])
    df = pd.get_dummies(df, columns=['Location Description'])
    df = pd.get_dummies(df, columns=['Arrest'])

    df.to_csv('Dataset/train_validate_5.csv')
    print(df.head())
    print(df['Date'])





if __name__ == '__main__':
    # df = pd.read_csv(DATA_PATH_TRAIN)
    df = CD.preprocess_data("Dataset/train.csv")
    df.to_csv('Dataset/train_4.csv')


