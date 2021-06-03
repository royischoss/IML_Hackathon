import pandas as pd
import numpy as np

crimes_dict = {0: 'BATTERY', 1: 'THEFT', 2: 'CRIMINAL DAMAGE',
               3: 'DECEPTIVE PRACTICE', 4: 'ASSAULT'}

def create_df(path):
    df = pd.read_csv(path, index_col=0)
    df = add_primary_code_col(df)
    return df

def add_primary_code_col(df):
    conditions = [
        (df["Primary Type"] == "BATTERY"),
        (df["Primary Type"] == "THEFT"),
        (df["Primary Type"] == "CRIMINAL DAMAGE"),
        (df["Primary Type"] == "DECEPTIVE PRACTICE"),
        (df["Primary Type"] == "ASSAULT")
    ]
    values = crimes_dict.keys()
    df["Primary Code"] = np.select(conditions, values)

