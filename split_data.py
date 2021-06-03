import pandas as pd

DATA_PATH = "Dataset/dataset_crimes.csv"
TRAIN_VALIDATE = 0.9
RANDOM_SEED = 207

def split_train_test():
    df = pd.read_csv(DATA_PATH)
    train = df.sample(frac=TRAIN_VALIDATE, random_state=RANDOM_SEED)  # random state is a seed value
    test = df.drop(train.index)

    train.to_csv('Dataset/train_validate.csv')
    test.to_csv('Dataset/test.csv')

def split_tran_valid():
    df = pd.read_csv("Dataset/train_validate.csv")
    train = df.sample(frac=TRAIN_VALIDATE, random_state=RANDOM_SEED)  # random state is a seed value
    validate = df.drop(train.index)
    train.to_csv('Dataset/train.csv')
    validate.to_csv('Dataset/validate.csv')

split_tran_valid()
