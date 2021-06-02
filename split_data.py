import pandas as pd

DATA_PATH = "Dataset/dataset_crimes.csv"
TRAIN_VALIDATE = 0.9
RANDOM_SEED = 207

df = pd.read_csv(DATA_PATH)
train=df.sample(frac=TRAIN_VALIDATE,random_state=RANDOM_SEED) #random state is a seed value
test=df.drop(train.index)

train.to_csv('Dataset/train_validate.csv')
test.to_csv('Dataset/test.csv')
