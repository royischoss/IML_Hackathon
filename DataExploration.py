import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import CreateDataframe

def plot_histogram(title, df):
    fig, ax = plt.subplots()
    ax.hist(x=df["Primary Code"], histtype="bar", bins=[0, 1, 2, 3, 4, 5],width=0.5)
    ax.set_xticks([0.3, 1.3, 2.3, 3.3, 4.3])
    ax.set_xticklabels(CreateDataframe.crimes_dict.values(), rotation=10)
    plt.title(title)
    plt.show()

def plot_label_histograms():
    plot_histogram("Overall dataset", CreateDataframe.create_df("Dataset/dataset_crimes.csv"))
    plot_histogram("Train Dataset", CreateDataframe.create_df("Dataset/train.csv"))
    plot_histogram("Validate Dataset", CreateDataframe.create_df("Dataset/validate.csv"))
    plot_histogram("Test Dataset", CreateDataframe.create_df("Dataset/test.csv"))


if __name__ == "__main__":
    plot_label_histograms()