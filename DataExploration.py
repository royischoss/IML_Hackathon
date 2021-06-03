# import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt
import CreateDataframe


def plot_histogram(title, df):
    """
    Plots a histogram with 5 bins.
    :param title: histogram title
    :param df: dataframe to plot
    """
    fig, ax = plt.subplots()
    ax.hist(x=df, histtype="bar", bins=[0, 1, 2, 3, 4, 5], width=0.5)
    ax.set_xticks([0.3, 1.3, 2.3, 3.3, 4.3])
    ax.set_xticklabels(CreateDataframe.crimes_dict.values(), rotation=10)
    plt.title(title)
    plt.show()


def plot_label_histograms(original_full, train, validation, test):
    """
    Plots 4 label histograms of different splits of the data (All of it, train,
    validation, test)
    """
    plot_histogram("Original Full Dataset", original_full["Primary Code"])
    plot_histogram("Train Dataset", train["Primary Code"])
    plot_histogram("Validation Dataset", validation["Primary Code"])
    plot_histogram("Test Dataset", test["Primary Code"])


if __name__ == "__main__":
    original_full, train, validation, test = CreateDataframe.create_4_df_splits()
    plot_label_histograms(original_full, train, validation, test)