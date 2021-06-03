# import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt
import CreateDataframe
import plotly.graph_objects as go


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
    plt.title("Label Histogram (" + title + ")")
    plt.ylabel("Number of samples")
    plt.show()


def create_label_histograms(original_full, train, validation, test):
    """
    Plots 4 label histograms of different splits of the data (All of it, train,
    validation, test)
    """
    plot_histogram("Original Full Dataset", original_full["Primary Code"])
    plot_histogram("Train Dataset", train["Primary Code"])
    plot_histogram("Validation Dataset", validation["Primary Code"])
    plot_histogram("Test Dataset", test["Primary Code"])

def plot_matrix_cor(df):
    """
    assumes raw df after catagoring data
    :param df:
    """



def plot_correlation_plot(df, df_name, feature1, feature2, style, with_errors):
    """
    Plots one graph of 2-feature correlation.
    :param style: 'lines+markers' or 'markers'
    :param with_errors: True for error bars, False for no error bars
    """
    my_title = df_name + " - Correlation between " + feature1 + " and " + feature2
    if with_errors:
        go.Figure([go.Scatter(x=df[feature1], y=df[feature2], mode=style,
                              error_y=dict(type='percent', value=50, visible=True))],
                  layout=go.Layout(title=my_title, xaxis_title=feature1,
                                   yaxis_title=feature2)).show()
    else:
        go.Figure([go.Scatter(x=df[feature1], y=df[feature2], mode=style)],
                  layout=go.Layout(title=my_title, xaxis_title=feature1,
                                   yaxis_title=feature2)).show()



def create_all_features_correlations(df, df_name):
    """
    Plots several 2-feature correlation graphs.
    To add a plot, call plot_correlation_plot with desired 2 features, style
    (lines or dots, and error mode (True for error bars, False for no error bars)
    """
    lines, dots = 'lines+markers', 'markers'     # styles to choose from
    plot_correlation_plot(df, df_name, "District", "Ward", dots, True)


if __name__ == "__main__":
    original_full_p, train_p, validation_p, test_p = CreateDataframe.create_4_df_splits_processed()
    original_full_r, train_r, validation_r, test_r = CreateDataframe.create_4_df_splits_raw()
    create_label_histograms(original_full_p, train_p, validation_p, test_p)
    create_all_features_correlations(train_p, "Train dataset")
