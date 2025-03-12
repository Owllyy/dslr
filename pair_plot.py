import pandas as pandas
import matplotlib.pyplot as plt
from utils import *
from maths import *

HOUSE_LABEL = "Hogwarts House"
COLORS = ['red', 'blue', 'purple', 'green']
def main():
    fileDataFrame = pandas.read_csv('./datasets/dataset_train.csv')
    numericdataFrame = getNumericsFromDataFrame(fileDataFrame)
    numericdataFrame = clearDataFrame(numericdataFrame)
    numericdataFrame = standardizeDataFrame(numericdataFrame)
    numericdataFrame[HOUSE_LABEL] = fileDataFrame[HOUSE_LABEL]
    houses = numericdataFrame[HOUSE_LABEL].unique()
    colors = {house: color for house, color in zip(houses, COLORS)}
    axes = pandas.plotting.scatter_matrix(
        numericdataFrame.drop(HOUSE_LABEL, axis=1),
        figsize=(15, 15),
        marker='o',
        hist_kwds={'bins': 20},
        s=1,
        alpha=0.8,
        c=numericdataFrame[HOUSE_LABEL].map(colors),
    )
    for house, color in colors.items():
        plt.scatter([], [], label=house, color=color)
    plt.legend(title='Maisons', bbox_to_anchor=(1, 0.7))


    for ax in axes.flatten():
        ax.set_xlabel(ax.get_xlabel(), rotation=45, ha='right', fontsize=10)
        ax.set_ylabel(ax.get_ylabel(), rotation=45, ha='right', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])


    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45, ha='right')

    plt.show()

if (__name__ == "__main__"):
    pandas.set_option('display.max_columns', None)
    pandas.set_option('display.max_rows', None)
    main()


