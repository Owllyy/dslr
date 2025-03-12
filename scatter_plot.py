import pandas as pandas
import matplotlib.pyplot as plt
from utils import *
from maths import *

HOUSE_LABEL = "Hogwarts House"
def main():
    fileDataFrame = pandas.read_csv('./datasets/dataset_train.csv')
    numericdataFrame = getNumericsFromDataFrame(fileDataFrame)
    numericdataFrame = clearDataFrame(numericdataFrame, True)
    numericdataFrame[HOUSE_LABEL] = fileDataFrame[HOUSE_LABEL]
    houses = numericdataFrame[HOUSE_LABEL].unique()
    colors = {house: color for house, color in zip(houses, ['red', 'blue', 'purple', 'green'])}
    x_column = 'Defense Against the Dark Arts'
    y_column = 'Astronomy'

    plt.figure(figsize=(10, 6))
    plt.title(f'Scatter plot de {x_column} vs {y_column}')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    for house, color in colors.items():
        data = numericdataFrame[numericdataFrame[HOUSE_LABEL] == house]
        plt.scatter(
            data[x_column],
            data[y_column],
            color=color,
            label=house
        )
    plt.legend(title='Maisons')
    plt.show()

if (__name__ == "__main__"):
    pandas.set_option('display.max_columns', None)
    pandas.set_option('display.max_rows', None)
    main()