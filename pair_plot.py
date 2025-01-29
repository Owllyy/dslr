import pandas as pandas
import matplotlib.pyplot as plt
from utils import *
from maths import *
import numpy as np

def main():
    fileDataFrame = pandas.read_csv('./datasets/dataset_train.csv')
    dataFrame = fileDataFrame.drop(['Index'], axis='columns')
    dataFrame = dataFrame.dropna()
    houses = splitHouse(dataFrame)
    colors = ['red', 'blue', 'purple', 'green']
    i = 0
    finalDataFrame = []
    for house in houses: 
        houseDataFrame = house.select_dtypes(include=['number'])
        houseDataFrame = standardizeDataFrame(houseDataFrame)
        finalDataFrame.append(houseDataFrame)
    g = pandas.plotting.scatter_matrix(pandas.concat(finalDataFrame), figsize=(15,15), marker = 'o', hist_kwds = {'bins': 20}, s = 1, alpha = 0.8, color=colors[0])
    plt.show()

if (__name__ == "__main__"):
    pandas.set_option('display.max_columns', None)
    pandas.set_option('display.max_rows', None)
    main()