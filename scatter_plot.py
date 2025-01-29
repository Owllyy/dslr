import pandas as pandas
import matplotlib.pyplot as plt
from utils import *
from maths import *
import numpy as np

def main():
    fileDataFrame = pandas.read_csv('./datasets/dataset_train.csv')
    # (a,b,c,d) = splitHouse(fileDataFrame)
    dataFrameA = fileDataFrame.select_dtypes(include=['number'])
    dataFrameA = dataFrameA.drop(['Index'], axis='columns')
    # print("dataFrameA = ", dataFrameA)
    # dataFrameB = b.select_dtypes(include=['number'])
    # dataFrameB = dataFrameB.drop(['Index'], axis='columns')
    
    x_column = 'Arithmancy'
    y_column = 'Astronomy'

    # Créer un scatter plot
    plt.figure(figsize=(10, 6))
    
    plt.scatter(dataFrameA[x_column], dataFrameA[y_column], color='blue', label='house 1')
    # plt.scatter(dataFrameB[x_column], dataFrameB[y_column], color='red', label='house 2')
    
    # plt.scatter(dataFrame[x_column], dataFrame[y_column], alpha=0.5)
    plt.title(f'Scatter plot de {x_column} vs {y_column}')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.show()
    


if (__name__ == "__main__"):
    pandas.set_option('display.max_columns', None)
    pandas.set_option('display.max_rows', None)
    main()