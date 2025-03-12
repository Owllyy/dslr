import pandas as pandas
import matplotlib.pyplot as plt
from utils import *
from maths import *
import numpy as numpy
import sys

def showCost(costs: tuple[list[float], str]):
    plt.figure(figsize=(10, 6))

    for values, house in costs[1]:
        plt.plot(values, label=house)

    plt.xlabel('Itération')
    plt.ylabel('Coût')
    plt.title('Évolution du coût pour chaque maison')
    plt.legend()
    plt.grid()
    plt.show()

def main():
    filePath = parseArgs(sys.argv, len(sys.argv))
    fileDataFrame = openCsv(filePath)
    clearedDataFrame = clearDataFrame(fileDataFrame)
    trainDataFrame, (predictionSet, verificationHouses) = predictionSubSet(clearedDataFrame)
    
    houses = trainDataFrame['Hogwarts House']
    numericdataFrame = getNumericsFromDataFrame(trainDataFrame)
    numericdataFrame = standardizeDataFrame(numericdataFrame)
    features = numericdataFrame[FEATURES]
    ret = fit(features, houses)
    showCost(ret)
    saveThetas(ret[0])

if (__name__ == "__main__"):
    pandas.set_option('display.max_columns', None)
    pandas.set_option('display.max_rows', None)
    main()