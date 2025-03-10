import pandas as pandas
import matplotlib.pyplot as plt
from utils import *
from maths import *
import numpy as numpy

HOUSE_LABEL = "Hogwarts House"

def predictionSubSet(dataFrame: pandas.DataFrame) -> tuple[pandas.DataFrame, tuple[pandas.DataFrame, pandas.DataFrame]]:
    dataFrameSplit = [dataFrame.iloc[:-100], dataFrame.iloc[-100:]]
    predictionSet = dataFrameSplit[1]
    houses = predictionSet['Hogwarts House']
    features =  predictionSet.drop([HOUSE_LABEL], axis='columns')
    trainDataFrame = dataFrameSplit[0]
    return (trainDataFrame, (features, houses))

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
    fileDataFrame = openCsv('./datasets/dataset_train.csv')
    clearedDataFrame = clearDataFrame(fileDataFrame)
    trainDataFrame, (predictionSet, verificationHouses) = predictionSubSet(clearedDataFrame)
    
    houses = trainDataFrame['Hogwarts House']
    numericdataFrame = getNumericsFromDataFrame(trainDataFrame)
    numericdataFrame = standardizeDataFrame(numericdataFrame)
    
    features = [
        # 'Arithmancy',
        # 'Astronomy',
        'Herbology',
        'Defense Against the Dark Arts',
        'Divination',
        #'Muggle Studies',
        'Ancient Runes',
        # 'History of Magic', 
        'Transfiguration',
        # 'Potions',
        # 'Care of Magical Creatures',
        'Charms',
        'Flying'
    ]
    features = numericdataFrame[features]
    ret = fit(features, houses)
    showCost(ret)
    saveThetas(ret[0])

if (__name__ == "__main__"):
    pandas.set_option('display.max_columns', None)
    pandas.set_option('display.max_rows', None)
    main()