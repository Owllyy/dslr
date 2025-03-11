import pandas as pandas
import matplotlib.pyplot as plt
from utils import *
from maths import *
import numpy as numpy

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
    fileDataFrame = openCsv('./datasets/dataset_train.csv') #TODO: Ajouter le path du fichier en parametre du programme (Surement faire une gestion d'erreur)
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