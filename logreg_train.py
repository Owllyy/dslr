import pandas as pandas
import matplotlib.pyplot as plt
from utils import *
from maths import *
import numpy as numpy
import sys
import random

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
    
def predict(features, thetas):
    if features.shape[1] != len(thetas[0][0]):
        features = numpy.insert(features, 0, 1, axis=1) # insert intercept value (comme dans dans la fonction fit)
    probabilities = numpy.zeros((features.shape[0], len(thetas)))
    house_names = []
    
    for i, (theta, house) in enumerate(thetas):
        weight = features.dot(theta)
        probabilities[:, i] = sigmoid(weight)
        house_names.append(house)
    
    predicted_indices = numpy.argmax(probabilities, axis=1)
    predictions = [house_names[idx] for idx in predicted_indices]
    
    return predictions

def analyze_features(numericdataFrame: pandas.DataFrame, houses: pandas.Series):
    feature1 = numericdataFrame["Astronomy"]
    feature2 = numericdataFrame["Herbology"]

    # Convertir les maisons en valeurs numériques pour la corrélation (ex. : one-hot encoding simplifié)
    houses_numeric = houses.factorize()[0]  # Convertit en 0, 1, 2, 3 pour Gryffindor, Hufflepuff, etc.

    corr1 = numpy.corrcoef(feature1, houses_numeric)[0, 1]
    corr2 = numpy.corrcoef(feature2, houses_numeric)[0, 1]

    print(f"Corrélation de Astronomy avec la cible : {corr1}")
    print(f"Corrélation de Defense Against the Dark Arts avec la cible : {corr2}")

def main():
    LENGTH = 40
    totalAccuracy = 0
    for i in range(LENGTH):
        seed = random.randint(1, 100)
        filePath = parseArgs(sys.argv, len(sys.argv))
        fileDataFrame = openCsv(filePath)
        clearedDataFrame = clearDataFrame(fileDataFrame)
        trainDataFrame, (predictionSet, verificationHouses) = predictionSubSet(clearedDataFrame, seed)
        houses = trainDataFrame['Hogwarts House']
        
        # numericdataFrame = getNumericsFromDataFrame(trainDataFrame)
        # feature1 = trainDataFrame["Astronomy"]
        # feature2 = trainDataFrame["Defense Against the Dark Arts"]
        # correlation = numpy.corrcoef(feature1, feature2)[0, 1]
        # print(f"Corrélation entre Astronomy et Herbology : {correlation}")
        # analyze_features(numericdataFrame, houses)
        # exit()
        
        numericdataFrame = getNumericsFromDataFrame(trainDataFrame)
        numericdataFrame = standardizeDataFrame(numericdataFrame)
        features = numericdataFrame[FEATURES]
        ret = fit(features, houses)
        # showCost(ret)
        
        numericdataFramePredict = getNumericsFromDataFrame(predictionSet)
        numericdataFramePredict = standardizeDataFrame(numericdataFramePredict)
        featureToPredict = numericdataFramePredict[FEATURES]
        thetas = ret[0]
        predictions = predict(featureToPredict, thetas)

        accuracy = score(predictions, verificationHouses.array)
        totalAccuracy = totalAccuracy + accuracy
        print(f"Précision du modèle : {accuracy * 100:.2f}%")
        # break
    print(f"Moyenne Précision du modèle : {(totalAccuracy / LENGTH) * 100:.2f}%")

if (__name__ == "__main__"):
    pandas.set_option('display.max_columns', None)
    pandas.set_option('display.max_rows', None)
    main()