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

def main():
    # filePath = parseArgs(sys.argv, len(sys.argv))
    # fileDataFrame = openCsv(filePath)
    # clearedDataFrame = clearDataFrame(fileDataFrame)
    # trainDataFrame, (predictionSet, verificationHouses) = predictionSubSet(clearedDataFrame)
    
    # houses = trainDataFrame['Hogwarts House']
    # numericdataFrame = getNumericsFromDataFrame(trainDataFrame)
    # numericdataFrame = standardizeDataFrame(numericdataFrame)
    # features = numericdataFrame[FEATURES]
    # ret = fitMiniBatch(features, houses)
    # showCost(ret)
    # saveThetas(ret[0])
    
    LENGTH = 60
    totalAccuracy = 0
    for i in range(LENGTH):
        seed = random.randint(1, 100)
        filePath = parseArgs(sys.argv, len(sys.argv))
        fileDataFrame = openCsv(filePath)
        clearedDataFrame = clearDataFrame(fileDataFrame)
        trainDataFrame, (predictionSet, verificationHouses) = predictionSubSet(clearedDataFrame, seed)
        houses = trainDataFrame['Hogwarts House']
        
        numericdataFrame = getNumericsFromDataFrame(trainDataFrame)
        numericdataFrame = standardizeDataFrame(numericdataFrame)
        features = numericdataFrame[FEATURES]
        ret = fitMiniBatch(features, houses)
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