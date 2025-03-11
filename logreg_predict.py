import pandas as pandas
import matplotlib.pyplot as plt
from utils import *
from maths import *
import numpy as numpy

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

#TODO: clean
def main():
    fileData = openCsv('./datasets/dataset_test.csv') #TODO: Ajouter le path du fichier en parametre du programme (Surement faire une gestion d'erreur)
    
    clearedDataFrame = clearDataFrame(fileData)
    trainDataFrame, (predictionSet, verificationHouses) = predictionSubSet(clearedDataFrame)
    numericdataFrame = getNumericsFromDataFrame(predictionSet)
    numericdataFrame = standardizeDataFrame(numericdataFrame)
    
    thetas = readTheta()
    features = numericdataFrame[FEATURES]
    predictions = predict(features, thetas)
    
    # TODO: Creer un fichier csv avec les maison (format sur le sujet)
    
    # Cette partie est pour debug
    i = 0
    for prediction in predictions:
        if (prediction != verificationHouses.array[i]):
            print("Not same value: Our prediction = ", prediction, " - Right house = ", verificationHouses.array[i])
        i = i + 1
    
    accuracy = score(predictions, verificationHouses.array)
    print(f"Précision du modèle : {accuracy * 100:.2f}%")
    # --------------------------------------------------------------------
    
if (__name__ == "__main__"):
    pandas.set_option('display.max_columns', None)
    pandas.set_option('display.max_rows', None)
    main()