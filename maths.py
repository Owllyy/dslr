from enum import Enum
import math
import numpy
import pandas

class LimitType(Enum):
    MIN = "min"
    MAX = "max"

def count(array):
    i = 0
    for _ in array:
        i += 1
    return i

def mean(col):
    sum = 0
    lenght = 0
    for line in col:
        sum += line
        lenght += 1
    return sum / lenght

def quartiles(collumn: pandas.DataFrame):
    length = count(collumn)
    quartile_size = length // 4
    return (collumn[quartile_size], collumn[length // 2], collumn[length // 2 + quartile_size])

def getMinOrMax(values: list[int], sort: LimitType): 
    return values[0] if sort == LimitType.MIN else values[-1]

def std(values: list[int]) -> float :
    meanValues = mean(values)
    somme = 0 
    for x in values:
        somme += (x - meanValues)**2
    variance = somme / count(values)
    ret = math.sqrt(variance)
    return ret

def sigmoid(x: float) -> float :
    return 1 / (1 + numpy.exp(-x))
    
def cost(prediction: float, y: bool) -> float:
    if (y):
        return -math.log(prediction)
    else:
        return -math.log(1 - prediction)

def costMean(prediction: list[float], houses: list[int]) -> float:
    len = min(len(prediction), len(houses))
    costs = 0.0
    for i in range(0, len):
        costs += cost(prediction[i], houses[i])
    return costs / len


LEARNING_RATE = 0.01
def gradient_descent(data: pandas.DataFrame, prediction: list[float], verity: list[float], theta: list[float]) -> list[float]:
    data_len = len(prediction)

    prediction_validity: list[float] = prediction - verity

    gradient_value = numpy.dot(data.T, prediction_validity) / data_len
    theta -= gradient_value * LEARNING_RATE
    
    return theta

Probability = float

ITTERATION_NUMBER = 100
# Return theta  and the cost for all houses
def fit(data, houses) -> tuple[list[float], float]:
    if numpy.shape(data)[1] != len(houses):
        raise Exception("Error: Unmatching data size")

    data = numpy.insert(data, 0, 1, axis=1) # intercept value
    data_lenght = len(numpy.shape(data)[1]) # Taille de la col maison
    thetas = []
    costs = []

    for house in numpy.unique(houses):
        real: list[Probability] = numpy.where(houses == house, 1, 0)  
        theta: list[float] = numpy.zeros(data_lenght)
        cost: list[float] = []
        for _ in range(ITTERATION_NUMBER):
            weight: list[float] = data.dot(theta)
            predicted: list[Probability] = sigmoid(weight)
            theta = gradient_descent(data, predicted, real, theta, data_lenght)
            cost.append(cost(predicted, real))
        thetas.append(theta)
        costs.append(costMean(cost, house))
    return (thetas, costs)