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

def ft_mean(data: list[float]):
    sum = 0
    lenght = 0
    for line in data:
        sum += line
        lenght += 1
    return sum / lenght

def variance(data: list[float]) -> float:
    mean: float = ft_mean(data)
    squared_diffs: list[float] = []

    for x in data:
        squared_diffs.append((x - mean) ** 2)

    return ft_mean(squared_diffs)

def skewness(data: list[float]) -> float:
    mean: float = ft_mean(data)
    std_dev: float = standardDeviation(data)
    lenght = len(data)
    
    skew: float = 0.
    for x in data:
        skew += ((x - mean) / std_dev) ** 3
    
    correction_factor: float = lenght / ((lenght - 1) * (lenght - 2))

    return skew * correction_factor

def quartiles(collumn: pandas.DataFrame):
    length = count(collumn)
    quartile_size = length // 4
    return (collumn[quartile_size], collumn[length // 2], collumn[length // 2 + quartile_size])

def getMinOrMax(values: list[int], sort: LimitType): 
    return values[0] if sort == LimitType.MIN else values[-1]

def standardDeviation(values: list[int]) -> float :
    meanValues = ft_mean(values)
    somme = 0 
    for x in values:
        somme += (x - meanValues)**2
    variance = somme / count(values)
    ret = math.sqrt(variance)
    return ret

def sigmoid(x) -> float :
    return 1 / (1 + numpy.exp(-x))
    
def costFunction(prediction: float, y: bool) -> float:
    if (y):
        return -math.log(prediction)
    else:
        return -math.log(1 - prediction)

def costMean(prediction: list[float], houses: list[int]) -> float:
    length = min(len(prediction), len(houses))
    costs = 0.0
    for i in range(0, length):
        costs += costFunction(prediction[i], houses[i])
    return costs / length

LEARNING_RATE = 0.001
# LEARNING_RATE = 0.1
def gradient_descent(data: pandas.DataFrame, prediction: list[float], verity: list[float], theta: list[float]) -> list[float]:
    data_len = len(prediction)

    prediction_validity: list[float] = prediction - verity

    gradient_value = numpy.dot(data.T, prediction_validity) / data_len
    theta -= gradient_value * LEARNING_RATE
    
    return theta

ITTERATION_NUMBER = 2000
# Return theta  and the cost for all houses
def fit(data, houses) -> tuple[list[float], float]:
    if numpy.shape(data)[0] != len(houses):
        raise Exception("Error: Unmatching data size. data size = ", numpy.shape(data)[1], " = houses len = ", len(houses), len(data))

    data = numpy.insert(data, 0, 1, axis=1) # intercept value
    data_lenght = numpy.shape(data)[1] # Taille de la col maison
    thetas = []
    costs = []

    for house in numpy.unique(houses):
        real: list[float] = numpy.where(houses == house, 1, 0)  
        theta: list[float] = numpy.zeros(data_lenght)
        cost: list[float] = []
        for _ in range(ITTERATION_NUMBER):
            weight: list[float] = data.dot(theta)
            predicted: list[float] = sigmoid(weight)
            theta = gradient_descent(data, predicted, real, theta)
            cost.append(costMean(predicted, real))
        thetas.append((theta, house))
        costs.append((cost, house))
    return (thetas, costs)


BATCH_SIZE = 32
EPOCH_NUMBER = 1000
# Return theta  and the cost for all houses
def fitMiniBatch(data, houses) -> tuple[list[float], float]:
    if numpy.shape(data)[0] != len(houses):
        raise Exception("Error: Unmatching data size. data size = ", numpy.shape(data)[1], " = houses len = ", len(houses), len(data))

    data = numpy.insert(data, 0, 1, axis=1) # intercept value
    n_samples, n_features = data.shape
    thetas = []
    costs = []

    for house in numpy.unique(houses):
        real: list[float] = numpy.where(houses == house, 1, 0)  
        theta: list[float] = numpy.zeros(n_features)
        cost: list[float] = []
        for epoch in range(EPOCH_NUMBER):
            indices = numpy.random.permutation(n_samples) # Melange les indices du tableau, donc l'ordre dans lequel on va le parcourir pour chaque epoch
            for start_idx in range(0, n_samples, BATCH_SIZE):
                batch_indices = indices[start_idx:start_idx + BATCH_SIZE] # Selectionne BATCH_SIZE index
                batch_data = data[batch_indices]
                batch_real = real[batch_indices]
                
                weight: list[float] = batch_data.dot(theta)
                predicted: list[float] = sigmoid(weight)
                theta = gradient_descent(batch_data, predicted, batch_real, theta)
                cost.append(costMean(predicted, batch_real))
        thetas.append((theta, house))
        costs.append((cost, house))
    return (thetas, costs)
