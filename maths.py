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
        costs += cost(prediction[i], )