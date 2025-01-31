from enum import Enum
import math
import numpy as numpy

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

def quartiles(col):
    lenght = count(col)
    quartile_size = round(lenght / 4)
    return (col[quartile_size], col[round(lenght / 2)], col[round(lenght / 2) + quartile_size])

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

def sigmoid(x):
    if x >= 0:
        z = numpy.exp(-x)
        return 1 / (1 + z)
    else:
        z = numpy.exp(x)
        return z / (1 + z)