import pandas as pandas
from maths import *

def splitHouse(dataFrame: pandas.DataFrame):
    houseRavenclaw = dataFrame[dataFrame["Hogwarts House"] == "Ravenclaw"]
    houseSlytherin = dataFrame[dataFrame["Hogwarts House"] == "Slytherin"]
    houseGryffindor = dataFrame[dataFrame["Hogwarts House"] == "Gryffindor"]
    houseHufflepuff = dataFrame[dataFrame["Hogwarts House"] == "Hufflepuff"]
    return (houseRavenclaw, houseSlytherin,houseGryffindor, houseHufflepuff)

def openCsv(path: str):
    fileDataFrame = pandas.read_csv(path)
    dataFrame = fileDataFrame.select_dtypes(include=['number'])
    return dataFrame

def standardizeDataFrame(dataFrame: pandas.DataFrame) -> list[float]:
    newDataFrame = {}
    for col in dataFrame:
        meanValue = mean(dataFrame[col])
        stdValue = std(dataFrame[col])
        standardiweCol = (dataFrame[col] - meanValue) / stdValue
        newDataFrame.update({col: standardiweCol})
    return pandas.DataFrame(newDataFrame)