import pandas as pandas
from maths import *

def splitHouse(dataFrame: pandas.DataFrame):
    houseRavenclaw = dataFrame[dataFrame["Hogwarts House"] == "Ravenclaw"]
    houseSlytherin = dataFrame[dataFrame["Hogwarts House"] == "Slytherin"]
    houseGryffindor = dataFrame[dataFrame["Hogwarts House"] == "Gryffindor"]
    houseHufflepuff = dataFrame[dataFrame["Hogwarts House"] == "Hufflepuff"]
    return (houseRavenclaw, houseSlytherin,houseGryffindor, houseHufflepuff)

def openCsv(path: str) -> pandas.DataFrame:
    fileDataFrame = pandas.read_csv(path)
    return fileDataFrame

def standardizeDataFrame(dataFrame: pandas.DataFrame) -> pandas.DataFrame:
    newDataFrame = {}
    for col in dataFrame:
        meanValue = mean(dataFrame[col])
        stdValue = std(dataFrame[col])
        standardiweCol = (dataFrame[col] - meanValue) / stdValue
        newDataFrame.update({col: standardiweCol})
    return pandas.DataFrame(newDataFrame)

def cleanDataFrame(dataFrame: pandas.DataFrame) -> pandas.DataFrame:
    cleanedDataFrame = dataFrame.drop(['Index'], axis='columns')
    cleanedDataFrame = cleanedDataFrame.dropna()
    return cleanedDataFrame

def getNumericsFromDataFrame(dataFrame: pandas.DataFrame) -> pandas.DataFrame:
    numericDataFrame = dataFrame.select_dtypes(include=['number'])
    return numericDataFrame