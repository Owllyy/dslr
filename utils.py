import pandas as pandas
from maths import *
import json

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

def clearDataFrame(dataFrame: pandas.DataFrame) -> pandas.DataFrame:
    clearedDataFrame = dataFrame.drop(['Index'], axis='columns')
    clearedDataFrame = clearedDataFrame.dropna()
    return clearedDataFrame

def getNumericsFromDataFrame(dataFrame: pandas.DataFrame) -> pandas.DataFrame:
    numericDataFrame = dataFrame.select_dtypes(include=['number'])
    return numericDataFrame

def saveThetas(data): #TODO: Error
    data_json = [(arr.tolist(), house) for arr, house in data]
    with open("data.json", "w") as f:
        json.dump(data_json, f)

def readTheta() -> tuple[numpy.ndarray, str]: #TODO: Error
    with open("data.json", "r") as f:
        loaded_data_json = json.load(f)

    return [(numpy.array(arr), house) for arr, house in loaded_data_json]

    
