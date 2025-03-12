import pandas as pandas
from maths import *
import json
import os
import os.path
import csv
import random
from sklearn.model_selection import train_test_split

FEATURES = [
    # 'Arithmancy',
    'Astronomy',
    'Herbology',
    'Defense Against the Dark Arts',
    # 'Divination',
    # 'Muggle Studies',
    'Ancient Runes',
    # 'History of Magic', 
    # 'Transfiguration',
    # 'Potions',
    # 'Care of Magical Creatures',
    'Charms',
    # 'Flying'
]

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
    numeric_columns = clearedDataFrame.select_dtypes(include=['int64', 'float64']).columns
    clearedDataFrame[numeric_columns] = clearedDataFrame[numeric_columns].fillna(clearedDataFrame[numeric_columns].mean())
    return clearedDataFrame

def getNumericsFromDataFrame(dataFrame: pandas.DataFrame) -> pandas.DataFrame:
    numericDataFrame = dataFrame.select_dtypes(include=['number'])
    return numericDataFrame

def saveThetas(data): #TODO: Error
    data_json = [(arr.tolist(), house) for arr, house in data]
    with open("data.json", "w") as f:
        json.dump(data_json, f)
        
def saveHouses(houses):
    housesLen = len(houses)
    data = {
        'Index': list(range(housesLen)),
        'Hogwarts House': houses
    }
    df = pandas.DataFrame(data)
    df.to_csv('houses.csv', index=False)

def readTheta() -> tuple[numpy.ndarray, str]: #TODO: Error
    with open("data.json", "r") as f:
        loaded_data_json = json.load(f)

    return [(numpy.array(arr), house) for arr, house in loaded_data_json]

def score(predictedHouses, verificationHouses):
    score = sum(predictedHouses == verificationHouses) / len(verificationHouses)
    return score

HOUSE_LABEL = "Hogwarts House"
def predictionSubSet(dataFrame: pandas.DataFrame) -> tuple[pandas.DataFrame, tuple[pandas.DataFrame, pandas.Series]]:
    trainDataFrame, predictionSet = train_test_split(dataFrame, test_size=400, random_state=97)
    # dataFrameSplit = [train_df, test_df]
    # dataFrameSplit = [dataFrame.iloc[:-150], dataFrame.iloc[-150:]]
    # predictionSet = dataFrameSplit[1]
    houses = predictionSet['Hogwarts House']
    features =  predictionSet.drop([HOUSE_LABEL], axis='columns')
    # trainDataFrame = dataFrameSplit[0]
    # print((trainDataFrame, (features, houses)))
    return (trainDataFrame, (features, houses))

def exitWithMessage(message: str):
    print(message)
    exit()

def parseArgs(argv, argc) -> str:
    if (argc < 2):
        exitWithMessage("A dataset file is needed")
    filePath = argv[1]
    split_tup = os.path.splitext(filePath)
    file_extension = split_tup[len(split_tup) - 1]
    if (file_extension != '.csv'):
        exitWithMessage("Bad extension file")
    if not os.path.isfile(filePath):
        exitWithMessage("File does not exist")
    return filePath