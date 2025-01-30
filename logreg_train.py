import pandas as pandas
import matplotlib.pyplot as plt
from utils import *
from maths import *
import numpy as numpy

HOUSE_LABEL = "Hogwarts House"

def main():
    fileDataFrame = openCsv('./datasets/dataset_train.csv')
    dataFrameSplit = [fileDataFrame.iloc[:-100], fileDataFrame.iloc[-100:]]
    dataFrameToCheckWithHouses = dataFrameSplit[1]
    dataFrameToCheckWithoutHouses = dataFrameSplit[1].drop([HOUSE_LABEL], axis='columns')
    trainDataFrame = dataFrameSplit[0]

if (__name__ == "__main__"):
    pandas.set_option('display.max_columns', None)
    pandas.set_option('display.max_rows', None)
    main()