from pandas.plotting import parallel_coordinates
import matplotlib.pyplot as plt
import pandas as pandas
from utils import *
from maths import *
import sys

def main():
    filePath = parseArgs(sys.argv, len(sys.argv))
    fileDataFrame = openCsv(filePath)
    houses = fileDataFrame['Hogwarts House']
    clearedDataFrame = clearDataFrame(fileDataFrame)
    numericdataFrame = getNumericsFromDataFrame(clearedDataFrame)
    numericdataFrame = standardizeDataFrame(numericdataFrame)
    numericdataFrame['Hogwarts House'] = houses
    plt.figure(figsize=(12, 6))
    parallel_coordinates(numericdataFrame[['Hogwarts House'] + FEATURES], 'Hogwarts House', colormap='viridis')
    plt.title('Coordonnées parallèles des notes par Maison')
    plt.show()
   

if (__name__ == "__main__"):
    pandas.set_option('display.max_columns', None)
    pandas.set_option('display.max_rows', None)
    main()