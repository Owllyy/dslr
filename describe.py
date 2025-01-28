import pandas as pandas
from utils import *
from maths import *

import numpy

import statistics

def processColumnStd(col: pandas.Series):
    values = {}
    clearedCol = col.dropna().to_list()
    values['count'] = len(clearedCol)
    values['mean'] = statistics.mean(clearedCol)
    values['std'] = numpy.std(clearedCol)
    values['min'] = min(clearedCol)
    quartilesValues = quartiles(clearedCol)
    values['25%'] = quartilesValues[0]
    values['50%'] = quartilesValues[1]
    values['75%'] = quartilesValues[2]
    values['max'] = max(clearedCol)
    return values

def processColumn(col: pandas.Series):
    values = {}
    clearedCol = col.dropna().to_list() # Clear NaN and return values as array
    values['count'] = count(clearedCol)
    values['mean'] = mean(clearedCol)
    values['std'] = std(clearedCol)
    values['min'] = getMinOrMax(clearedCol, LimitType.MIN)
    quartilesValues = quartiles(clearedCol)
    values['25%'] = quartilesValues[0]
    values['50%'] = quartilesValues[1]
    values['75%'] = quartilesValues[2]
    values['max'] = getMinOrMax(clearedCol, LimitType.MAX)
    return values

def cleanDataFrame(dataFrame: pandas.DataFrame):
    clearedDataFrame = dataFrame.drop(['Index'], axis='columns')
    return clearedDataFrame

def processDataframe(dataFrame: pandas.DataFrame):
    procecedColums = {}
    procecedColums2 = {}
    clearedDataFrame = cleanDataFrame(dataFrame)
    for col in clearedDataFrame.columns:
        procecedColums[col] = processColumn(clearedDataFrame[col])
        # procecedColums2[col] = processColumnStd(clearedDataFrame[col])
    print(pandas.DataFrame(procecedColums))
    # print(pandas.DataFrame(procecedColums2))
    return (pandas.DataFrame(procecedColums))

def main():
    dataFrame = openCsv('./datasets/dataset_train.csv')
    procecedDataFrame = processDataframe(dataFrame)
    # print(dataFrame)

if (__name__ == "__main__"):
    main()