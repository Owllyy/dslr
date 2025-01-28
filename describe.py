import pandas as pandas
from utils import *
from maths import *

def processColumn(col: pandas.Series):
    values = {}

    clearedCol = col.dropna().to_list() # Clear NaN and return values as array
    clearedCol.sort() # Sort is needed by quartiles and min/max

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
    clearedDataFrame = cleanDataFrame(dataFrame)

    for col in clearedDataFrame.columns:
        procecedColums[col] = processColumn(clearedDataFrame[col])

    return (pandas.DataFrame(procecedColums))

def main():
    dataFrame = openCsv('./datasets/dataset_train.csv')
    procecedDataFrame = processDataframe(dataFrame)
    print(procecedDataFrame)

if (__name__ == "__main__"):
    main()