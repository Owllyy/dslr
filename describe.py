import pandas as pandas
from utils import *
from maths import *
import sys

def processColumn(col: pandas.Series):
    values = {}

    clearedCol = col.to_list() # Return values as array
    clearedCol.sort() # Sort is needed by quartiles and min/max

    values['count'] = count(clearedCol)
    values['mean'] = ft_mean(clearedCol)
    values['std'] = standardDeviation(clearedCol)
    values['min'] = getMinOrMax(clearedCol, LimitType.MIN)
    quartilesValues = quartiles(clearedCol)
    values['25%'] = quartilesValues[0]
    values['50%'] = quartilesValues[1]
    values['75%'] = quartilesValues[2]
    values['max'] = getMinOrMax(clearedCol, LimitType.MAX)
    values['bonus_variance'] = variance(clearedCol)
    values['bonus_skewness'] = skewness(clearedCol)

    return values

def processDataframe(dataFrame: pandas.DataFrame):
    procecedColums = {}
    clearedDataFrame = clearDataFrame(dataFrame) # Remove index col + NaN

    for col in clearedDataFrame.columns:
        procecedColums[col] = processColumn(clearedDataFrame[col])

    return (pandas.DataFrame(procecedColums))

def main():
    filePath = parseArgs(sys.argv, len(sys.argv))
    fileDataFrame = openCsv(filePath)
    dataFrame = getNumericsFromDataFrame(fileDataFrame)
    procecedDataFrame = processDataframe(dataFrame)
    print(procecedDataFrame.to_string())

if (__name__ == "__main__"):
    pandas.set_option('display.max_columns', None)
    pandas.set_option('display.max_rows', None)
    main()