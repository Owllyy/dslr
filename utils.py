import pandas as pandas

def openCsv(path: str):
    fileDataFrame = pandas.read_csv(path)
    dataFrame = fileDataFrame.select_dtypes(include=['number'])
    return dataFrame
