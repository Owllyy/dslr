import pandas as pandas
import matplotlib.pyplot as plt
from utils import *
from maths import *
import sys
import seaborn as seaborn

def main():
    filePath = parseArgs(sys.argv, len(sys.argv))
    dataframe = pandas.read_csv(filePath)
    numericDataframe = getNumericsFromDataFrame(dataframe)
    numericDataframe = clearDataFrame(numericDataframe, True)
    correlation_matrix = numericDataframe.corr()
    
    plt.figure(figsize=(12, 10))
    seaborn.heatmap(correlation_matrix,
                annot=True,
                cmap='coolwarm',
                center=0,
                square=True,
                fmt='.2f',
                linewidths=0.5)

    plt.title('Heatmap des Corrélations', fontsize=16)
    plt.tight_layout()
    plt.show()

if (__name__ == "__main__"):
    main()