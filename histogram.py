import pandas as pandas
import matplotlib.pyplot as plt
from utils import *
from maths import *

def clearDataframe(dataFrame):
    clearedDataframe = dataFrame.dropna()
    return clearedDataframe.drop(['Index'], axis='columns')

def main():
    dataframe = pandas.read_csv('./datasets/dataset_train.csv')
    dataframe = clearDataframe(dataframe)
    colors = ["red", "green", "blue", "purple"]
    labels = ["Gryffindor", "Slytherin", "Hufflepuff", "Ravenclaw"]
    plt.ylabel("frenquency")
    (a,b,c,d) = splitHouse(dataframe)

    plt.hist(x=[a["Arithmancy"], b["Arithmancy"], c["Arithmancy"],d["Arithmancy"]], color=colors, label=labels, histtype="barstacked")
    plt.legend(title='Maisons')
    plt.show()

if (__name__ == "__main__"):
    main()