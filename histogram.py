import pandas as pandas
import matplotlib.pyplot as plt
from utils import *
from maths import *

def main():
    fileDataFrame = pandas.read_csv('./datasets/dataset_train.csv')
    (a,b,c,d) = splitHouse(fileDataFrame)
    print(a, b, c, d)
    # plt.hist(dataFrame)
    # plt.show()

if (__name__ == "__main__"):
    main()