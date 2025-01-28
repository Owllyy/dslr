import pandas as pandas
from utils import *
from maths import *

def main():
    dataFrame = openCsv('./datasets/dataset_train.csv')
    print(dataFrame)

if (__name__ == "__main__"):
    main()