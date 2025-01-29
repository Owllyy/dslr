import pandas as pandas
import matplotlib.pyplot as plt
from utils import *
from maths import *
import numpy as np

HOUSE = "Hogwarts House"
def main():
    fileDataFrame = pandas.read_csv('./datasets/dataset_train.csv')
    dataFrame = fileDataFrame.drop(['Index'], axis='columns')
    dataFrame = dataFrame.dropna()
    if HOUSE not in dataFrame.columns:
        raise ValueError("La colonne 'House' n'existe pas dans le DataFrame.")
    numeric_df = dataFrame.select_dtypes(include=['number'])
    numeric_df = standardizeDataFrame(numeric_df)
    numeric_df[HOUSE] = dataFrame[HOUSE]
    houses = numeric_df[HOUSE].unique()
    colors = {house: color for house, color in zip(houses, ['red', 'blue', 'purple', 'green'])}
    pandas.plotting.scatter_matrix(
        numeric_df.drop(HOUSE, axis=1),
        figsize=(15, 15),
        marker='o',
        hist_kwds={'bins': 20},
        s=1,
        alpha=0.8,
        c=numeric_df[HOUSE].map(colors),
    )    
    plt.show()

if (__name__ == "__main__"):
    pandas.set_option('display.max_columns', None)
    pandas.set_option('display.max_rows', None)
    main()