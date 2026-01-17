import seaborn as sns
import matplotlib
import matplotlib as plt
from sklearn.preprocessing import OneHotEncoder
matplotlib.use('TkAgg')
import pandas as pd
import numpy as np

def datasetloader():
    data = pd.read_csv("/Users/joyongjae/PROJECTS/spaceship_titanic/spaceship-titanic/train.csv")
    return data

def EDA(data):
    print("="*10)
    print("info")
    print("="*10)
    print(data.info())
    print("=" * 10)
    print("head(5)")
    print("=" * 10)
    print(data.head())
    print("=" * 10)
    print("describe")
    print("=" * 10)
    print(data.describe(include='all'))

def one_hot(data):
    data_one = data.copy()
    data_one = pd.get_dummies(data_one)
    print("=============OneHotEncoding=============")
    print(data_one)
    return data_one


def data_corr_cols(data_select):
    cols=[]
    for col, val in data_select.corr()['Transported'].items():
        if abs(val) > 0.2:
            cols.append(col)
    print(cols)
    return cols

def data_drop(data):
    data_second = data.copy()
    data_second = data_second.drop(['PassengerId','Cabin','Name'],axis = 1)
    return data_second

