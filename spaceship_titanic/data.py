import seaborn as sns
import matplotlib
import matplotlib as plt
from sklearn.preprocessing import OneHotEncoder
matplotlib.use('TkAgg')
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import  train_test_split

def datasetloader():
    df_train = pd.read_csv("/Users/joyongjae/PROJECTS/spaceship_titanic/spaceship-titanic/train.csv")
    df_test = pd.read_csv("/Users/joyongjae/PROJECTS/spaceship_titanic/spaceship-titanic/test.csv")
    return df_train, df_test

def drop_set_id(df_train, df_test):
    df_train_drop = df_train.copy()
    df_test_drop = df_test.copy()
    df_train_drop = df_train_drop.drop(['PassengerId', 'Cabin', 'Name'], axis=1)
    #submission 파일 만들 때 사용할 passengerid 데이터
    df_test_id = df_test_drop['PassengerId']
    df_test_drop = df_test_drop.drop(['PassengerId', 'Cabin', 'Name'], axis=1)
    print(df_train_drop)
    return df_train_drop, df_test_drop, df_test_id

def Labelencoder(df_train_drop, df_test_drop):
    df_train_label = df_train_drop.copy()
    df_test_label = df_test_drop.copy()
    scaler = LabelEncoder()

    for col in df_train_label.select_dtypes(include='object'):
        df_train_label[col] = scaler.fit_transform(df_train_label[col])

    for col in df_test_label.select_dtypes(include='object'):
        df_test_label[col] = scaler.fit_transform(df_test_label[col])

    print(df_train_label)
    print(df_test_label)

    return df_train_label, df_test_label

def select_cols(df_train_label):
    corr = df_train_label.corr()
    cols = []
    for col in df_train_label.columns:
        if abs(corr[col]['Transported']) > 0.2:
            cols.append(col)
    print(cols)
    return cols

def scaling(df_train_label, df_test_label):
    df_train_label = df_train_label.fillna(df_train_label.mode().iloc[0])
    df_test_label = df_test_label.fillna(df_test_label.mode().iloc[0])
    return df_train_label, df_test_label

def data_split(df_train_label):
    x = df_train_label[['CryoSleep', 'RoomService', 'Spa', 'VRDeck']]
    y = df_train_label['Transported']
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=42)
    print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
    return train_x, test_x, train_y, test_y

def minmax_scale(df_test_label, train_x, test_x):
    df_test_scaled = df_test_label.copy()
    df_test_scaled = df_test_scaled[['CryoSleep', 'RoomService', 'Spa', 'VRDeck']]
    minmax = MinMaxScaler()
    train_x = minmax.fit_transform(train_x)
    test_x = minmax.transform(test_x)
    df_test_scaled = minmax.transform(df_test_scaled)
    return  train_x, test_x, df_test_scaled
