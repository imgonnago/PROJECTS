from data import  datasetloader
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')
import matplotlib as plt
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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

def check(model, test_x, test_y):
    auto_pred = model.predict(test_x)
    auto_acc = accuracy_score(test_y, auto_pred)
    auto_mat = confusion_matrix(test_y, auto_pred)
    auto_rep = classification_report(test_y, auto_pred)
    print(f'===== acc =====')
    print(auto_acc)
    print(f'=====matrix=====')
    print(auto_mat)
    print(f'=====report=====')
    print(auto_rep)

def submission(model, df_test_scaled, df_test_id):
    Transported = model.predict(df_test_scaled)
    PassengerId = df_test_id
    Submission = pd.DataFrame({ 'PassengerId' : PassengerId , 'Transported' : Transported})

    print(Submission)
    submission = Submission.to_csv('submission.csv', index=False)
    return submission