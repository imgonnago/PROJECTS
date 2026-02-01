import numpy as np
import pandas as pd
import matplotlib as plt
import matplotlib
matplotlib.use('TkAgg')
import seaborn as sns

data = pd.read_csv('/Users/joyongjae/PROJECTS/age-related-conditions/icr-identify-age-related-conditions/train.csv')

print(f'{"=" * 10} data info {"="*10}')
print(data.info())
print(f'{"=" * 10} data describe {"="*10}')
print(data.describe())
print(f'{"=" * 10} data head {"="*10}')
print(data.head())

for col in data.columns:
    if data[col].isnull().sum() > 0:
        num = data[col].isnull().sum()
        print(f'{col} : {num}')

for col in data.columns:
    sns.scatterplot(data=data,x=col, y=col)
