from data import  datasetloader, one_hot
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')
import matplotlib as plt
import pandas as pd
import numpy as np

data = datasetloader()
data_one = one_hot(data)

sns.heatmap(data = data_select.corr(),annot=True)
sns.boxplot(data = data, x = 'Transported', y = 'Age')
sns.barplot(data = data, x = 'Transported', y = 'VIP')