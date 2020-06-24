import matplotlib as matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
train_df = pd.read_csv("C:/Users/Devna Chaturvedi/Desktop/Summer Semester - Python/Python lesson 5/Python_Lesson5/train.csv")
plt.scatter(train_df.GarageArea, train_df.SalePrice)
plt.show()
train_df['GarageArea'].describe()

print(train_df['GarageArea'].describe())

plt.boxplot(x = train_df.GarageArea)
plt.show()

train_df.shape

print(train_df.shape)

first_quartile = train_df['GarageArea'].quantile(.25)
third_quartile = train_df['GarageArea'].quantile(.75)
IQR = third_quartile - first_quartile

new_boundary = third_quartile + 3*IQR

train_df.drop(train_df[train_df['GarageArea']>new_boundary].index,axis = 0, inplace = True)

train_df.shape

print(train_df.shape)

plt.boxplot(x = train_df.GarageArea)
plt.show()
plt.scatter(train_df.GarageArea, train_df.SalePrice)
plt.show()

