#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 22:28:43 2019

@author: ChihChi
"""

import math


import numpy as np
import pandas as pd
import seaborn as sns



import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load Data
train_df = pd.read_csv('/Users/ChihChi/Desktop/titanic/train.csv')
test_df = pd.read_csv('/Users/ChihChi/Desktop/titanic/test.csv')

# information
summary_df = train_df.describe()

# 觀察有多少缺失值
train_df_copy = train_df.copy()
train_df_copy.isnull().sum(axis=0)

# 另外一種方法
sns.heatmap(train_df_copy.isnull(),yticklabels=False,cbar=False,cmap='viridis')


"""
發現Age有177個缺失, Cabin有687個缺失, Embarked有兩個NA值
打算刪除Cabin欄位, Age補上中位數或其餘的矢量單位
"""

# 觀察數據



# 刪除無意義的名字與過多的Na值
train_df_copy = train_df_copy.drop(['Cabin'], axis=1)
train_df_copy = train_df_copy.drop(['Name'], axis=1)
train_df_copy = train_df_copy.drop(['PassengerId'], axis=1)
train_df_copy = train_df_copy.drop(['Ticket'], axis=1)


# 圖形查看缺失值
sns.heatmap(train_df_copy.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# 單欄查看
sns.countplot(x='Pclass', data=train_df_copy, palette='RdBu_r')
sns.countplot(x='Survived', data=train_df_copy, palette='RdBu_r')
sns.countplot(x='Age', data=train_df_copy, palette='RdBu_r')
sns.countplot(x='SibSp', data=train_df_copy, palette='RdBu_r')




# 雙欄交叉查看(看不出差異放上面, 看得初差異的放下面)
sns.countplot(x='Survived', hue='SibSp', data=train_df_copy, palette='RdBu_r')           

# 想要變成0, 1變數
sns.countplot(x='Survived', hue='Parch', data=train_df_copy, palette='RdBu_r')           


# Label有肉眼明顯差異
sns.countplot(x='Survived', hue='Age', data=train_df_copy, palette='RdBu_r')           

sns.countplot(x='Survived', hue='Pclass', data=train_df_copy, palette='RdBu_r')           
sns.countplot(x='Survived', hue='Sex', data=train_df_copy, palette='RdBu_r')           
sns.countplot(x='Survived', hue='Fare', data=train_df_copy, palette='RdBu_r')           
sns.countplot(x='Survived', hue='Embarked', data=train_df_copy, palette='RdBu_r')           





# 處理年紀的缺失值
age_median = np.median(train_df_copy['Age'].dropna())
train_df_copy = train_df_copy.fillna(age_median)


train_df_copy.Sex[train_df_copy['Sex'] == 'male'] = 1
train_df_copy.Sex[train_df_copy['Sex'] == 'female'] = 0

dummy_pclass = pd.get_dummies(train_df_copy['Pclass'], prefix='Pclass')
dummy_parch = pd.get_dummies(train_df_copy['Parch'], prefix='Parch')
dummy_embarked = pd.get_dummies(train_df_copy['Embarked'], prefix='Em')


train_df_copy = train_df_copy.drop(['Pclass', 'Parch', 'Embarked'], axis=1)


new_train_df = pd.concat([train_df_copy, dummy_pclass], axis=1)
new_train_df = pd.concat([new_train_df, dummy_parch], axis=1)
new_train_df = pd.concat([new_train_df, dummy_embarked], axis=1)


train_df_label = new_train_df['Survived']
new_train_df = new_train_df.drop('Survived', axis=1)

# 轉成numpy
origin_label = np.array(train_df_label)
origin_data = np.array(new_train_df)


# 切驗證集
X_train, X_test, y_train, y_test = train_test_split(origin_data, origin_label, 
                                                    test_size=0.3, random_state=7777)


knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train) 
knn_model.score(X_test, y_test)



