#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 22:28:43 2019

@author: ChihChi
"""


import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt


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


# 圖形查看缺失值
sns.heatmap(train_df_copy.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# 單欄查看
sns.countplot(x='Survived', data=train_df_copy, palette='RdBu_r')
sns.countplot(x='Age', data=train_df_copy, palette='RdBu_r')
sns.countplot(x='SibSp', data=train_df_copy, palette='RdBu_r')


# 雙欄交叉查看
sns.countplot(x='Survived', hue='Pclass', data=train_df_copy, palette='RdBu_r')           









