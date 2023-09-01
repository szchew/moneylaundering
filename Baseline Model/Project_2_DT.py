# -*- coding: utf-8 -*-
"""Project 2 Decision Tree Final

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1BgGE_PkL-_KKVt4YDqeNXobW2oXeXQxC
"""

## Code has some reference from CS3244 Project 1 Group 3(Previous Project's Decision Tree Code) and adapted accordingly
from google.colab import drive
drive.mount('/content/drive')

#Import all required libraries
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, average_precision_score
from sklearn import preprocessing
from sklearn import tree
from sklearn.preprocessing import OneHotEncoder
from imblearn.under_sampling import RandomUnderSampler

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import joblib

##Used to merge the 2 small transcaction files HI-Small_Trans.csv and LI-Small_Trans.csv initially

#path='/content/drive/My Drive/CS3244 Project 2/HI-Small_Trans.csv'
#path='/content/drive/My Drive/CS3244 Project 2/LI-Small_Trans.csv'
##df = pd.read_csv(path)
#df2 = pd.read_csv(path2)
#df = pd.concat([df,df2])
#del df2
#df.to_csv('merge_Small_Trans.csv')

#Read the merged files from Google Drive
path='/content/drive/My Drive/CS3244 Project 2/merge_Small_Trans.csv'
df = pd.read_csv(path)
df

#Doing a preliminary analysis of the distribution of the Receiving Currency
counts = df.loc[df['Is Laundering'] == 1,'Payment Format'].value_counts()
#plt.xticks(rotation=90)
plt.bar(counts.index, counts.values)
weights = {0 : (counts[0]/len(df))*100, 1 : (counts[1]/len(df))*100}
names = list(weights.keys())
values = list(weights.values())
plt.savefig("Distribution of Payment Format among all Money Laundering Cases.png",dpi=300)

#Last row has na input for amount paid(Only removeds 1 row)
df = df.dropna(axis=0)

#Select the columns used for DT 
df = df[["Amount Received","Receiving Currency","Amount Paid","Payment Currency","Payment Format","Is Laundering"]]

#Preprocess the 3 columns that are required to be ont hot-coded
to_encode = ["Receiving Currency","Payment Currency","Payment Format"]
lab = preprocessing.OneHotEncoder()
for i in to_encode:
  data = lab.fit_transform(df[[i]]).toarray()   
  temp = pd.DataFrame(data,columns=lab.categories_[0]).add_prefix(i+"_")
  df = pd.concat([df, temp], axis=1)
  df = df.drop(i,axis=1)


df.dtypes

#Split the dataframe into input vectors and output vectors
x, y = df.loc[:,df.columns != 'Is Laundering'], df.loc[:,'Is Laundering']
x

y

#Doing a preliminary analysis of the distribution of the dataset labels
counts = y.value_counts()
weights = {0 : (counts[0]/len(df))*100, 1 : (counts[1]/len(df))*100}

names = list(weights.keys())
values = list(weights.values())

#Plot the distribution of the labels
plt.bar(range(len(weights)), values, tick_label=names)
plt.show()
plt.savefig('Distribution of Money Laundering Labels.jpeg',format='jpeg',bbox_inches = "tight",dpi=1000)

##Train-Test Split
X_trainval, X_test, y_trainval, y_test = train_test_split(x, y, test_size=0.1, stratify=y, random_state=13)
# example of random undersampling to balance the class distribution

##Create Undersampler to deal with the lack of data in label 1 and conduct undersampling
rus = RandomUnderSampler(random_state=13)

##Resample training and test dataset respectively to prevent Data Leakage
X_trainval, y_trainval = rus.fit_resample(X_trainval, y_trainval)
X_test, y_test = rus.fit_resample(X_test, y_test)

##Create Stratified k-fold for GridSearchCV
skf = StratifiedKFold(n_splits=10, shuffle= True, random_state=13)

##Input the parameters to tune and direct into GridSearchCV
min_sam_split = range(2,20)
information_criterion = ["gini","entropy","log_loss"]
tree_d = range(1,30)
class_w = ["balanced",None]
param_g = {"min_samples_split":min_sam_split,
           "criterion":information_criterion,
           "max_depth":tree_d,
           "class_weight":class_w}
scoring = make_scorer(average_precision_score)

#Create a DecisionTreeClassfier and optimize the hyperparameters with GridSearchCV - auprc
dt = DecisionTreeClassifier(random_state=13)
dt_model_auprc = GridSearchCV(dt,param_grid=param_g, cv=skf, refit=True,scoring=scoring,verbose=1).fit(X_trainval, y_trainval)

#Print out relevant optimized hyperparameter values and its relevant statistics - auprc
'The optimal hyperparameters chosen are ' + str(dt_model_auprc.best_params_)

"The best AUPRC results over the training/validation dataset using 10-fold CV is " + str(dt_model_auprc.best_score_)

"The best AUPRC results over the test dataset is " + str(dt_model_auprc.score(X_test, y_test))

data = {'Optimal Hyperparemater':str(dt_model_auprc.best_params_), 'Best AUPRC Results over the Training/Validation dataset':dt_model_auprc.best_score_ ,'Best AUPRC Results over the Test dataset':dt_model_auprc.score(X_test, y_test)}
  
# Creates pandas DataFrame.  
result = pd.DataFrame(data, index =['1'])  
result

#Save optimized GridSearchCV model for auprc
joblib.dump(dt_model_auprc, 'dt_model_auprc.pkl')

##Plotting out decision tree - auprc
dt = DecisionTreeClassifier(max_depth = 2,min_samples_split = 2, class_weight = "balanced",criterion="gini",random_state=13)
dt.fit(X_trainval, y_trainval)
tree.plot_tree(dt,feature_names=x.columns)
plt.savefig('Decision Tree.png',format='png',bbox_inches = "tight",dpi=1000)