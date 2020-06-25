# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 22:38:16 2019

@author: Nagul
"""

#import packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#import dataset
data = pd.read_csv("creditcard.csv")

#explore dataset
print(data.columns)
print(data.shape)
print(data.describe())


data = data.sample(frac = 0.1, random_state = 1)
print(data.shape)
data.hist(figsize=(20,20))
plt.show()

#Determine the no. of fraud cases
Fraud = data[data['Class'] == 1]
Valid = data[data['Class'] == 0]

outlier_fraction = len(Fraud)/float(len(Valid))
print(outlier_fraction)

print("Fraud cases : {}".format(len(Fraud)))
print("Valid cases : {}".format(len(Valid)))

#Correlation matrix
corrmat = data.corr()
fig = plt.figure(figsize = (12,9))
sns.heatmap(corrmat, vmax = 0.8, square = True)
plt.show()

#Get all the columns from dataframe
columns = data.columns.tolist()

#Filter the columns to remove the data we dont want
columns = [c for c in columns if c not in ["Class"]]

#Store the variable we will be predicting
target = "Class"

X = data[columns]
Y = data[target]

print(X.shape)
print(Y.shape)

#Import the sklearn packages
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

#random state
state = 1

#define the outlier detection methods
classifiers = {
      "Isolation Forest" : IsolationForest(max_samples = len(X),
                                           contamination = outlier_fraction,
                                           random_state = state),
      "Local Outlier factor" : LocalOutlierFactor(n_neighbors = 20,
                                                  contamination = outlier_fraction)
    }


#Fitting the model
n_outliers = len(Fraud)

for i, (clf_name, clf) in enumerate(classifiers.items()):
    
    #fitting the data and tag outliers
    if clf_name == "Local Outlier factor":
        y_pred = clf.fit_predict(X)
        scores_pred = clf.negative_outlier_factor_
    else:
        clf.fit(X)
        scores_pred = clf.decision_function(X)
        y_pred = clf.predict(X)
        
    #Reshape the predictions 
    y_pred[y_pred==1] = 0
    y_pred[y_pred==-1] = 1  
    
    n_errors = (y_pred != Y).sum()
    
    #Run classification matrices
    print('{} : {}'.format(clf_name, n_errors))
    print(accuracy_score(Y, y_pred))
    print(classification_report(Y, y_pred))    

