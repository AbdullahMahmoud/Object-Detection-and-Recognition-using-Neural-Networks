# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 03:01:46 2018

@author: Lenovo-PC
"""
import numpy as np
import copy
import pandas as pd
class pca :
    def fit(self,Components,x_train,epochs,eta):
        import random
        self.weights = np.full((Components , len(x_train[0,:])) , random.uniform(0,0.5))
        print(self.weights)
        PrevW = copy.deepcopy(self.weights)
        for epoch in range(epochs):
            y = x_train.dot(self.weights.T)
            x = y.T.dot(x_train)
            _w = np.full((Components , len(x_train[0,:])) , random.uniform(-1,1))
            for j in range(Components):
                for i in range(len(x_train[0,:])):
                    tmp1 = self.weights[0:j+1,i:i+1].T.dot(y[:,0:j+1].T)
                    _w[j,i] = tmp1.dot(y[:,j:j+1])
            self.weights = self.weights - eta*(x-_w)
            if (PrevW == self.weights).all():
                break;
            PrevW = copy.deepcopy(self.weights)
        print(epoch)
        import csv
        with open("PCA_Weights.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerows(self.weights)
        return self.weights
    def LoadWeights(self):
        
        data = pd.read_csv("PCA_Weights.csv" , header = None)
        self.weights = data.iloc[:,:].values
    def transform(self,x):
        return (self.weights.dot(x.T)).T
            

        