import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math as m
import pandas as pd
import pprint
class RBF:
    def __init__(self):
        self.centroids=[]
        self.KmeanThreshold=.0001
        self.max_iterations=500
    def initialize_Centroid(self,k,x_data):
        for i in range(k):
            self.centroids.append(x_data[i])
        return self.centroids
    def sigmoid(self,x):
            return 1 / (1 + np.exp(-x))
         
     
    def Euclidean_distance(self,first, second):
        squDist = 0
        for i in range(len(first)):
                squDist += (first[i] - second[i])*(first[i] - second[i])
        euclDist = m.sqrt(squDist)
        return euclDist
     
    def kmean(self,k,x_data):
        self.centroids=self.initialize_Centroid(k,x_data)
        for i in range(self.max_iterations):
            clusters = {}
            for j in range(k):
                clusters[j] = []
            for features in x_data:
                distances = [self.Euclidean_distance(features ,self.centroids[centroid])  for centroid in range(len(self.centroids) ) ]
                classification = distances.index(min(distances))
                clusters[classification].append(features)
            previous = self.centroids
            for classification in clusters:
                self.centroids[classification] = np.average(clusters[classification], axis = 0)
            converge=True
            for centroid in range(len(self.centroids)):
                previous_centroid = previous[centroid]
                curr_centroid = self.centroids[centroid]
                if self.Euclidean_distance(curr_centroid , previous_centroid) < self.KmeanThreshold:
                    converge = False
            if converge:        	
                break
        return self.centroids
     
    
     
    def sigmaSpread(self,centroids,k):
        distance=[]
        for i in range(len(centroids)):
            for j in range(len(centroids)):
                if i == j:
                    continue
                else: 
                    distance.append(self.Euclidean_distance(centroids[i],centroids[j]))
     
        self.sigma=distance[distance.index(max(distance))]/m.sqrt(2*k)
        return self.sigma
     
     
    def compute_Gaussian_fun(self,feature,centroids,sigma):
        gaussian=[]
        for i in  range(len(centroids)):
            r=self.Euclidean_distance(feature,centroids[i])
            gaussian.append(m.exp(-(r**2/(2*sigma**2))))
     
        return gaussian
     
    def update_weights(self,weights,eta,error,gaussian,classes):
     
        for k in range(len(classes)):
            weights[k]=weights[k]-(eta*error*gaussian[k])
        return weights
     
     
    def MSE(self,y_act , y_pred,classes):
        error=0
        for j in range(len(y_act)):
            for i in range(classes):
                error+=(y_act[j,i] - y_pred[j,i])**2
        #print(error)
        return error/(2*len(y_act)) 
     
    
     
    def GradientDescent(self,Y,Ypred,Alpha,Xs,Cs,classes):
        NewCs = np.full((len(Xs[0,:]),classes),0)
        Subt = Y - Ypred
        
        dcs = Xs.T.dot(Subt)
        NewCs = Cs + (Alpha* dcs)
        return NewCs
     
    def TrainTheModel_rbf(self,Neurons_Entry,LearningRate_Entry,Mse_threshold,epochs_Entry,classes,x,y_train,x_test,y_test):
        self.Num_of_classes=int(classes)
        self.Num_epochs=int(epochs_Entry)
     
        self.eta=float(LearningRate_Entry)
     
        self.NumOfNeurons=int(Neurons_Entry)
     
        self.MseThreshold=float(Mse_threshold)
        self.centroids=self.kmean(self.NumOfNeurons,x)
        self.sigma = self.sigmaSpread(self.centroids,self.NumOfNeurons)
        x_train= np.full((len(x) , self.NumOfNeurons) , 0.0)
        for i in range(len(x)):
            x_train[i] = self.compute_Gaussian_fun(x[i],self.centroids,self.sigma)
            
        import random
        self.weights = np.full((self.NumOfNeurons , self.Num_of_classes) , random.uniform(0,1))
        '''
        for i in range(self.Num_of_classes):
            self.weights[:,i:i+1] = np.full((self.NumOfNeurons,1),random.uniform(0,1))
        '''
        prev=-1e10
        xs=[]
        ys=[]
        for i in range(self.Num_epochs):
            #print(i)
            xs.append(i)
            pred=x_train.dot(self.weights)
            #pred=sigmoid(pred)
           # pprint.pprint(pred)
            #pprint.pprint(y_train)
            c=0
        
            for j in range(len(pred)):
                s=0
                m=-1e10
                for k in range(len(pred[j])):
                    if pred[j][k]>m:
                        m=pred[j][k]
                        s=k
    
                #print(s)    
                if y_train[j,s]==1:
                    c=c+1
            print("train : ",c/len(x_train))    
           
            mse=self.MSE(pred,y_train,self.Num_of_classes)
            ys.append(mse)
            #error=y_train-pred
           # weights=update_weights(weights,eta,error,x_train,Num_of_classes)
            self.weights=self.GradientDescent(y_train,pred,self.eta,x_train,self.weights,self.Num_of_classes)
            #pprint.pprint(weights)
            a=self.Test(x_test,y_test)
            if a>=prev:
                prev = a
                prev_weights=self.weights 
            else:
                break
    #        print(mse)
            if mse < self.MseThreshold:
                break;
        fig = plt.figure()
        ax1 = fig.add_subplot(1,1,1)         
        ax1.clear()        
        ax1.plot(xs,ys)  
        plt.show()   
        import csv
        with open("RBF_Weights.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerows(prev_weights)
        return prev
    def Test (self,x_t,y_test):
        x_test= np.full((len(x_t) , self.NumOfNeurons) , 0.0)
        for i in range(len(x_t)):
            x_test[i] = self.compute_Gaussian_fun(x_t[i],self.centroids,self.sigma)
        pred=x_test.dot(self.weights)
        #pred=sigmoid(pred)
        c=0
        confusion_matrix=np.full((len(pred[0]),len(pred[0])),0.0)
        confusion_matrix=np.full((len(y_test[0,:]),len(y_test[0,:])),0)
        for j in range(len(pred)):
            s=0
            m=-1e10
            for k in range(len(pred[j])):
                if pred[j][k]>m:
                    m=pred[j][k]
                    s=k
    
                #print(s) 
            e=0   
            if y_test[j,s]==1:
                c=c+1
            for k in range(len(pred[j])):    
                if y_test[j,k]==1:
                        e=k
            confusion_matrix[e,s]=confusion_matrix[e,s]+1     
        pprint.pprint(confusion_matrix)
        print("test : ",c/len(x_test))    
      
        return c/len(x_test)
    
    def Classify (self,x_t,classes):
        x_test= np.full((len(x_t) , self.NumOfNeurons) , 0.0)
        for i in range(len(x_t)):
            x_test[i] = self.compute_Gaussian_fun(x_t[i],self.centroids,self.sigma)
        pred=x_test.dot(self.weights)
        
        p=[]
        for k in range(len(pred)):
            s=0
            m=-1e10
            for j in range(len(pred[k])):
                if pred[k][j]>m:
                    m=pred[k][j]
                    s=j
            p.append(classes[s])
        #pred=sigmoid(pred)
      
           
        print(p)
        return p
    def LoadWeights(self,Neurons_Entry,LearningRate_Entry,Mse_threshold,epochs_Entry,classes,x,y_train,x_test,y_test):
        self.Num_of_classes=int(classes)
        self.Num_epochs=int(epochs_Entry)
     
        self.eta=float(LearningRate_Entry)
     
        self.NumOfNeurons=int(Neurons_Entry)
     
        self.MseThreshold=float(Mse_threshold)
        self.centroids=self.kmean(self.NumOfNeurons,x)
        self.sigma = self.sigmaSpread(self.centroids,self.NumOfNeurons)
        
        data = pd.read_csv("RBF_Weights.csv" , header = None)
        self.weights = data.iloc[:,:].values
        
        self.Test(x_test,y_test)