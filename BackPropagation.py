import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
class MLP:
    def __init__(self):
        
        self.Activations = {}
        self.Sigmas = {}
        self.Weights = {}
        self.Bias = {}
        self.Num_of_Neurons = []
    def sigmoid(self,x):
            return 1 / (1 + np.exp(-x))
        
    def Dsigmoid(self,x):
            return x * (1 - x)
        
    def DHyper_bolic_Tangent(self,x):
            return (1-(x**2))
    
    def Hyper_bolic_Tangent(self,x):
        return (1 - np.exp(-x))/ (1 + np.exp(-x))
    
    
    def SigmaError(self, y , y_hat ):
       Errors=[]
       sum=0
       for j in range(0,5):
            if(self.Activation=='sigmoid'):
               cost =  np.float((y[0][j] - y_hat[0][j]) * (y_hat[0][j]) * (1- y_hat[0][j]))
            else:
               cost =  np.float((y[0][j] - y_hat[0][j]) * (1-((y_hat[0][j])**2)))
            sum+=((y[0][j] - y_hat[0][j]))**2
            Errors.append(cost)
       return sum,Errors
    
                    
    def Feedforward(self,x,y): 
        
        for i in range(0,len(x)):
             a = x[i].reshape((1,23))
             self.Activations[0] = a
           
             """forward"""
             for l in range( 1 , self.Num_Hidden_Layer+2):
                   if(self.Activation=='sigmoid'):
                     a = self.sigmoid(np.dot(a,self.Weights[l])+self.Bias[l] )
                   else:
                     a = self.Hyper_bolic_Tangent(np.dot(a,self.Weights[l])+self.Bias[l]) 
                   self.Activations[l] = a
                   if(l==self.Num_Hidden_Layer+1):
                     Y = y[i].reshape((1,5))
                     output = self.Activations[l].reshape((1,5))
                     cost,sigma = self.SigmaError( Y, output )
                     sigma = np.array(sigma)
                     self.Sigmas[l] = sigma.reshape((1,5))   
             self.Backward(x,y)
             self.Update_Weight()
             
    def Classify(self,x,classes): 
        Pred = []
        for i in range(0,len(x)):
             a = x[i].reshape((1,23))
             self.Activations[0] = a
           
             """forward"""
             for l in range( 1 , self.Num_Hidden_Layer+2):
                   if(self.Activation=='sigmoid'):
                     a = self.sigmoid(np.dot(a,self.Weights[l])+self.Bias[l] )
                   else:
                     a = self.Hyper_bolic_Tangent(np.dot(a,self.Weights[l])+self.Bias[l]) 
                   self.Activations[l] = a
                   if(l==self.Num_Hidden_Layer+1):
                     output = self.Activations[l].reshape((1,5))
                     idx = 0
                     mx = -1000.0
                     for i in range(5):
                         if output[0,i] > mx:
                             mx = output[0,i]
                             idx = i
                     Pred.append(classes[idx])
        return Pred
                     
                
             
                     
    
    def Backward(self,x,y): 
        for l in range( self.Num_Hidden_Layer , 0,-1):
                    if(self.Activation=='sigmoid'):
                      da = self.Dsigmoid(self.Activations[l])
                    else:
                      da = self.DHyper_bolic_Tangent(self.Activations[l]) 
                    self.Sigmas[l] = np.dot(self.Sigmas[l+1],self.Weights[l+1].T) * da
                    
                    
    def Update_Weight(self):
              for l in range( 1 ,self.Num_Hidden_Layer+2):
                 self.Weights[l] = self.Weights[l] + (self.Sigmas[l] * self.eta *  self.Activations[l-1].T)
                 if(self.Use_Bias==True):
                   self.Bias[l] = self.Bias[l] + (self.Sigmas[l] * self.eta )
    
    
    def Train(self,x,y):
               PrevACC = -1
               PrevWeights = copy.deepcopy(self.Weights)
               BestAcc = []
               ys = []
               xs = []
               if(self.Mse_threshold == 0):
                   for epoch in range(0,self.Num_epochs):
                      self.Feedforward(x,y)
                      confusion,cost= self.Mse(self.x_train,self.y_train )
                      ys.append(cost)
                      xs.append(epoch)
                      if(epoch%50==0):
                        Confusion,Mean_square_error = self.Mse(self.x_train,self.y_train)
                        print(Confusion)
                        print("Mse : " , Mean_square_error)
                        TestConf,TestMSE = self.Mse(self.x_test,self.y_test)
                        Acc = (TestConf[0][0]+TestConf[1][1]+TestConf[2][2] + TestConf[3][3] + TestConf[4][4])/26.0
                        print(TestConf)
                        print("\n",Acc,"\n---------------------------------------------------\n")
                        BestAcc.append(Acc)
                        if Acc < PrevACC:
                            self.Weights = copy.deepcopy(PrevWeights)
                            break;
                        PrevACC = Acc
                        PrevWeights = copy.deepcopy(self.Weights)
               else:
                   j = 0
                   Mean_square_error=1000
                   while  Mean_square_error > self.Mse_threshold:
                        self.Feedforward(x,y)
                        Confusion,Mean_square_error = self.Mse(self.x_train,self.y_train)
                        ys.append(Mean_square_error)
                        xs.append(j)
                        j+=1
                        if(j%50==0):
                            Confusion,Mean_square_error = self.Mse(self.x_train,self.y_train)
                            print(Confusion)
                            print("Mse : " , Mean_square_error)
                            TestConf,TestMSE = self.Mse(self.x_test,self.y_test)
                            Acc = (TestConf[0][0]+TestConf[1][1]+TestConf[2][2] + TestConf[3][3] + TestConf[4][4])/26.0
                            print(TestConf)
                            print("\n",Acc,"\n---------------------------------------------------\n")
                            BestAcc.append(Acc)
                            if Acc < PrevACC:
                                self.Weights = copy.deepcopy(PrevWeights)
                                break;
                            PrevACC = Acc
                            PrevWeights = copy.deepcopy(self.Weights)
                            
               fig = plt.figure()
               ax1 = fig.add_subplot(1,1,1)         
               ax1.clear()        
               ax1.plot(xs,ys)  
               plt.show()     
               Confusion,Mean_square_error = self.Mse(self.x_test,self.y_test)
               print(Confusion)
               print("Mse : " , Mean_square_error)
               print("Accuracy : ", (Confusion[0][0]+Confusion[1][1]+Confusion[2][2] + Confusion[3][3] + Confusion[4][4])/26.0  )
               print("The best accuracy is: " ,max(BestAcc))     
                     
    
    def Mse(self,x,y):
       Tot = 0
       m = len(x)
       Confusion = np.zeros((5,5))
       for i in range(len(x)):
          a = x[i].reshape((1,23))
          """forward"""
          for l in range( 1 , self.Num_Hidden_Layer+2):
              if(self.Activation=='sigmoid'):
                 a = self.sigmoid(np.dot(a,self.Weights[l])+ self.Bias[l])
              else:
                 a = self.Hyper_bolic_Tangent(np.dot(a,self.Weights[l])+self.Bias[l])  
              if(l==self.Num_Hidden_Layer+1):
                  Y = y[i].reshape((1,5))
                  output = a.reshape((1,5))
                  idx = [ j for j in range(0,5) if y[i][j]==1]
                  ind = np.argmax(a[0], axis=0)
                  Confusion[idx[0]][ind]+=1
                  #print(output)
                  cost,sigma = self.SigmaError( Y, output )
                  Tot+=cost
      
       return Confusion , (1/(2*m))*Tot
                
    
    def TrainTheModel(self,Hidden_Entry,epochs_Entry,LearningRate_Entry,Neurons_Entry,Activation_Entry,MSE_Entry,var,x_tr,y_tr,x_ts,y_ts):
        '''
        put the whole code here (training ---> Testing ---> Show the graph)
        this function will be called using "Train The Model" button
        '''
        self.Num_Hidden_Layer = int(Hidden_Entry) 
        self.Num_epochs = int(epochs_Entry) 
        self.eta = float(LearningRate_Entry) 
        Num_of_Neuron=str(Neurons_Entry)  #you must split to get the actual values
        self.Activation = str(Activation_Entry) 
        self.Mse_threshold = float(MSE_Entry) 
        self.x_train = x_tr
        self.y_train = y_tr
        self.x_test = x_ts
        self.y_test = y_ts
        if var:
            self.Use_Bias = True
        else:
            self.Use_Bias = False
        
        self.Num_of_Neurons.append(len(self.x_train[0,:]))
        for neuron in Num_of_Neuron.split(","):
          self.Num_of_Neurons.append(int(neuron))   
        self.Num_of_Neurons.append(len(self.y_train[0,:]))
        
        print(type(self.Num_of_Neurons))
        
        for l in range( 1 , self.Num_Hidden_Layer+2):
            self.Weights[l] = np.random.randn(self.Num_of_Neurons[l-1], self.Num_of_Neurons[l])*0.01
            self.Bias[l] = np.zeros(( 1 ,self.Num_of_Neurons[l]))

        self.Train(self.x_train,self.y_train)
        np.save('MLP_Weights.npy', self.Weights)
        np.save('MLP_Bias.npy', self.Bias)
        
        
    def LoadWeights(self,Hidden_Entry,epochs_Entry,LearningRate_Entry,Neurons_Entry,Activation_Entry,MSE_Entry,var,x_tr,y_tr,x_ts,y_ts):
        self.Num_Hidden_Layer = int(Hidden_Entry) 
        self.Num_epochs = int(epochs_Entry) 
        self.eta = float(LearningRate_Entry) 
        Num_of_Neuron=str(Neurons_Entry)  #you must split to get the actual values
        self.Activation = str(Activation_Entry) 
        self.Mse_threshold = float(MSE_Entry) 
        self.x_train = x_tr
        self.y_train = y_tr
        self.x_test = x_ts
        self.y_test = y_ts
        if var:
            self.Use_Bias = True
        else:
            self.Use_Bias = False
        
        self.Num_of_Neurons.append(len(self.x_train[0,:]))
        for neuron in Num_of_Neuron.split(","):
          self.Num_of_Neurons.append(int(neuron))   
        self.Num_of_Neurons.append(len(self.y_train[0,:]))
        
        self.Weights = np.load('MLP_Weights.npy').item()
        self.Bias = np.load('MLP_Bias.npy').item()
        
        TestConf,TestMSE = self.Mse(self.x_test,self.y_test)
        Acc = (TestConf[0][0]+TestConf[1][1]+TestConf[2][2] + TestConf[3][3] + TestConf[4][4])/26.0
        print(TestConf)
        print("\n",Acc,"\n---------------------------------------------------\n")

        
        
