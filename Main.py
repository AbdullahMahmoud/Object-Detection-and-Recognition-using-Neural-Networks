from BackPropagation import MLP
from rbf import RBF
from PrepareDataset import Preparation
import numpy as np
import matplotlib.pyplot as plt
from PCA import pca
def Train():
    global MLPObj , PrepareObj , RBFObj , PCAObj , sc_x
    MLPObj = MLP()
    PrepareObj = Preparation()
    RBFObj = RBF()
    PCAObj = pca()
    x_train,y_train,x_test,y_test,Original_x_train , Original_x_test = PrepareObj.GetDataset("C:\\Users\\Lenovo-PC\\Desktop\\neural-network-course\\Project\\Data set\\Training","C:\\Users\\Lenovo-PC\\Desktop\\neural-network-course\\Project\\Data set\\Testing")
    if NNPCAVar.get():
        PCAObj.LoadWeights()
        x_train = PCAObj.transform(Original_x_train)
        x_test = PCAObj.transform(Original_x_test)
    from sklearn.preprocessing import StandardScaler
    sc_x = StandardScaler()
    x_train = sc_x.fit_transform(x_train)
    x_test = sc_x.transform(x_test)
    if LoadTrainVar.get():
        if AlgoVar.get():
            MLPObj.TrainTheModel(Hidden_Entry.get(),epochs_Entry.get(),LearningRate_Entry.get(),Neurons_Entry.get(),Activation_Entry.get(),MSE_Entry.get(),var.get() , x_train,y_train,x_test,y_test)
        else:
            RBFObj.TrainTheModel_rbf(Neurons_Entry.get(),LearningRate_Entry.get(),MSE_Entry.get(),epochs_Entry.get(),5, x_train,y_train,x_test,y_test)
    else:
        if AlgoVar.get():
            MLPObj.LoadWeights(Hidden_Entry.get(),epochs_Entry.get(),LearningRate_Entry.get(),Neurons_Entry.get(),Activation_Entry.get(),MSE_Entry.get(),var.get() , x_train,y_train,x_test,y_test)
        else:
            RBFObj.LoadWeights(Neurons_Entry.get(),LearningRate_Entry.get(),MSE_Entry.get(),epochs_Entry.get(),5, x_train,y_train,x_test,y_test)
            

def OpenImage():
    from PIL import ImageTk, Image
    canvas.image = ImageTk.PhotoImage(Image.open(str(ImageName_Entry.get()) + ".jpg"))
    canvas.create_image(0,0,anchor = 'nw' , image = canvas.image)
def Classify():
    global MLPObj , PrepareObj , RBFObj , PCAObj , sc_x
    Features , OriginalFeatures = PrepareObj.PrepareSample(str(ImageName_Entry.get()))
    if NNPCAVar.get():
        Features = PCAObj.transform(OriginalFeatures)
    Features = sc_x.transform(Features)
    if AlgoVar.get():
        Preds = MLPObj.Classify(Features , PrepareObj.classes)
    else:
        Preds = RBFObj.Classify(Features , PrepareObj.classes)
    PrepareObj.Display(Preds , str(ImageName_Entry.get()))

#x_train,y_train,x_test,y_test , data = GetDataset("C:\\Users\\Lenovo-PC\\Desktop\\neural-network-course\\Project\\Data set\\Training","C:\\Users\\Lenovo-PC\\Desktop\\neural-network-course\\Project\\Data set\\Testing")


































#------------------------------------ GUI ------------------------------------
from tkinter import *

#Creating the main window
root = Tk()
#Controls
Hidden_Label = Label(root , text = "number of hidden layers")
Hidden_Entry = Entry(root)
epochs_Label = Label(root , text = "number of epochs")
epochs_Entry = Entry(root)
LearningRate_Label = Label(root , text = "learning rate")
LearningRate_Entry = Entry(root)
Neurons_Label = Label(root , text = "number of neurons per each layer")
Neurons_Entry = Entry(root)
Activation_Label = Label(root , text = "Activation function type")
Activation_Entry = Entry(root)
MSE_Label = Label(root , text = "MSE threshold")
MSE_Entry = Entry(root)
var = IntVar()
cBox = Checkbutton(root , text = "Use bias" , variable = var)
AlgoVar = IntVar()
AlgorithmCBox = Checkbutton(root , text = "Checked(MLP)/Not Checked(RBF)" , variable = AlgoVar)
LoadTrainVar = IntVar()
LoadTrainCBox = Checkbutton(root , text = "Checked(Train)/Not Checked(Load)" , variable = LoadTrainVar)
NNPCAVar = IntVar()
NNPCACBox = Checkbutton(root , text = "Checked(NN PCA)/Not Checked(Statistical PCA)" , variable = NNPCAVar)
Train_Button = Button(root , text = "Load/Train The Model" , command = Train)
ImageName_Label = Label(root , text = "Image Name")
ImageName_Entry = Entry(root)
OpenImage_Button = Button(root , text = "Open The Image" , command = OpenImage)
canvas = Canvas(root,width=500,height=500)
Classify_Button = Button(root , text = "Classify" , command = Classify)


#Controls' positions
Hidden_Label.grid(row=0, column=0)
Hidden_Entry.grid(row=0, column=1)
epochs_Label.grid(row=1 , column=0 )
epochs_Entry.grid(row=1 , column=1 )
LearningRate_Label.grid(row=2 , column=0 )
LearningRate_Entry.grid(row=2 , column=1 )
Neurons_Label.grid(row=3, column=0)
Neurons_Entry.grid(row=3, column=1)
Activation_Label.grid(row=4, column=0)
Activation_Entry.grid(row=4, column=1)
MSE_Label.grid(row = 5 , column = 0)
MSE_Entry.grid(row = 5 , column = 1)
cBox.grid(row=6, column=1)
AlgorithmCBox.grid(row=7,column=1)
NNPCACBox.grid(row=8 , column=1)
LoadTrainCBox.grid(row=9, column=1)
Train_Button.grid(row=10, column=1)
ImageName_Label.grid(row=11,column=0)
ImageName_Entry.grid(row=11,column=1)
OpenImage_Button.grid(row=11 , column=2)
canvas.grid(row = 12 , column = 1)
Classify_Button.grid(row = 13 , column = 1)
#For Making the window still displayed
root.mainloop()
