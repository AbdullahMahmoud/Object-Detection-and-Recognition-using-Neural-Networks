import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2 
from skimage import measure
import glob
from sklearn.decomposition import PCA
class Preparation:
    def __init__(self):
        self.clf=PCA(0.99,whiten=True)     #converse 90% variance
        self.RegionIndex = []
        self.classes = []
    def GetDataset(self,TrainingDatasetPath , TestingDatasetPath):
        tmp_x_train = np.full((25,2500),0)
        y_train = np.full((25,5),0)
        idx=0
        for filename in glob.glob(TrainingDatasetPath + '/*.jpg'): 
            img = cv2.imread(filename,0)
            GrayImage = cv2.resize(img, (50, 50)) 
            tmp_x_train[idx,:] = np.array(GrayImage).reshape((1,2500))
            image = filename[len(TrainingDatasetPath)+2:]
            if image.split("- ")[1][:-4] not in self.classes:
                self.classes.append(image.split("- ")[1][:-4])
            y_train[idx,self.classes.index(image.split("- ")[1][:-4])] = 1
            idx = idx + 1
        X_train=self.clf.fit_transform(tmp_x_train)
           
        tmp_x_test = np.full((26,2500),0)
        y_test = np.full((26,5),0)
        idx=0
        for filename in glob.glob(TestingDatasetPath + '/*.jpg'): 
            img = cv2.imread(filename,0)
            GrayImage = cv2.resize(img, (50, 50)) 
            tmp_x_test[idx,:] = np.array(GrayImage).reshape((1,2500))
            image = filename[len(TrainingDatasetPath)+2:]
            y_test[idx,self.classes.index(image.split("- ")[1][:-4])] = 1
            idx = idx + 1        
        X_test=self.clf.transform(tmp_x_test)
        return X_train , y_train , X_test , y_test , tmp_x_train , tmp_x_test
    def PrepareSample(self,SamplePath):
        self.RegionIndex = []
        RealImage = Image.open(SamplePath + ".jpg").convert('L')
        ColoredImage = Image.open(SamplePath + ".png").convert('L')
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        cannyimg = cv2.dilate(cv2.Canny(np.array(ColoredImage),20,20) , kernel, iterations = 1)
        SigmentedImage = cv2.bitwise_not(cannyimg)
        labels = measure.label(SigmentedImage)
        self.regions = measure.regionprops(labels)
        idx = 0
        for region in self.regions:
            r0, c0, r1, c1 = region.bbox
            ratio = ((r1-r0)*(c1-c0))/(len(np.array(RealImage)[:,0]) * len(np.array(RealImage)[0,:]))
            if not(ratio > 3/4):
                self.RegionIndex.append(idx)
            idx = idx + 1
        TmpFeatures = np.full((len(self.RegionIndex),2500),0)
        i = 0
        idx = 0
        for region in self.regions:
            if idx == self.RegionIndex[i]:
                r0, c0, r1, c1 = region.bbox
                img = np.array(RealImage)[r0:r1,c0:c1]
                GrayImage = cv2.resize(img, (50, 50)) 
                TmpFeatures[i,:] = np.array(GrayImage).reshape((1,2500))
                i = i + 1
            idx = idx + 1
        return self.clf.transform(TmpFeatures) , TmpFeatures
    def Display(self , Pred,SamplePath):
        print("\n\t\t-------------------------- Result of Image: " + SamplePath + " --------------------------\n")
        RealImage = Image.open(SamplePath + ".jpg")
        i = 0
        idx = 0
        for region in self.regions:
            if idx == self.RegionIndex[i]:
                r0, c0, r1, c1 = region.bbox
                img = np.array(RealImage)[r0:r1,c0:c1,:]
                plt.figure()
                plt.imshow(img)
                plt.show()
                print(Pred[i])
                i = i + 1
            idx = idx + 1
        