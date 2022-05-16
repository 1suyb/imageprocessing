import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import filedialog as fd
from math import sqrt

class SIFT :
    def __init__(self):
        #self.objectImage = self.__fileLoad("ObjectImage")
        #self.sceneImage = self.__fileLoad("sceneImage")
        self.objectImage = cv2.imread(r"C:\Users\user\Documents\Su_Github\ComputerVision\Test.jpg", cv2.IMREAD_GRAYSCALE)
        self.sceneImage = cv2.imread(r"C:\Users\user\Documents\Su_Github\ComputerVision\Test.jpg", cv2.IMREAD_GRAYSCALE)
        self.lv1py=0
        self.lv2py=0
        self.lv3py=0
        self.lv4py=0
        # self.doglv1=0
        # self.doglv2=0
        # self.doglv3=0
        # self.doglv4=0
        return
    # 이미지 파일 가져오기
    def __fileLoad(self,_title) :
        filename = fd.askopenfilename(initialdir="/", title=_title)
        grayimage = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        return grayimage

    def makeScaleSpace(self, s=1.6, k=sqrt(2), c=5):
        lv1sigma = np.zeros(c)
        for i in range(c) :
            lv1sigma[i] = s*(k**i)
        lv2sigma = 2*lv1sigma.copy()
        lv3sigma = 2*lv2sigma.copy()
        lv4sigma = 2*lv3sigma.copy()
        doubleimage = cv2.resize(self.objectImage, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        normalimage = self.objectImage
        halfimage = cv2.resize(self.objectImage, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        quaterimage = cv2.resize(self.objectImage, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)
        self.lv1py = np.zeros((doubleimage.shape[0],doubleimage.shape[1],c))
        self.lv2py = np.zeros((normalimage.shape[0],normalimage.shape[1],c))
        self.lv3py = np.zeros((halfimage.shape[0],halfimage.shape[1],c))
        self.lv4py = np.zeros((quaterimage.shape[0],quaterimage.shape[1],c))
        for i in range(c):
            self.lv1py[:, :, i] = cv2.GaussianBlur(doubleimage, (0, 0), lv1sigma[i])
            self.lv2py[:, :, i] = cv2.GaussianBlur(normalimage, (0, 0), lv2sigma[i])
            self.lv3py[:, :, i] = cv2.GaussianBlur(halfimage, (0, 0), lv3sigma[i])
            self.lv4py[:, :, i] = cv2.GaussianBlur(quaterimage, (0, 0), lv4sigma[i])

    def differenceofGaussian(self,c=5):
        print(self.lv1py.shape[0])
        self.doglv1 = np.zeros((self.lv1py.shape[0], self.lv1py.shape[1], c-1))
        self.doglv2 = np.zeros((self.lv2py.shape[0],self.lv2py.shape[1], c-1))
        self.doglv3 = np.zeros((self.lv3py.shape[0],self.lv3py.shape[1], c-1))
        self.doglv4 = np.zeros((self.lv4py.shape[0],self.lv4py.shape[1], c-1))
        for i in range(c-1):
            self.doglv1[:, :, i] = self.lv1py[:, : , i]-self.lv1py[:, : , i+1]
            self.doglv2[:, :, i] = self.lv2py[:, : , i]-self.lv2py[:, : , i+1]
            self.doglv3[:, :, i] = self.lv3py[:, : , i] - self.lv3py[:, : , i + 1]
            self.doglv4[:, :, i] = self.lv4py[:, : , i] - self.lv4py[:, : , i + 1]

    def getExtremaDetection(self,dog,c=5):
        print(dog.shape[2])
        for k in range(1,dog.shape[2]-1):
            print("===================================================================================================")
            for i in range(1,dog[0].shape[0]-1):
                for j in range(1,dog[0].shape[1]-1):
                    local = [dog[i-1:i+2, j-1:j+2, k-1], dog[i-1:i+2, j-1:j+2, k], dog[i-1:i+2, j-1:j+2, k+1]]
                    if local[0].shape[0]<3 : return
                    print(local)
                    max = np.max(local)
                    min = np.min(local)
                    if(max == dog[i,j,k] or min==dog[i,j,k]):

    def makeHashanMatrix(self,local):
        xxmask = np.array([[1,0,1],[2,-8,2],[1,0,1]])
        xymask = np.array([[0,1,0],[1,-4,1],[0,1,0]])
        yymask = np.array([[1,2,1],[0,-8,0],[1,2,1]])
        hashan = np.array([local*xxmask,local*xymask],[local*xymask,local*yymask])
        R = hashan[0,0]*hashan[1,1]-hashan[0,1]**2 - 0.05*(hashan[0,0]+hashan[1,1])
        if R >0 and R>0.5 :
            return 1
        return 0

if __name__  == "__main__" :
    sift = SIFT()
    sift.makeScaleSpace()
    sift.differenceofGaussian()
    sift.getExtremaDetection(sift.doglv1,5)
    cv2.destroyAllWindows()
    # cv2.imshow("hello", cv2.GaussianBlur(sift.objectImage, (0, 0), 2*1.6*(2**0)))
    # cv2.waitKey(0)
