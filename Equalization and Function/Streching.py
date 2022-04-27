import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import  filedialog as fd

def StrechingLut(a, b) :
    LUT = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        nv = 255/(b-a)*(i-a)
        if nv > 255 : nv = 255
        if nv < 0 : nv = 0
        LUT[i] = nv
    return LUT

def ShowHist(img,name) :
    plt.subplot(121),plt.imshow(img,"gray"),plt.title(name)
    plt.subplot(122),plt.hist(img.ravel(), 256, [0, 256]), plt.title(name+"Hist")
    plt.title("Histogram of "+name)
    plt.show()

root = Tk()
filename = fd.askopenfilename(initialdir="/", title= "Select file",filetypes=(("image files",("*.jpg","*.png")),("allfiles","*")))
w = 300
h = 400
cvRgbImg = cv2.imread(filename,cv2.IMREAD_COLOR);
cvRgbImg = cv2.resize(cvRgbImg,(300,400))
cv2.imshow("cvRgbImg",cvRgbImg)
cv2.waitKey(0)

cvGrayImg = cv2.cvtColor(cvRgbImg,cv2.COLOR_RGB2GRAY)
ShowHist(cvGrayImg,"cvGrayImg")

cv2.imshow("cvGrayImg",cvGrayImg)
cv2.waitKey(0)

cvStreching = cvGrayImg
LUT = StrechingLut(5,150)
for y in range(h) :
    for x in range(w) :
        cvStreching[y,x] = LUT[cvGrayImg[y,x]]
ShowHist(cvStreching, "cvStreching")
cv2.imshow("histogram Streching",cvStreching)
cv2.waitKey(0)



cvEqualize = cv2.equalizeHist(cvGrayImg)
ShowHist(cvEqualize,"cvEqualize")
cv2.imshow("histogram Equalize", cvEqualize)
cv2.waitKey(0)


cvMinus = cvStreching-cvEqualize
for y in range(h) :
    for x in range(w) :
        if cvMinus[y,x] < 0 : cvMinus[y,x] = 0
        if cvMinus[y,x] > 255 : cvMinus[y,x] = 255
ShowHist(cvMinus, "cvMinus")
cv2.imshow("cvMinus",cvMinus)

cv2.waitKey(0)

h = 768
w = 1024
Graph = np.zeros((768, 1024))
cv2.imshow("Graph", Graph)
cv2.waitKey(0)
for y in range(h) :
    for x in range(w) :
        Graph[y,x] = 255
        if y == (x**3 - 2*x**2 + x - 150) :
            Graph[y,x] = 0
cv2.imshow("Graph",Graph)
cv2.waitKey(0)
preX = 0
for y in range(h) :
    for x in range(w) :
        Graph[y,x] = 255
        if y == x**3 - 2*x**2 + x - 150:
            Graph[y,x] = 255/2
            preX = x;
        if x == preX :
            Graph[y,x] = 0
cv2.imshow("Graph",Graph)
cv2.waitKey(0)

# https://daewoonginfo.blogspot.com/2019/05/opencv-python-resize.html
# https://wikidocs.net/21113