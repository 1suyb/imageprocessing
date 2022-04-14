import cv2
import numpy as np
from tkinter import *
import tkinter.filedialog as filedia

root = Tk();
root.filename = filedia.askopenfilename(filetypes=(("JPEG files", ".jpg"),("All files","*.*")))
cvBgrImg = cv2.imread(root.filename,cv2.IMREAD_COLOR)
cvGrayImg = cv2.imread(root.filename,cv2.IMREAD_GRAYSCALE)

cv2.namedWindow('BGR Image')
cv2.imshow('BGR Image', cvBgrImg)
cv2.waitKey(0)

cv2.namedWindow('Gray Image')
cv2.imshow('Gray Image', cvGrayImg)
cv2.waitKey(0)

hsv = cv2.cvtColor(cvBgrImg, cv2.COLOR_BGR2GRAY)
h, s, v = cv2.split(cvBgrImg)
cv2.namedWindow('C Gray Image')
cv2.imshow('C Gray Image', v)
cv2.waitKey(0)

cv2.namedWindow('Source Gray')
cv2.imshow('Source Gray', cvGrayImg)
cv2.waitKey(0)
"""
h, w = cvGrayImg.shape

for y in range(h) :
    for x in range(w) :
        tp = cvGrayImg[y,x] - 30
        if tp < 0 :
            tp = 0
        cvGrayImg[y, x] = tp

cv2.namedWindow('Gray + Image')
cv2.imshow('Gray+ Image', cvGrayImg)
cv2.waitKey(0)

for y in range(h) :
    for x in range(w) :
        tp = cvGrayImg[y,x] * 1.4
        if tp > 255 :
            tp = 255
        cvGrayImg[y, x] = tp

cv2.namedWindow('Gray +L Image')
cv2.imshow('Gray+L Image', cvGrayImg)
cv2.waitKey(0)
"""
"""
LUT = np.zeros(256, dtype=np.uint8)
for i in range(256) :
    nv = i + 30
    if nv > 255 : nv = 255
    LUT[i] = nv

h, w = cvGrayImg.shape
for y in range(h) :
    for x in range(w) :
        cvGrayImg[y,x] = LUT[cvGrayImg[y,x]]

cv2.namedWindow('Gray_L Image')
cv2.imshow('Gray+L Image', cvGrayImg)
cv2.waitKey(0)
"""
gamma = 0.5
LUT = np.zeros(256, dtype=np.uint8)
for i in range(256) :
    #nv = 255.0*((i/255.0)**(1/gamma))
    #nv = 255-i
    #nv = 255/(255-50)*(i-50)
    #if(i<=127) :
    #    nv = i**2/127
    #else :
    #    nv = -(i-255)**2/127+255
    nv = -255/(127**2)*(i-128)**2+255
    if nv > 255 : nv = 255
    LUT[i] = nv

h, w = cvGrayImg.shape
for y in range(h) :
    for x in range(w) :
        cvGrayImg[y,x] = LUT[cvGrayImg[y,x]]

cv2.namedWindow('Gray_L Image')
cv2.imshow('Gray+L Image', cvGrayImg)
cv2.waitKey(0)

cv2.destroyAllWindows()