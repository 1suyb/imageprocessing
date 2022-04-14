import numpy as np
import cv2
import matplotlib.pyplot as plt
from tkinter import *
import tkinter.filedialog as fd

root = Tk()

root.fileName = fd.askopenfilename(filetypes=(("JPEG files", ".jpg"), ("All files","*.*")))
cvGrayImg = cv2.imread(root.fileName, cv2.IMREAD_GRAYSCALE)

cv2.namedWindow('Original Gray')
cv2.imshow('Original Gray', cvGrayImg)
cv2.waitKey(0)

dstImg = cv2.medianBlur(cvGrayImg,9)

cv2.namedWindow('Median')
cv2.imshow('Median', dstImg)
cv2.waitKey(0)

dstImg = cv2.fastNlMeansDenoising(cvGrayImg,None,10,7,21)

cv2.namedWindow('meansdenoising')
cv2.imshow('measdenoising',dstImg)
cv2.waitKey(0)
cv2.destroyAllWindows()