import numpy as np
import cv2
import matplotlib.pyplot as plt
from tkinter import *
import tkinter.filedialog as fd

root = Tk()
root.filename = fd.askopenfilename(filetypes=(("JPEG files",".jpg"),("ALL files","*.*")))
cvGrayImg = cv2.imread(root.filename,cv2.IMREAD_GRAYSCALE)
cv2.namedWindow('Source Gray')
cv2.imshow('Source Gray', cvGrayImg)
cv2.waitKey(0)

cvHist = cv2.calcHist([cvGrayImg],[0],None,[256],[0,256])
npHist, bins = np.histogram(cvGrayImg.ravel(),256,[0,256])
plt.subplot(221), plt.imshow(cvGrayImg,'gray'), plt.title("Source Image")
plt.subplot(222),plt.hist(cvGrayImg.ravel(),256,[0,256]), plt.title("Matplotlib Histogram")
plt.subplot(223), plt.plot(cvHist), plt.title('OpenCV Histogram')
plt.subplot(224), plt.plot(npHist), plt.title('Numpy Histogram')
plt.show()

eqImg = cv2.equalizeHist(cvGrayImg)
cvHist2 = cv2.calcHist([eqImg], [0], None, [256], [0, 256])
plt.subplot(221), plt.imshow(cvGrayImg,'gray'),plt.title('Source Image')
plt.subplot(222), plt.plot(cvHist), plt.title('OpenCV1 Histogram')
plt.subplot(223), plt.imshow(eqImg,'gray'), plt.title('equlized Image')
plt.subplot(224), plt.plot(cvHist2), plt.title('OpenCV2 Histogram')
plt.show()

cv2.destroyAllWindows()
