import numpy as np
import cv2
import matplotlib.pyplot as plt
from tkinter import *
import tkinter.filedialog as fd

root = Tk()

root.filename = fd.askopenfilename(filetypes=(("JPEG files",".jpg"),("All files","*.*")))
cvGrayImg = cv2.imread(root.filename, cv2.IMREAD_GRAYSCALE)

cv2.namedWindow('Original Gray')
cv2.imshow('Original Gray', cvGrayImg)
cv2.waitKey(0)

mask = np.array(([0,-1,0],[-1,0,1],[0,1,0]),dtype=np.float32)
dstImg = cv2.filter2D(cvGrayImg,-1,mask)

cv2.namedWindow('Convex Gray')
cv2.imshow('Convex Gray',dstImg)
cv2.waitKey(0)

plt.subplot(121), plt.imshow(cvGrayImg,'gray'), plt.title('Original Image')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(dstImg,'gray'), plt.title('Convex Image')
plt.xticks([]), plt.yticks([])
plt.show()

mask = np.ones((5,5),np.float32)/25
dstImg = cv2.filter2D(cvGrayImg,-1,mask)

cv2.namedWindow('averaging Gray')
cv2.imshow('averaging Gray',dstImg)
cv2.waitKey(0)

plt.subplot(121), plt.imshow(cvGrayImg,'gray'), plt.title('Original Image')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(dstImg,'gray'), plt.title('averaging Image')
plt.xticks([]), plt.yticks([])
plt.show()

mask = np.array(([-1,-1,-1],[-1,9,-1],[-1,-1,-1]),dtype=np.float32)
dstImg = cv2.filter2D(cvGrayImg,-1,mask)

cv2.namedWindow('sharping Gray')
cv2.imshow('sharping Gray',dstImg)
cv2.waitKey(0)

plt.subplot(121), plt.imshow(cvGrayImg,'gray'), plt.title('Original Image')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(dstImg,'gray'), plt.title('sharping Image')
plt.xticks([]), plt.yticks([])
plt.show()

mask = np.array(([-1, 0, 1], [-2, 0, 2], [-1, 0, 1]), dtype=np.float32)
dstImg = cv2.filter2D(cvGrayImg, -1, mask)

cv2.namedWindow('sobelx Image')
cv2.imshow('sobelx Image',dstImg)
cv2.waitKey(0)

plt.subplot(121), plt.imshow(cvGrayImg,'gray'), plt.title('Original Image')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(dstImg,'gray'), plt.title('sobelx Image')
plt.xticks([]), plt.yticks([])
plt.show()

mask = np.array(([-1, -2, -1], [0, 0, 0], [1, 2, 1]), dtype=np.float32)
dstImg = cv2.filter2D(cvGrayImg, -1, mask)

cv2.namedWindow('sobely Image')
cv2.imshow('sobely Image',dstImg)
cv2.waitKey(0)

plt.subplot(121), plt.imshow(cvGrayImg,'gray'), plt.title('Original Image')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(dstImg,'gray'), plt.title('sobely Image')
plt.xticks([]), plt.yticks([])
plt.show()

mask = np.array(([-1, 0, 1], [-1, 0, 1], [-1, 0, 1]), dtype=np.float32)
dstImg = cv2.filter2D(cvGrayImg, -1, mask)

cv2.namedWindow('prewittX Image')
cv2.imshow('prewittX Image',dstImg)
cv2.waitKey(0)

plt.subplot(121), plt.imshow(cvGrayImg,'gray'), plt.title('Original Image')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(dstImg,'gray'), plt.title('prewittX Image')
plt.xticks([]), plt.yticks([])
plt.show()

mask = np.array(([-1, -1, -1], [0, 0, 0], [1, 1, 1]), dtype=np.float32)
dstImg = cv2.filter2D(cvGrayImg, -1, mask)

cv2.namedWindow('prewittY Image')
cv2.imshow('prewittY Image',dstImg)
cv2.waitKey(0)

plt.subplot(121), plt.imshow(cvGrayImg,'gray'), plt.title('Original Image')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(dstImg,'gray'), plt.title('prewittY Image')
plt.xticks([]), plt.yticks([])
plt.show()

mask = np.array(([-1, 0, 0], [0, 1, 0], [0, 0, 0]), dtype=np.float32)
dstImg = cv2.filter2D(cvGrayImg, -1, mask)

cv2.namedWindow('RobersY Image')
cv2.imshow('RobersY Image',dstImg)
cv2.waitKey(0)

plt.subplot(121), plt.imshow(cvGrayImg,'gray'), plt.title('Original Image')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(dstImg,'gray'), plt.title('RobersY Image')
plt.xticks([]), plt.yticks([])
plt.show()

mask = np.array(([0, 0, -1], [0, 1, 0], [0, 0, 0]), dtype=np.float32)
dstImg = cv2.filter2D(cvGrayImg, -1, mask)

cv2.namedWindow('RobersX Image')
cv2.imshow('RobersX Image',dstImg)
cv2.waitKey(0)

plt.subplot(121), plt.imshow(cvGrayImg,'gray'), plt.title('Original Image')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(dstImg,'gray'), plt.title('RobersX Image')
plt.xticks([]), plt.yticks([])
plt.show()



mask = np.array(([0, -1, 0], [-1, 4, -1], [0, -1, 0]), dtype=np.float32)
dstImg = cv2.filter2D(cvGrayImg, -1, mask)

cv2.namedWindow('Laplacian4 Image')
cv2.imshow('Laplacian4 Image',dstImg)
cv2.waitKey(0)

plt.subplot(121), plt.imshow(cvGrayImg,'gray'), plt.title('Original Image')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(dstImg,'gray'), plt.title('Laplacian4 Image')
plt.xticks([]), plt.yticks([])
plt.show()

mask = np.array(([-1, -1, -1], [-1, 8, -1], [-1, -1, -1]), dtype=np.float32)
dstImg = cv2.filter2D(cvGrayImg, -1, mask)

cv2.namedWindow('Laplacian8 Image')
cv2.imshow('Laplacian8 Image',dstImg)
cv2.waitKey(0)

plt.subplot(121), plt.imshow(cvGrayImg,'gray'), plt.title('Original Image')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(dstImg,'gray'), plt.title('Laplacian8 Image')
plt.xticks([]), plt.yticks([])
plt.show()

mask = np.array(([0, -1, 0], [-1, 4, -1], [0, -1, 0]), dtype=np.float32)
msize = 3
sigma = 0.3*((msize-1)*0.5-1)+0.8
blurImg = cv2.GaussianBlur(cvGrayImg,(msize,msize),sigma,0)
dstImg = cv2.filter2D(cvGrayImg, -1, mask)

cv2.namedWindow('Laplacian4G Image')
cv2.imshow('Laplacian4G Image', dstImg)
cv2.waitKey(0)

plt.subplot(121), plt.imshow(cvGrayImg,'gray'), plt.title('Original Image')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(dstImg,'gray'), plt.title('Laplacian4G Image')
plt.xticks([]), plt.yticks([])
plt.show()

mask = np.array(([-1, -1, -1], [-1, 8, -1], [-1, -1, -1]), dtype=np.float32)
sigma = 0.3*((msize-1)*0.5-1)+0.8
blurImg = cv2.GaussianBlur(cvGrayImg,(msize,msize),sigma,0)
dstImg = cv2.filter2D(cvGrayImg, -1, mask)

cv2.namedWindow('Laplacian8G Image')
cv2.imshow('Laplacian8G Image', dstImg)
cv2.waitKey(0)

plt.subplot(121), plt.imshow(cvGrayImg,'gray'), plt.title('Original Image')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(dstImg,'gray'), plt.title('Laplacian8G Image')
plt.xticks([]), plt.yticks([])
plt.show()

dstImg = cv2.Canny(cvGrayImg,30,200)

cv2.namedWindow('Canny')
cv2.imshow('Canny',dstImg)
cv2.waitKey(0)

plt.subplot(121), plt.imshow(cvGrayImg,'gray'), plt.title('Original Image')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(dstImg,'gray'), plt.title('Canny')
plt.xticks([]), plt.yticks([])
plt.show()

dstImg = cv2.cornerHarris(cvGrayImg, 2, 3, 0.04)

cv2.namedWindow('Harris')
cv2.imshow('Harris',dstImg)
cv2.waitKey(0)
plt.subplot(121), plt.imshow(cvGrayImg,'gray'), plt.title('Original Image')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(dstImg,'gray'), plt.title('Harris')
plt.xticks([]), plt.yticks([])
plt.show()


cv2.destroyAllWindows()