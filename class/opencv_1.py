import numpy as np
import matplotlib.pyplot as mpt
from numpy.linalg import inv
from numpy.random import normal, rand
import cv2

a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
b = a.transpose()
print(a)
print(b)
print(a.T)
print(inv(a))

npRandNum = np.random.rand(256, 256)
r = (255*npRandNum).astype(np.uint8)
npRandNum = np.random.rand(256, 256)
g = (255*npRandNum).astype(np.uint8)
npRandNum = np.random.rand(256, 256)
b = (255*npRandNum).astype(np.uint8)

trgb = np.array([r, g, b])
rgb = trgb.transpose((1, 2, 0))


mpt.subplot(221), mpt.imshow(r, cmap=mpt.get_cmap('gray')), mpt.title('Red->Gray Image')
mpt.subplot(222), mpt.imshow(g, cmap=mpt.get_cmap('gray')), mpt.title('Green->Gray Image')
mpt.subplot(223), mpt.imshow(b, cmap=mpt.get_cmap('gray')), mpt.title('Blue->Gray Image')
mpt.subplot(224), mpt.imshow(rgb), mpt.title('RGB Image')
mpt.show()

x = np.linspace(0,10,100)
y = np.exp(x)
mpt.plot(x,y)
mpt.show()
x = np.random.normal(size=300)
mpt.hist(x, bins=25)
mpt.show()

npRandNum = np.random.rand(256, 256)
npRandImg = (255*npRandNum).astype(np.uint8)
cv2.imshow('Numpy Random Image', npRandImg)
cv2.waitKey(0)

print(npRandImg)

npArrayNum = np.zeros(256) + np.arange(0, 256, 1)[:, np.newaxis]
npArrayImg = npArrayNum.astype(np.uint8)
cv2.namedWindow('Numpy Array to Image')
cv2.imshow('Numpy Array to Image', npArrayImg)
cv2.waitKey(0)

mpt.imshow(npArrayImg, cmap=mpt.get_cmap('gray'))
mpt.title('python Matplotlib Gray Image')
mpt.xticks([]), mpt.yticks([])
mpt.show()

cvBgrImg = cv2.imread('C:/Users/user/Documents/Su_Github/ComputerVision/class/ImageProcessingHW.jpg')
cv2.namedWindow('OpenCV BGR Image')
cv2.imshow('OpenCV BGR Image', cvBgrImg)
cv2.waitKey(0)

mpt.imshow(cvBgrImg)
mpt.xticks([]), mpt.yticks([])
mpt.show()

b, g, r = cv2.split(cvBgrImg)
rgbImg = cv2.merge([r, g, b])
mpt.imshow(rgbImg)
mpt.xticks([]), mpt.yticks([])
mpt.show()

cvBgrImg = cv2.imread('C:/Users/user/Documents/Su_Github/ComputerVision/class/ImageProcessingHW.jpg',cv2.IMREAD_COLOR)
cv2.namedWindow('OpenCV BGR Image')
cv2.imshow('OpenCV BGR Image', cvBgrImg)
cv2.waitKey(0)

grayImage = cv2.cvtColor(cvBgrImg, cv2.COLOR_BGR2GRAY)

cv2.namedWindow('OpenCV Gray Image')
cv2.imshow('OpenCV Gray Image', grayImage)
cv2.waitKey(0)

mpt.imshow(grayImage, cmap=mpt.get_cmap('gray'))
mpt.title('Python Matplotlib Gray Image')
mpt.xticks([]), mpt.yticks([])
mpt.show()

guiImg = np.zeros((256, 256, 3), np.uint8)

guiImg = cv2.line(guiImg, (0, 255), (255, 0), (0, 0, 255), 8)
guiImg = cv2.circle(guiImg, (127, 127), 120, (0, 255, 0), -1)
guiImg = cv2.rectangle(guiImg, (100, 100), (150, 140), (255, 0, 0), 5)
guiImg = cv2.ellipse(guiImg, (127, 50), (100, 40), 0, 0, 300, 255, -1)

pt = np.array([[100, 5], [120, 30], [150, 70], [90, 100]], np.int32)
pt = pt.reshape((-1,1,2))
img = cv2.polylines(guiImg, [pt], True, (0,255,255))

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(guiImg, 'Hello!', (5,200), font, 3, (255, 0, 255), 5, cv2.LINE_AA)

cv2.namedWindow('OpenCV GUI Image')
cv2.imshow('OpenCV GUI Image', guiImg)
cv2.waitKey(0)
