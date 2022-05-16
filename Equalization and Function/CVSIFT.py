import numpy as np
import cv2
from matplotlib import pyplot as plt
from tkinter import *
from tkinter import filedialog as fd


class SIFT:
    # 생성자
    def __init__(self,MIN_MATCH_COUNT = 5):
        self.objectimage = self.fileLoad("ObjectImage")
        self.sceneimage = self.fileLoad("SceneImage")
        self.sift = cv2.SIFT_create()
        self.MIN_MATCH_COUNT = MIN_MATCH_COUNT
        self.FLANN_INDEX_KDTREE = 1

    # tkinter이용 이미지 불러오기
    def fileLoad(self,_title):
        filename = fd.askopenfilename(initialdir="/", title=_title)
        grayimage = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        return grayimage

    # 오브젝트 이미지에서 Featurepoint보여주기
    def showFeaturePoint(self):
        img = 0
        kp = self.sift.detect(self.objectimage,None)
        img = cv2.drawKeypoints(self.objectimage,kp,img,(0,0,255))
        cv2.imshow("FeaturePoints",img)
        cv2.waitKey(0)
    # 씬이미지랑 오브젝트이미지랑 피쳐포인트 매칭해서 씬이미지에서 오브젝트 검출하기
    def matchingImage(self):
        # 특징점과 특징디스크립터를 찾고 서로 매칭해주기
        kp1,des1 = self.sift.detectAndCompute(self.objectimage,None)
        kp2,des2 = self.sift.detectAndCompute(self.sceneimage,None)
        index_params = dict(algorithm = self.FLANN_INDEX_KDTREE,trees = 5)
        search_params = dict(checks = 50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        good =[]
        for m,n in matches:
            if m.distance < 0.7*n.distance :
                good.append(m)

        if len(good) > self.MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()
            h, w  = self.objectimage.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            img2 = cv2.polylines(self.sceneimage, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
        else:
            print("Not enough matches are found - {}/{}".format(len(good), self.MIN_MATCH_COUNT))
            matchesMask = None
        # 매칭된 점들을 서로 이어서 보여주기
        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=matchesMask,  # draw only inliers
                           flags=2)
        img3 = cv2.drawMatches(self.objectimage, kp1, self.sceneimage, kp2, good, None, **draw_params)
        plt.imshow(img3, 'gray'), plt.show()


if __name__ == "__main__":
    sift = SIFT()
    sift.showFeaturePoint()
    sift.matchingImage()
    cv2.destroyAllWindows()

#https://toitoitoi79.tistory.com/m/85

#
# MIN_MATCH_COUNT = 10
#
# img1 = __fileLoad("object")         # queryImage
# img2 = __fileLoad("scene") # trainImage
#
#
# sift = cv.SIFT_create()
#
# # find the keypoints and descriptors with SIFT
# kp1, des1 = sift.detectAndCompute(img1,None)
# kp2, des2 = sift.detectAndCompute(img2,None)
# FLANN_INDEX_KDTREE = 1
# index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
# search_params = dict(checks = 50)
# flann = cv.FlannBasedMatcher(index_params, search_params)
# matches = flann.knnMatch(des1,des2,k=2)
#
#
# # store all the good matches as per Lowe's ratio test.
# good = []
# for m,n in matches:
#     if m.distance < 0.7*n.distance:
#         good.append(m)
#
# if len(good)>MIN_MATCH_COUNT:
#     src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
#     dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
#     M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
#     matchesMask = mask.ravel().tolist()
#     h,w = img1.shape
#     pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
#     dst = cv.perspectiveTransform(pts,M)
#     img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
# else:
#     print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
#     matchesMask = None
#
# draw_params = dict(matchColor = (0,255,0), # draw matches in green color
#                    singlePointColor = None,
#                    matchesMask = matchesMask, # draw only inliers
#                    flags = 2)
# img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
# plt.imshow(img3, 'gray'),plt.show()