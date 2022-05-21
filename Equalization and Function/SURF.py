from __future__ import print_function
import cv2
import numpy as np
from tkinter import *
from tkinter import filedialog as fd
from matplotlib import pyplot as plt

class Utility :
    def fileLoad(self,_title) :
        filename = fd.askopenfilename(initialdir="/", title=_title)
        grayimage = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        return grayimage

class SURF :
    def __init__(self,minHessian = 400):
        self.objectimage = Utility.fileLoad(0,"Objectimage")
        self.sceneimage = Utility.fileLoad(0,"Sceneimage")
        self.SURFdetector = cv2.xfeatures2d_SURF.create(hessianThreshold=minHessian)

    def DAC(self):
        self.keypoints_obj, self.descriptors_obj = self.SURFdetector.detectAndCompute(self.objectimage,None)
        self.keypoints_scene, self.descriptors_scene = self.SURFdetector.detectAndCompute(self.sceneimage,None)

    def MatchingDescriptor(self,ratio_thresh = 0.75):
        matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
        knn_matches = matcher.knnMatch(self.descriptors_obj,self.descriptors_scene,2)
        self.good_matches = []
        for m,n in knn_matches :
            if m.distance < ratio_thresh * n.distance :
                self.good_matches.append(m)

    def Drawmathes(self):
        self.img_matches = np.empty(
            (max(self.objectimage.shape[0], self.sceneimage.shape[0]), self.objectimage.shape[1] + self.sceneimage.shape[1], 3), dtype=np.uint8)
        cv2.drawMatches(self.objectimage, self.keypoints_obj, self.sceneimage, self.keypoints_scene, self.good_matches, self.img_matches, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        obj = np.empty((len(self.good_matches),2), dtype=np.float32)
        scene = np.empty((len(self.good_matches), 2), dtype=np.float32)
        for i in range(len(self.good_matches)):
            obj[i, 0] = self.keypoints_obj[self.good_matches[i].queryIdx].pt[0]
            obj[i, 1] = self.keypoints_obj[self.good_matches[i].queryIdx].pt[1]
            scene[i, 0] = self.keypoints_scene[self.good_matches[i].trainIdx].pt[0]
            scene[i, 1] = self.keypoints_scene[self.good_matches[i].trainIdx].pt[1]
        H, _ = cv2.findHomography(obj, scene, cv2.RANSAC)
        obj_corners = np.empty((4, 1, 2), dtype=np.float32)
        obj_corners[0, 0, 0] = 0
        obj_corners[0, 0, 1] = 0
        obj_corners[1, 0, 0] = self.objectimage.shape[1]
        obj_corners[1, 0, 1] = 0
        obj_corners[2, 0, 0] = self.objectimage.shape[1]
        obj_corners[2, 0, 1] = self.objectimage.shape[0]
        obj_corners[3, 0, 0] = 0
        obj_corners[3, 0, 1] = self.objectimage.shape[0]
        self.scene_corners = cv2.perspectiveTransform(obj_corners, H)

    def Drawline(self):
        cv2.line(self.img_matches, (int(self.scene_corners[0, 0, 0] + self.objectimage.shape[1]), int(self.scene_corners[0, 0, 1])),
                (int(self.scene_corners[1, 0, 0] + self.objectimage.shape[1]), int(self.scene_corners[1, 0, 1])), (0, 255, 0), 4)
        cv2.line(self.img_matches, (int(self.scene_corners[1, 0, 0] + self.objectimage.shape[1]), int(self.scene_corners[1, 0, 1])),
                (int(self.scene_corners[2, 0, 0] + self.objectimage.shape[1]), int(self.scene_corners[2, 0, 1])), (0, 255, 0), 4)
        cv2.line(self.img_matches, (int(self.scene_corners[2, 0, 0] + self.objectimage.shape[1]), int(self.scene_corners[2, 0, 1])),
                (int(self.scene_corners[3, 0, 0] + self.objectimage.shape[1]), int(self.scene_corners[3, 0, 1])), (0, 255, 0), 4)
        cv2.line(self.img_matches, (int(self.scene_corners[3, 0, 0] + self.objectimage.shape[1]), int(self.scene_corners[3, 0, 1])),
                (int(self.scene_corners[0, 0, 0] + self.objectimage.shape[1]), int(self.scene_corners[0, 0, 1])), (0, 255, 0), 4)

    def ShowDetectMatches(self):
        plt.imshow(self.img_matches,'gray'),plt.show()

    def Surf(self):
        self.DAC()
        self.MatchingDescriptor()
        self.Drawmathes()
        self.Drawline()
        self.ShowDetectMatches()

if __name__ == "__main__" :
    surf = SURF()
    surf.Surf()