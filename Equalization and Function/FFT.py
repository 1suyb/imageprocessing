import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import  *
from  tkinter import  filedialog as fd
from math import sqrt

class  FT:
    def __init__(self):
        self.filename = ""
    # 이미지 파일 가져오기
    def fileload(self,_title):
        self.filename = fd.askopenfilename(initialdir="/", title=_title)
        self.grayImage = cv2.imread(self.filename, cv2.IMREAD_GRAYSCALE)
        self.h, self.w = self.grayImage.shape

    # 점사이 거리
    def distance(self, point1,point2):
        return sqrt((point1[0] - point2[0])**2+(point1[1]-point2[1])**2)

    # matplotlib로 영상 두개 한번에 보여주기
    def show_two_image(self,_img1,_img2, _chimg1='gray', _chimg2='gray'):
        plt.subplot(121), plt.imshow(_img1, cmap=_chimg1)
        plt.subplot(122), plt.imshow(_img2, cmap=_chimg2)
        plt.show()
    # matplotlib로 영상 세개 한번에 보여주기
    def show_three_image(self,_img1,_img2,_img3, _chimg1='gray', _chimg2='gray', _chimg3='gray'):
        plt.subplot(131), plt.imshow(_img1, cmap=_chimg1)
        plt.subplot(132), plt.imshow(_img2, cmap=_chimg2)
        plt.subplot(133), plt.imshow(_img3, cmap=_chimg3)
        plt.show()

    # fft 변환후 원본이미지와 변환 이미지를 동시에 보여줌
    def fft(self):
        f = np.fft.fft2(self.grayImage)
        self.fftImage = np.fft.fftshift(f)
        fftspectrum = 20*np.log(np.abs(self.fftImage))
        self.show_two_image(self.grayImage, fftspectrum)

    # ifft복원 후 원본 이미지와 변환 이미지를 동시에 보여줌
    def ifft(self):
        f = np.fft.ifftshift(self.fftImage)
        f = np.fft.ifft2(f)
        self.ifftImage = np.abs(f)
        self.show_two_image(self.grayImage, self.ifftImage)

    # lpf 필터 적용후 원본 이미지와 변환 이미지를 동시에 보여줌.
    def lpf(self,_r = 30):
        ch, cw = int(self.h/2), int(self.w/2)
        self.lpfImage = self.fftImage.copy()
        for h in range(self.h):
            for w in range(self.w):
                if self.distance((h,w),(ch,cw))>_r:
                    self.lpfImage[h,w] = 0
        self.lpfImage = np.fft.ifftshift(self.lpfImage)
        self.lpfImage = np.fft.ifft2(self.lpfImage)
        self.lpfImage = np.abs(self.lpfImage)
        self.show_two_image(self.grayImage,self.lpfImage)


    # hpf 필터 적용후 원본 이미지와 변환 이미지를 동시에 보여줌줌
    def hpf(self):
        ch, cw = int(self.h/2), int(self.w/2)
        self.hpfImage = self.fftImage.copy()
        self.hpfImage[ch-30:ch+30, cw-30:cw+30] = 0
        self.hpfImage = np.fft.ifftshift(self.hpfImage)
        self.hpfImage = np.fft.ifft2(self.hpfImage)
        self.hpfImage = np.abs(self.hpfImage)
        self.show_two_image(self.grayImage,self.hpfImage)

    # butterworth_lpf 필터 적용후 원본이미지와 변환 이미지를 동시에 보여줌.
    def butterworth_lpf(self, _r=30, _n=1):
        H = self.fftImage.copy()
        center = (self.h/2,self.w/2)
        for h in range(self.h):
            for w in range(self.w):
                H[h,w] = H[h,w]*1/(1+(self.distance((h,w),center)/_r)**(2*_n))
        H = np.fft.ifftshift(H)
        H = np.fft.ifft2(H)
        self.blpfImage = np.abs(H)
        self.show_two_image(self.grayImage,self.blpfImage)





if __name__ == "__main__" :
    indoorft = FT()
    indoorft.fileload("IndoorImage")
    indoorft.fft()
    indoorft.ifft()
    indoorft.hpf()
    indoorft.lpf(10)
    indoorft.butterworth_lpf(10)

    outdoorft = FT()
    outdoorft.fileload("OutdoorImage")
    outdoorft.fft()
    outdoorft.ifft()
    outdoorft.hpf()
    outdoorft.lpf(10)
    outdoorft.butterworth_lpf(10)

# https://throwexception.tistory.com/823
# https://www.donike.net/frequency-domain-of-images-fourier-transform-and-filtering/