import cv2
import numpy as np
from matplotlib import pyplot as plt

def getCenter(img_file = './test_pics/u.jpg'):
    frame = cv2.imread(img_file)

    # Convert BGR color space to YUV color space
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)

    MedianBarPos = 21
    # Blur the image using the median blur algorithm
    yuvMedian = cv2.medianBlur(yuv, MedianBarPos)

    # Split the 3 channels
    y, img, v1 = cv2.split(yuvMedian)
    #cv2.imshow('y', y)
    #cv2.imshow('u', img)
    #cv2.imshow('v1', v1)

    #img = cv2.imread(img_file, 0)
    rows, cols = img.shape
    img_C = img.copy()

    edges = cv2.Canny(img, 50, 170, apertureSize=3)
    # 精度：取值范围0.1-2（默认值1）
    accuracy = 2
    # 阈值：取值范围1-255（小于200意义不大，默认值200）
    threshold = 200


    _, thresh= cv2.threshold(img, 150, 255, cv2.THRESH_TOZERO)
    # 霍夫圆变换
    # dp累加器分辨率与图像分辨率的反比默认1.5，取值范围0.1-10
    dp = 2
    # minDist检测到的圆心之间的最小距离。如果参数太小，则除了真实的圆圈之外，还可能会错误地检测到多个邻居圆圈。 如果太大，可能会错过一些圈子。取值范围10-500
    minDist = 50
    circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, dp, minDist, param1=80, param2=30, minRadius=0, maxRadius=40)
    try:
        circles = np.uint16(np.around(circles))
        #print(circles)
        for i in circles[0, :]:
            # 绘制外圆
            cv2.circle(img_C, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # 绘制圆心
            cv2.circle(img_C, (i[0], i[1]), 2, (0, 0, 255), 2)
        #cv2.imshow("img_C", img_C)
        #cv2.waitKey()
        return circles
    except:
        return None

if __name__ == '__main__':
    print(getCenter('./Temp/corrt129.jpg'))