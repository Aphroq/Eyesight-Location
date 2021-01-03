import cv2
import math
import os
import numpy as np

def ImgInfo(inputDir):
    srcPic = cv2.imread(inputDir)
    length = srcPic.shape[0]
    depth = srcPic.shape[1]
    polyPic = srcPic
    shrinkedPic = srcPic
    # 读取图片信息

    blurred = cv2.GaussianBlur(srcPic, (3, 3), 0)
    gray = cv2.cvtColor(srcPic, cv2.COLOR_BGR2GRAY)
    xgrad = cv2.Sobel(gray, cv2.CV_16SC1, 1, 0)
    ygrad = cv2.Sobel(gray, cv2.CV_16SC1, 1, 0)
    cannyPic = cv2.Canny(gray, 50, 150)
    #二值化处理+高斯滤波+边缘提取

    #cv2.imshow("binary", cannyPic)
    return srcPic, length, depth, cannyPic,polyPic,shrinkedPic

def defClosure(cannyPic,length,depth):
    contours, hierarchy = cv2.findContours(cannyPic, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.imwrite('binary2.png', cannyPic)
    i = 0
    maxArea = 0
    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) > cv2.contourArea(contours[maxArea]):
            maxArea = i
    #检查面积最大的闭包
    hull = cv2.convexHull(contours[maxArea])
    s = [[1, 2]]
    z = [[2, 3]]
    for i in hull:
        s.append([i[0][0], i[0][1]])
        z.append([i[0][0], i[0][1]])
    del s[0]
    del z[0]
    #检查可能的角点

    # 限制四个角分别分布在图像的四等分的区间上，也就是矩形在图像中央
    # x y坐标值与原点差值的正负 判断属于哪一个区间

    center = [length / 2, depth / 2]


    for i in range(len(s)):
        s[i][0] = s[i][0] - center[0]
        s[i][1] = s[i][1] - center[1]
    one = []
    two = []
    three = []
    four = []

    for i in range(len(z)):
        if s[i][0] <= 0 and s[i][1] < 0:
            one.append(i)
        elif s[i][0] > 0 and s[i][1] < 0:
            two.append(i)
        elif s[i][0] >= 0 and s[i][1] > 0:
            four.append(i)
        else:
            three.append(i)
    #判断区间

    p = []
    distance = 0
    temp = 0

    for i in one:
        x = z[i][0] - center[0]
        y = z[i][1] - center[1]
        d = x * x + y * y
        if d > distance:
            temp = i
        distance = d
    p.append([z[temp][0], z[temp][1]])
    distance = 0
    temp = 0

    for i in two:
        x = z[i][0] - center[0]
        y = z[i][1] - center[1]
        d = x * x + y * y
        if d > distance:
            temp = i
        distance = d
    p.append([z[temp][0], z[temp][1]])
    distance = 0
    temp = 0
    for i in three:
        x = z[i][0] - center[0]
        y = z[i][1] - center[1]
        d = x * x + y * y
        if d > distance:
            temp = i
        distance = d
    p.append([z[temp][0], z[temp][1]])
    distance = 0
    temp = 0
    for i in four:
        x = z[i][0] - center[0]
        y = z[i][1] - center[1]
        d = x * x + y * y
        if d > distance:
            temp = i
        distance = d
    p.append([z[temp][0], z[temp][1]])

    # 寻找角点
    print(p)
    return p,contours,maxArea

def Perspective_transform(p):
    pts1 = np.float32([[p[0][0], p[0][1]], [p[1][0], p[1][1]], [p[2][0], p[2][1]], [p[3][0], p[3][1]]])
    dst = np.float32([[0, 0], [2048, 0], [0, 1152], [2048, 1152]])


    M = cv2.getPerspectiveTransform(pts1, dst)
    dstImage = cv2.warpPerspective(srcPic, M, (2048, 1152))
    return dstImage

if __name__ == "__main__":
    dir = 'E:\program\Video\TestVideo'
    imgList = os.listdir(dir)
    #print(imgList)
    imgList.sort(key=lambda x: int(x.replace("TestVideo", "").split('.')[0]))  # 按照数字进行排序后按顺序读取文件夹下的图片
    #print(imgList)
    for count in range(0, len(imgList)):
        im_name = imgList[count]
        im_path = os.path.join(dir, im_name)
        #print(im_path)
        inputDir = im_path
        srcPic, length, depth, cannyPic, polyPic, shrinkedPic = ImgInfo(inputDir)
        p, contours, maxArea = defClosure(cannyPic,length,depth)
        dstImage = Perspective_transform(p)

        for i in p:
            cv2.circle(polyPic, (i[0], i[1]), 2, (0, 255, 0), 2)

        black = np.zeros((shrinkedPic.shape[0], shrinkedPic.shape[1]), dtype=np.uint8)
        # 二值图转为三通道图
        black3 = cv2.merge([black, black, black])
        # black=black2
        cv2.drawContours(black, contours, maxArea, 255, 11)
        cv2.drawContours(black3, contours, maxArea, (255, 0, 0), 11)
        '''cv2.imwrite('cv.png', black)

        cv2.namedWindow("cannyPic", 0)
        cv2.imshow("cannyPic", black)
        cv2.namedWindow("shrinkedPic", 0)
        cv2.imshow("shrinkedPic", polyPic)
        cv2.namedWindow("dstImage", 0)
        cv2.imshow("dstImage", dstImage)'''
        common_name = "corrt"
        common_path = "Temp"+'/'
        cv2.imwrite(common_path+common_name+str(count)+'.jpg', dstImage)
        #cv2.waitKey(0)


