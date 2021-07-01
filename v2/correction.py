import cv2
import math
import os
import numpy as np
import re
import shi

def getMask(img, corner_img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    ret, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.erode(mask, kernel)

    mask_inv = cv2.bitwise_not(mask)


    img1_bg = cv2.bitwise_and(img, img, mask=mask_inv)
    img2_fg = cv2.bitwise_and(corner_img, corner_img, mask=mask)

    dst = cv2.add(img1_bg, img2_fg)

    #cv2.imshow('mask_result', dst)

    return dst


def Perspective_transform(p):
    pts1 = np.float32([[p[0][0], p[0][1]], [p[1][0], p[1][1]], [p[2][0], p[2][1]], [p[3][0], p[3][1]]])
    dst = np.float32([[0, 0], [2048, 0], [0, 1152], [2048, 1152]])


    M = cv2.getPerspectiveTransform(pts1, dst)
    dstImage = cv2.warpPerspective(imgCopy, M,(2048, 1152))
    return dstImage

# Harris角点
def HarrisDetect(img):
    # 转换成灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 高斯模糊
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # 图像转换为float32
    gray = np.float32(gray)
    # harris角点检测
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    # 图像膨胀
    dst = cv2.dilate(dst, None)
    # 图像腐蚀
    dst = cv2.erode(dst, None)
    # 阈值设定
    img[dst > 0.0001*dst.max()] = [255, 0, 0]
    #cv2.imshow('Harris', img)
    return img

def HarrisDetect2(img):
    #img = cv2.imread(filename)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # find Harris corners
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.04)
    dst = cv2.dilate(dst,None)
    ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
    dst = np.uint8(dst)

    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)

    # Now draw them
    res = np.hstack((centroids,corners))
    res = np.int0(res)
    img[res[:,1],res[:,0]]=[255,0,0]
    #img[res[:,3],res[:,2]] = [0,255,0]

    cv2.imshow('Harris', img)

    return img

# 求顶点
def pointDetect(corner_img, img):
    # 求图像大小
    shape = img.shape
    height = shape[0]
    width = shape[1]

    upLeftX = 0
    upLeftY = 0
    downLeftX = 0
    downLeftY = 0
    upRightX = 0
    upRightY = 0
    downRightX = 0
    downRightY = 0

    # 求左上顶点
    for i in range(round(width/4),0,-1):
        for j in range(0, round(height/2)):
            if upLeftX == 0 and upLeftY == 0 and corner_img[j][i][0] == 255:
                upLeftX = i
                upLeftY = j
                break
        if upLeftX or upLeftY:
            break

    # 求右上顶点
    for i in range(  width-1,(round(width * (3/4))) ,-1):
        for j in range(0, round(height/2)):
            if upRightX == 0 and upRightY == 0 and corner_img[j][i][0] == 255:
                upRightX = i
                upRightY = j
                break
        if upRightX or upRightY:
            break

    # 求左下顶点
    for j in range(round(height*3/4),height-1):
        for i in range(round(width/4),0,-1):
            if downLeftX == 0 and downLeftY == 0 and corner_img[j][i][0] == 255:
                downLeftX = i
                downLeftY = j
                break
        if downLeftX or downLeftY:
            break

    # 求右下顶点
    for j in range(round(height/2),height-1): 
        for i in range( round(width/2),width-1):  
            if downRightX == 0 and downRightY == 0 and corner_img[j][i][0] == 255:
                downRightX = i
                downRightY = j
                break
        if downRightY or downRightY:
            break

    img[upLeftY][upLeftX][0] = 0
    img[upLeftY][upLeftX][1] = 255
    img[upLeftY][upLeftX][2] = 0

    print("左上坐标：", upLeftY, upLeftX)

    img[upRightY][upRightX][0] = 0
    img[upRightY][upRightX][1] = 255
    img[upRightY][upRightX][2] = 0

    print("右上坐标：", upRightY, upRightX)

    img[downRightY][downRightX][0] = 0
    img[downRightY][downRightX][1] = 255
    img[downRightY][downRightX][2] = 0

    print("右下坐标：", downRightY, downRightX)

    img[downLeftY][downLeftX][0] = 0
    img[downLeftY][downLeftX][1] = 255
    img[downLeftY][downLeftX][2] = 0

    print("左下坐标：", downLeftY, downLeftX)

    corner = []
    corner.append([upLeftX,upLeftY])
    corner.append([upRightX,upRightY])
    corner.append([downLeftX,downLeftY])
    corner.append([downRightX,downRightY])

    # 图像膨胀
    img = cv2.dilate(img, None)

    # 描边
    cv2.line(img, (upLeftX, upLeftY), (upRightX, upRightY), (255, 0, 0), 1)
    cv2.line(img, (upRightX, upRightY), (downRightX, downRightY), (255, 0, 0), 1)
    cv2.line(img, (downRightX, downRightY), (downLeftX, downLeftY), (255, 0, 0), 1)
    cv2.line(img, (downLeftX, downLeftY), (upLeftX, upLeftY), (255, 0, 0), 1)
    #cv2.imshow('result', img)
    return corner



if __name__ == "__main__":
    dir = './Video/TestVideo'
    imgList = os.listdir(dir)
    #print(imgList)
    imgList.sort(key=lambda x: int(x.replace("TestVideo", "").split('.')[0]))  # 按照数字进行排序后按顺序读取文件夹下的图片
    #print(imgList)
    before_p = None
    cnt_None = 0
    for count in range(0, len(imgList)):
        im_name = imgList[count]
        number = re.findall('\d+', im_name)[0]
        print(number)
        im_path = os.path.join(dir, im_name)
        print(im_path)
        
        inputDir = im_path
        
        img = cv2.imread(inputDir)
        imgCopy = img.copy()
        myImg = img.copy() 
        # Harris角点检测
        #corner_img = HarrisDetect2(myImg)

        # 求掩模，剔掉异常点
        #mask = getMask(imgCopy, corner_img)

        # 求四个角点，标出
        #p = pointDetect(corner_img, imgCopy)
        p = shi.Shi_Tomasi(im_path)
        #('len_p',len(p))
        if p is None:
            p = before_p
            cnt_None +=1
        dstImage = Perspective_transform(p)
        before_p = p
       
        common_name = "corrt"
        common_path = "Temp"+'/'
        cv2.imwrite(common_path+common_name+str(number)+'.jpg', dstImage)
        #cv2.waitKey(0)
    #cv2.waitKey(0)   #wait for a keyboard input
    #cv2.destroyAllWindows()
    print('use_before_frame_cor:',cnt_None)


