import cv2
import numpy as np

def HarrisDetect(img):
    # 转换成灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 高斯模糊
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # 图像转换为float32
    gray = np.float32(gray)

    # harris角点检测
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    print('harris_dst',dst)
    # 图像膨胀
    dst = cv2.dilate(dst, None)

    # 图像腐蚀
    dst = cv2.erode(dst, None)

    # 阈值设定
    img[dst > 0.0001*dst.max()] = [255, 0, 0]
    cv2.imshow('Harris', img)

    return img


# Harris角点
def HarrisDetect2(img):
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # find Harris corners
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.04)
    dst = cv2.dilate(dst,None)
    ret, dst = cv2.threshold(dst,0.001*dst.max(),255,0)
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

# Shi_Tomasi角点
def Shi_TomasiDetect(img):
    # 转换成灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
    print('shi_dst',corners)
    corners = np.int0(corners)
    print('shi_after_np',corners)
    dst = corners
    # 图像膨胀
    dst = cv2.dilate(dst, None)

    # 图像腐蚀
    dst = cv2.erode(dst, None)

    # 阈值设定
    img[dst > 0.0001*dst.max()] = [255, 0, 0]
    cv2.imshow('Shi_Tomasi', img)

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

    #print("左上坐标：", upLeftY, upLeftX)

    img[upRightY][upRightX][0] = 0
    img[upRightY][upRightX][1] = 255
    img[upRightY][upRightX][2] = 0

    #print("右上坐标：", upRightY, upRightX)

    img[downRightY][downRightX][0] = 0
    img[downRightY][downRightX][1] = 255
    img[downRightY][downRightX][2] = 0

    #print("右下坐标：", downRightY, downRightX)

    img[downLeftY][downLeftX][0] = 0
    img[downLeftY][downLeftX][1] = 255
    img[downLeftY][downLeftX][2] = 0

    #print("左下坐标：", downLeftY, downLeftX)
    corner = []
    corner.append([upLeftX,upLeftY])
    corner.append([upRightX,upRightY])
    corner.append([downLeftX,downLeftY])
    corner.append([downRightX,downRightY])
    print('corner',corner)
    # 图像膨胀
    img = cv2.dilate(img, None)

    # 描边
    cv2.line(img, (upLeftX, upLeftY), (upRightX, upRightY), (255, 0, 0), 3)
    cv2.line(img, (upRightX, upRightY), (downRightX, downRightY), (255, 0, 0), 3)
    cv2.line(img, (downRightX, downRightY), (downLeftX, downLeftY), (255, 0, 0), 3)
    cv2.line(img, (downLeftX, downLeftY), (upLeftX, upLeftY), (255, 0, 0), 3)

    cv2.imshow('result', img)



# 利用掩模排除多余的点
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

    cv2.imshow('mask_result', dst)

    return dst

# 导入图像
filename = 'Video/TestVideo/TestVideo328.jpg'
img = cv2.imread(filename)
imgCopy = img.copy()
myImg = img.copy()
# Harris角点检测
corner_img = HarrisDetect2(myImg)
#corner_img = Shi_TomasiDetect(myImg)
# 求掩模，剔掉异常点
#mask = getMask(imgCopy, corner_img) 	#调用第一个角点检测时，使用掩膜
# 求四个角点，标出
#pointDetect(mask, imgCopy)
pointDetect(corner_img, imgCopy)


cv2.waitKey(0)
cv2.destroyAllWindows()
