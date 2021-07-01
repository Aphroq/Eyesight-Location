import cv2
import os
import csv
import json
import getCenter
# 载入并显示图片
dir = 'Temp'
imgList = os.listdir(dir)
#print(imgList)
imgList.sort(key=lambda x: int(x.replace("corrt", "").split('.')[0]))  # 按照数字进行排序后按顺序读取文件夹下的图片
#print(imgList)
coordinations = {}
for count in range(0, len(imgList)):
    im_name = imgList[count]
    im_path = os.path.join(dir, im_name)
    print(im_path)

    img = cv2.imread(im_path)
    #cv2.imshow('1', img)
    # 降噪（模糊处理用来减少瑕疵点）
    result = cv2.blur(img, (5, 5))
    #  cv2.imshow('2', result)
    # 灰度化,就是去色（类似老式照片）
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('3', gray)

    # 霍夫变换圆检测
    #circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 50, param1=80, param2=30, minRadius=1, maxRadius=200)
    circles = getCenter.getCenter(im_path)
    # 输出返回值，方便查看类型
    #data=open("E:\program\data.txt",'w+')
    print(circles)
    try:
        circle = circles.tolist()[0][0]
    #print('mycircle',circle)
    except:
        circle = [0,0]
    x = circle[0]
    y = circle[1]
    coordinations[count+1] = [x,y]
    with open('data.txt','a') as f:
        s = "{}:{},{}".format(str(count+1),str(x),str(y))
        f.write(s)
        f.write('\n')
    if count == 0:
        with open('data.csv','w') as f:
            head = ['帧','x坐标','y坐标']
            writer = csv.writer(f)
            writer.writerow(head)
    with open('data.csv','a') as f:
            row = [count+1,x,y]
            writer = csv.writer(f)
            writer.writerow(row)
    # 输出检测到圆的个数
    #print(len(circles[0]),file=data)
    print('------------------------------')
    print("loading...")
    #data.close()
    # 根据检测到圆的信息，画出每一个圆
    try:
        for circle in circles[0]:
            # 圆的基本信息
            print(circle[2])
            # 坐标行列(就是圆心)
            x = int(circle[0])
            y = int(circle[1])

            # 半径
            r = int(circle[2])
            # 在原图用指定颜色圈出圆，参数设定为int所以圈画存在误差
            img = cv2.circle(img, (x, y), r, (0, 0, 255), 1, 8, 0)
            # 显示新图像
            #cv2.imshow('getCenter', img)
            # 按任意键退出
            #cv2.waitKey(0)
    except:
        print('None')
    cv2.destroyAllWindows()
#cjson = json.dumps(coordinations)
#print(cjson)
with open("data.json", "w") as f:
    f.write(json.dumps(coordinations, ensure_ascii=False, indent=4))