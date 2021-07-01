import cv2 
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pylab
import countor


def Shi_Tomasi(img_path = 'Video/TestVideo/TestVideo326.jpg'):
    x_limit = 650
    y_limit = 1050
    mask = np.zeros((960, 1280), dtype=np.uint8)
    for x in range(50,x_limit):
        for y in range(0,y_limit):
            mask[x][y] = 255

    #img =cv2.imread('Video/TestVideo/TestVideo800.jpg')
 
    img =cv2.imread(img_path)
    imgCopy = img.copy()
    imgray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #harris角点检测图像需为float32
    gray=np.float32(imgray)
    dst=cv2.goodFeaturesToTrack(gray,70,0.003,10,mask=mask,blockSize=7,gradientSize=7)
    #dst=cv2.goodFeaturesToTrack(gray,200,0.01,10,useHarrisDetector=True,k=0.04)
    #dst=cv2.goodFeaturesToTrack(gray,200,0.004,10)
    #=cv2.goodFeaturesToTrack(gray,40,0.003,20,mask=mask,blockSize=7,gradientSize=7)

    res=np.int0(dst)
    #print(res)
    '''
    for i in res:
        x,y=i.ravel()
        cv2.circle(img,(x,y),3,255,-1)
    img=img[:,:,::-1]
    plt.imshow(img)
    pylab.show()
    '''

    ll =[ [38, 115],[1005,97], [128, 614], [960, 579] ]
    corn  = np.array(ll)
    print('set_countor',corn)
    correct_corn = []
    for y in corn:
        for x in res:
            x = x[0]
            d1=np.sqrt(np.sum(np.square(x-y)))
            if d1<30:
                correct_corn.append(x)

    print('correct_corn',correct_corn)


    upleft = None
    upright = None
    downleft = None
    downright = None
    ul_min = 10000000000
    ur_min = 10000000000
    dl_min = 10000000000
    dr_min = 10000000000

    for i in correct_corn:

        ul = np.array([300,260])
        dist_ul =  np.sqrt(np.sum(np.square(ul-i)))
        if dist_ul < ul_min:
            ul_min = dist_ul
            upleft = i
        
        ur = np.array([770,250])
        dist_ur =  np.sqrt(np.sum(np.square(ur-i)))
        if dist_ur < ur_min:
            ur_min = dist_ur
            upright = i

        dl = np.array([330,520])
        dist_dl =  np.sqrt(np.sum(np.square(dl-i)))
        if dist_dl < dl_min:
            dl_min = dist_dl
            downleft = i

        dr = np.array([760,500])
        dist_dr =  np.sqrt(np.sum(np.square(dr-i)))
        if dist_dr < dr_min:
            dr_min = dist_dr
            downright = i
    
    #print(ul_min,ur_min,dl_min,dr_min)
    z = [] 
    z.append(upleft)
    z.append(upright)
    z.append(downleft)
    z.append(downright)
    print('z',z)

    oneFlag = False
    for a in z:
        for b in z:
            print(a,b)
            c = (a==b)
            if c.all():
                print('continue',a,b)
                continue
            dist =  np.sqrt(np.sum(np.square(a-b)))
            print('dist:',dist)
            if dist<100:
                oneFlag = True
                print('onemore Cor in one Corner, their\'s dist:',dist)
                break
        if oneFlag:
            break
    #return z
    
    check_set = set()
    for x in z:
        check_set.add(tuple(x))
    if len(check_set)!=len(z):
        z = None
    if len(correct_corn)<4:
        z = None
    if oneFlag:
        z = None
    if z is not None:
        for i in z:
            x,y=i.ravel()
            cv2.circle(img,(x,y),3,255,-1)
        img=img[:,:,::-1]
        #plt.imshow(img)
        #pylab.show()

    return z
    
    
    


if __name__ == '__main__':
    Shi_Tomasi()