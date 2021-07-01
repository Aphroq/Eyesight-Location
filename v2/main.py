import os

if __name__ == '__main__':
    print('-------------正在拆分帧-------------')
    os.system("python ./video.py")
    print('-------------正在分析帧-------------')
    os.system("python ./correction.py")
    print('-------------正在导出数据-----------')
    os.system("python ./location.py")