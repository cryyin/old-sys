# -*- coding: utf-8 -*-
'''
摔倒检测模型主程序

用法：
python testingfalldetection.py
python testingfalldetection.py --filename tests/corridor_01.avi
'''

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from imutils.object_detection import non_max_suppression
import numpy as np
import cv2
import time
import argparse

# 传入参数
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--filename", required=False, default = '',help="")
args = vars(ap.parse_args())
input_video = args['filename']

# 控制陌生人检测
fall_timing = 0 # 计时开始
fall_start_time = 0 # 开始时间
fall_limit_time = 1 # if >= 1 seconds, then he/she falls.

# 全局变量
model_path = 'models/fall_detection.hdf5'

# 全局常量
TARGET_WIDTH = 64
TARGET_HEIGHT = 64

# 初始化摄像头
if not input_video:
	vs = cv2.VideoCapture(0)
	time.sleep(2)
else:
	vs = cv2.VideoCapture(input_video)

# 加载模型
model = load_model(model_path)


hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

print('[INFO] 开始检测是否有人摔倒...')
# 不断循环
counter = 0
while True:
    counter += 1
    # grab the current frame
    (grabbed, image) = vs.read()

	# if we are viewing a video and we did not grab a frame, then we
	# have reached the end of the video
    if input_video and not grabbed:
        break

    if not input_video:
        image = cv2.flip(image, 1)

    roi = cv2.resize(image, (TARGET_WIDTH, TARGET_HEIGHT))
    roi = roi.astype("float") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)

    '''

    scale=0

    #canny 边缘检测
    image= image.copy()
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
    xgrad = cv2.Sobel(gray, cv2.CV_16SC1, 1, 0) #x方向梯度
    ygrad = cv2.Sobel(gray, cv2.CV_16SC1, 0, 1) #y方向梯度
    edge_output = cv2.Canny(xgrad, ygrad, 50, 150)
    # edge_output = cv2.Canny(gray, 50, 150)
    cv2.imshow("Canny Edge", edge_output)
    # edge_output = cv2.dilate(edge_output,cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8,3)),iterations=1)
    #背景减除
    fg = cv2.createBackgroundSubtractorMOG2()
    fgmask = fg.apply(edge_output)
    # cv2.imshow("fgmask", fgmask)

    #闭运算
    hline = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 4), (-1, -1)) #定义结构元素，卷积核
    vline = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 1), (-1, -1))
    result = cv2.morphologyEx(fgmask,cv2.MORPH_CLOSE,hline)#水平方向
    result = cv2.morphologyEx(result,cv2.MORPH_CLOSE,vline)#垂直方向
    cv2.imshow("result", result)


    # erodeim = cv2.erode(th,cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)),iterations=1)  # 腐蚀
    dilateim = cv2.dilate(result,cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4,4)),iterations=1) #膨胀
    # cv2.imshow("dilateimfgmask", dilateim)
    # dst = cv2.bitwise_and(image, image, mask= fgmask)
    # cv2.imshow("Color Edge", dst)
    #查找轮廓
    contours, hier = cv2.findContours(dilateim, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        if cv2.contourArea(c) > 1200:
            (x,y,w,h) = cv2.boundingRect(c)
            if scale==0:scale=-1;break
            scale = w/h
            cv2.putText(image, "scale:{:.3f}".format(scale), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.drawContours(image, [c], -1, (255, 0, 0), 1)
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),1)
            image = cv2.fillPoly(image, [c], (255, 255, 255)) #填充
            
            
    '''





    # determine facial expression
    (fall, normal) = model.predict(roi)[0]
    label = "Fall (%.2f)" %(fall) if fall > normal else "Normal (%.2f)" %(normal)

    # display the label and bounding box rectangle on the output frame
    cv2.putText(image, label, (image.shape[1] - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    cv2.imshow('Fall detection', image)

    # Press 'ESC' for exiting video
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break


vs.release()
cv2.destroyAllWindows()
