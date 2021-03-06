# -*- coding: utf-8 -*-
'''
测试情感分析模型

用法：
python testingfacialexpression.py
python testingfacialexpression.py --filename tests/room_04.avi
'''

# import the necessary packages
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from oldcare.facial import FaceUtil
import numpy as np
import imutils
import cv2
import time
import argparse

# 传入参数
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--filename", required=False, default='',
                help="")
args = vars(ap.parse_args())

# 全局变量
model_path = 'models/face_expression-adam.hdf5'
input_video = args['filename']
test_pic = 'E:/MyCode/python/OldCareSys/images/Surprise/177.jpg'

# 全局常量
FACIAL_EXPRESSION_TARGET_WIDTH = 32
FACIAL_EXPRESSION_TARGET_HEIGHT = 32

# load the face detector cascade and smile detector CNN
model = load_model(model_path)

# if a video path was not supplied, grab the reference to the webcam
if not input_video:
    camera = cv2.VideoCapture(0)
    time.sleep(2)
else:
    camera = cv2.VideoCapture(input_video)

faceutil = FaceUtil()

# keep looping
while True:
    # grab the current frame
    (grabbed, frame) = camera.read()

    # if we are viewing a video and we did not grab a frame, then we
    # have reached the end of the video
    if input_video and not grabbed:
        break

    if not input_video:
        frame = cv2.flip(frame, 1)

    # resize the frame, convert it to grayscale, and then clone the
    # original frame so we can draw on it later in the program
    frame = imutils.resize(frame, width=600)

    face_location_list = faceutil.get_face_location(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # loop over the face bounding boxes
    for (left, top, right, bottom) in face_location_list:
        # extract the ROI of the face from the grayscale image,
        # resize it to a fixed 28x28 pixels, and then prepare the
        # ROI for classification via the CNN
        roi = gray[top:bottom, left:right]
        roi = cv2.resize(roi, (FACIAL_EXPRESSION_TARGET_WIDTH,
                               FACIAL_EXPRESSION_TARGET_HEIGHT))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)





        test_img = cv2.imread(test_pic, cv2.IMREAD_COLOR)
        test_img = gray[top:bottom, left:right]
        test_img = cv2.resize(test_img, (FACIAL_EXPRESSION_TARGET_WIDTH, FACIAL_EXPRESSION_TARGET_HEIGHT))
        test_img = test_img.astype("float") / 255.0
        test_img = img_to_array(test_img)
        test_img = np.expand_dims(test_img, axis=0)
        #cv2.imshow('input_image', test_img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()









        # determine facial expression
        # result = model.predict(roi)[0]
        #result = model.predict(test_img)
        result = model.predict(roi)
        tag = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']
        print('result:' + str(np.argmax(result)))
        # label=str(np.argmax(result[0]))

        # display the label and bounding box rectangle on the output
        # frame
        cv2.putText(frame, tag[np.argmax(result)], (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(frame, (left, top), (right, bottom),
                      (0, 0, 255), 2)

    # show our detected faces along with smiling/not smiling labels
    cv2.imshow("Facial Expression Detect", frame)

    # Press 'ESC' for exiting video
    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
