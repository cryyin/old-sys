# import the necessary packages
import imutils
import cv2
import  numpy as np


class AspectAwarePreprocessor_fall:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        # store the target image width, height, and interpolation
        # method used when resizing
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image):
        # grab the dimensions of the image and then initialize
        # the deltas to use when cropping
        (h, w) = image.shape[:2]
        dW = 0
        dH = 0

        # if the width is smaller than the height, then resize
        # along the width (i.e., the smaller dimension) and then
        # update the deltas to crop the height to the desired
        # dimension
        if w < h:
            image = imutils.resize(image, width=self.width,
                                   inter=self.inter)
            dH = int((image.shape[0] - self.height) / 2.0)

        # otherwise, the height is smaller than the width so
        # resize along the height and then update the deltas
        # crop along the width
        else:
            image = imutils.resize(image, height=self.height,
                                   inter=self.inter)
            dW = int((image.shape[1] - self.width) / 2.0)

        # now that our images have been resized, we need to
        # re-grab the width and height, followed by performing
        # the crop
        (h, w) = image.shape[:2]
        image = image[dH:h - dH, dW:w - dW]
        image = cv2.resize(image, (self.width, self.height),
                          interpolation=self.inter)

        '''fg = cv2.createBackgroundSubtractorMOG2()
        fgmask = fg.apply(image)
        cv2.imshow('fgmask',fgmask)
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

        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        img_contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
        img_contours = sorted(img_contours, key=cv2.contourArea)
        for i in img_contours:
            if cv2.contourArea(i) > 100:
                break
        mask = np.zeros(image.shape[:2], np.uint8)
        cv2.drawContours(mask, [i],-1, 255, -1)
        new_img = cv2.bitwise_and(image, image, mask=mask)
        cv2.imshow("new_img", new_img)
        #cv2.waitKey(0)

        new_img=edge_output

        # finally, resize the image to the provided spatial
        # dimensions to ensure our output image is always a fixed
        # size
        return new_img
