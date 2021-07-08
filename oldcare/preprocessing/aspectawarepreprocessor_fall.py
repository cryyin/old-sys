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
        cv2.waitKey(0)

        # finally, resize the image to the provided spatial
        # dimensions to ensure our output image is always a fixed
        # size
        return new_img
