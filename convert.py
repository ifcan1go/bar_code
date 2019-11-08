# encoding utf-8
import cv2
import numpy
import os
def convert_pic(img_path):
    for root, dirs, files in os.walk(img_path):
        for img_name in files:
            img = cv2.imread(os.path.join(img_path, img_name))
            width = img.shape[1]
            length = img.shape[0]
            if length<width:
                img.transpose(1,0,2)
                print (img_name)
            cv2.imwrite(os.path.join(img_path, img_name), img)


