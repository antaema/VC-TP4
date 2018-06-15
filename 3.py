# img – image to be analyzed, must be in grayscale and with float32 values.
# blockSize – size of the windows considered for the corner detection
# ksize – parameter for the derivative of Sobel
# k – free parameter for the Harris equation.

import numpy as np
from matplotlib import pyplot as plt
import cv2

cap = cv2.VideoCapture('calibration.mp4')
i=0
while(1):
    ret ,frame = cap.read()
    if ret == True:
        img = frame
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray,2,3,0.04)

        #result is dilated for marking the corners, not important
        dst = cv2.dilate(dst,None)
        img[dst>0.01*dst.max()]=[0,0,255]
        cv2.imshow('img',img)
        cv2.waitKey(0)
        i+=1
    else:
        break
cv2.destroyAllWindows()
cap.release()

 
