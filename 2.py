# cv2.pyrMeanShiftFiltering(src, sp, sr[, dst[, maxLevel[, termcrit]]])
# src – The source 8-bit, 3-channel image.
# dst – The destination image of the same format and the same size as the source.
# sp – The spatial window radius.
# sr – The color window radius.
# maxLevel – Maximum level of the pyramid for the segmentation.
# termcrit – Termination criteria: when to stop meanshift iterations.

sp = 0
sr = 0
def update():
    cv2.destroyWindow('shifted')
    cv2.namedWindow('shifted', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('shifted', 600,600)
    cv2.moveWindow("shifted", 700,73)

    img_shifted = cv2.pyrMeanShiftFiltering(img, sp, sr)
    cv2.imshow('shifted', img_shifted)
    cv2.waitKey(0)

def updateP(value):
    global sp
    sp = value
    update()

def updateR(value):
    global sr
    sr = value
    update()

import numpy as np
import cv2 
img = cv2.imread('Spring.png')

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 600,600)
cv2.createTrackbar('SP', 'image', 0, 100, updateP)
cv2.createTrackbar('SR', 'image', 0, 100, updateR)
cv2.imshow('image', img)

update()
