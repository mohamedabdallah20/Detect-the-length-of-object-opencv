import cv2
import numpy as np
def empty(a):
    pass
cv2.namedWindow("trackBars")
cv2.resizeWindow("trackBars",500,200)
cv2.createTrackbar('HRed Min', "trackBars" , 0,10,empty)
cv2.createTrackbar('HRed max', "trackBars" , 10,10,empty)
cv2.createTrackbar('SRed Min', "trackBars" , 150,255,empty)
cv2.createTrackbar('SRed Max', "trackBars" , 255,255,empty)
cv2.createTrackbar('VRed Min', "trackBars" , 50,255,empty)
cv2.createTrackbar('VRed max', "trackBars" , 255,255,empty)

cv2.createTrackbar('HBlue Min', "trackBars" , 0,179,empty)
cv2.createTrackbar('HBlue max', "trackBars" , 179,179,empty)
cv2.createTrackbar('SBlue Min', "trackBars" , 0,255,empty)
cv2.createTrackbar('SBlue Max', "trackBars" , 255,255,empty)
cv2.createTrackbar('VBlue Min', "trackBars" , 0,255,empty)
cv2.createTrackbar('VBlue max', "trackBars" , 255,255,empty)

while True :
    img = cv2.imread('len_8.6.jpg')
    img = cv2.resize(img, (500, 600))
    imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


    hred_min= cv2.getTrackbarPos('HRed Min', "trackBars")
    hred_max = cv2.getTrackbarPos('HRed max', "trackBars")
    sred_min = cv2.getTrackbarPos('SRed Min', "trackBars" )
    sred_max = cv2.getTrackbarPos('SRed Max', "trackBars")
    vred_min = cv2.getTrackbarPos('VRed Min', "trackBars")
    vred_max = cv2.getTrackbarPos('VRed max', "trackBars")
    hblue_min = cv2.getTrackbarPos('HBlue Min', "trackBars")
    hblue_max = cv2.getTrackbarPos('HBlue Max', "trackBars")
    sblue_min = cv2.getTrackbarPos('SBlue Min', "trackBars")
    sblue_max = cv2.getTrackbarPos('SBlue Max', "trackBars")
    vblue_min = cv2.getTrackbarPos('VBlue Min', "trackBars")
    vblue_max = cv2.getTrackbarPos('VBlue Max', "trackBars")

    print(hblue_min, sblue_min, vblue_min,hblue_max, sblue_max, vblue_max,hred_min, sred_min, vred_min,hred_max, sred_max, vred_max)
    lower_blue = np.array([hblue_min, sblue_min, vblue_min])
    upper_blue = np.array([hblue_max, sblue_max, vblue_max])
    lower_red = np.array([(hred_min, sred_min, vred_min)])
    upper_red = np.array((hred_max, sred_max, vred_max))
    # imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    redMask = cv2.inRange(imgHsv, lower_red, upper_red)
    blueMask = cv2.inRange(imgHsv, lower_blue, upper_blue)
    # blueCanny = cv2.Canny(blueMask, 1, 1, )
    # redCanny = cv2.Canny(redMask, 1, 1, )
    cv2.imshow('original', imgHsv)
    cv2.imshow('red mask', redMask)
    cv2.imshow('blue mask', blueMask)
    cv2.waitKey(1)