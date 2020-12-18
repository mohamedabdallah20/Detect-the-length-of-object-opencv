import cv2
import numpy as np
import imutils
from scipy.spatial.distance import euclidean
from imutils import perspective

# img = cv2.imread('len_8.6.jpg')
img = cv2.imread('len_8.8.jpg')
# img = cv2.imread('len_9.jpg')
# img = cv2.imread('len_7.2.jpg')
blue_length_cm = 8.8

def imgscal(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return img


img = imgscal(img, 30)

imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_red = np.array([(0, 150, 50)])
upper_red = np.array([10, 255, 255])
lower_blue = np.array([90, 50, 50])
upper_blue = np.array([150, 255, 255])

imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
redMask = cv2.inRange(imgHsv, lower_red, upper_red)
blueMask = cv2.inRange(imgHsv, lower_blue, upper_blue)
# -----------------------------------------------------------
def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


# -----------------------------------------------------------
# ----------------------------------------------------------------------------------------------
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(100,100))
blueCanny = cv2.Canny(blueMask, 1, 1)
redCanny = cv2.Canny(redMask, 1, 1)
blueCanny = cv2.dilate(blueCanny, kernel, iterations=1)
blueCanny = cv2.erode(blueCanny, kernel, iterations=1)
redCanny = cv2.dilate(redCanny, kernel, iterations=1)
redCanny = cv2.erode(redCanny, kernel, iterations=1)

blueCanny = cv2.morphologyEx(blueCanny, cv2.MORPH_CLOSE, kernel)
redCanny = cv2.morphologyEx(redCanny, cv2.MORPH_CLOSE, kernel)

blueCanny = cv2.Canny(blueCanny, 1, 1)
redCanny = cv2.Canny(redCanny, 1, 1)
#
# # ----------------------------------------------------------------------------------------------
blueContours, blueher = cv2.findContours(blueCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
redContours, redher = cv2.findContours(redCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

blueContours = [x for x in blueContours if cv2.contourArea(x) > 100]
redContours = [x for x in redContours if cv2.contourArea(x) > 100]

# areas = [cv2.contourArea(c) for c in blueContours]
# max_index  = np.argmax(areas)
# blueContours = blueContours[max_index]
#
# areas = [cv2.contourArea(c) for c in redContours]
# max_index  = np.argmax(areas)
# redContours = redContours[max_index]

blue_rect = cv2.minAreaRect(blueContours[0])
blue_box = cv2.boxPoints(blue_rect)
blue_box = np.array(blue_box)
blue_box = perspective.order_points(blue_box.astype(int))
(tl, tr, br, bl) = blue_box
tm = midpoint(tl, tr)
tb = midpoint(bl, br)
dist_in_pixel = euclidean(tm,tb)
dist_in_cm = blue_length_cm
pixel_per_cm = dist_in_cm/dist_in_pixel
# pixel_per_cm = dist_in_pixel / dist_in_cm
cv2.drawContours(img, [blue_box.astype(int)], 0, (0, 0, 0), 5)

for cnt in redContours:
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.array(box)
    box = perspective.order_points(box.astype(int))
    (tl, tr, br, bl) = box
    tm = midpoint(tl, tr)
    tb = midpoint(bl, br)
    wid = euclidean(tm, tb)*pixel_per_cm
    print(wid)
    cv2.drawContours(img, [box.astype("int")], -1, (0, 0, 255), 2)
    cv2.putText(img, "{:.1f}cm".format(wid), (int(tm[0] - 20), int(tb[1] - 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

for cnt in blueContours:
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.array(box)
    box = perspective.order_points(box.astype(int))
    (tl, tr, br, bl) = box
    tm = midpoint(tl, tr)
    tb = midpoint(bl, br)
    wid = euclidean(tm, tb) * pixel_per_cm
    print(wid)
    cv2.drawContours(img, [box.astype("int")], -1, (0, 0, 255), 2)
    cv2.putText(img, "{:.1f}cm".format(wid), (int(tm[0] - 15), int(tb[1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

cv2.imshow('red canny', redCanny)
cv2.imshow('blue canny', blueCanny)
cv2.imshow('original', img)
# cv2.imshow('gray', gray)
cv2.waitKey(0)
