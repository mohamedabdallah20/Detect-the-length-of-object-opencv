import cv2
import numpy as np
import imutils
from scipy.spatial.distance import euclidean
from imutils import perspective
from imutils import contours

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

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)

edged = cv2.Canny(gray, 50, 100)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)
# find contours in the edge map
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
(cnts, _) = contours.sort_contours(cnts)
cnts = [x for x in cnts if cv2.contourArea(x) > 100]


# cv2.drawContours(gray,cnts,-1,(255,0,0),3)
# cv2.waitKey(1)
# cv2.destroyAllWindows()
#

# imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# lower_red = np.array([(0, 150, 50)])
# upper_red = np.array([10, 255, 255])
# lower_blue = np.array([90, 50, 50])
# upper_blue = np.array([150, 255, 255])
#
# imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# redMask = cv2.inRange(imgHsv, lower_red, upper_red)
# blueMask = cv2.inRange(imgHsv, lower_blue, upper_blue)


# -----------------------------------------------------------
def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


# -----------------------------------------------------------
# ----------------------------------------------------------------------------------------------
# blueCanny = cv2.Canny(blueMask, 1, 1)
# redCanny = cv2.Canny(redMask, 1, 1)
# blueCanny = cv2.dilate(blueCanny, None, iterations=1)
# blueCanny = cv2.erode(blueCanny, None, iterations=1)
# redCanny = cv2.dilate(redCanny, None, iterations=1)
# redCanny = cv2.erode(redCanny, None, iterations=1)
# ----------------------------------------------------------------------------------------------
# blueContours, blueher = cv2.findContours(blueCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# # blueContours = imutils.grab_contours(blueContours)
# redContours, redher = cv2.findContours(redCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# # redContours = imutils.grab_contours(redContours)
#
# blueContours = [x for x in blueContours if cv2.contourArea(x) > 500]
# redContours = [x for x in redContours if cv2.contourArea(x) > 500]


bluerect = cv2.minAreaRect(cnts[0])
bluebox = cv2.boxPoints(bluerect)
bluebox = np.array(bluebox)
bluebox = perspective.order_points(bluebox.astype(int))
(tl, tr, br, bl) = bluebox
tm = midpoint(tl, tr)
tb = midpoint(bl, br)
dist_in_pixel = euclidean(tm,tb)
dist_in_cm = blue_length_cm
pixel_per_cm = dist_in_pixel / dist_in_cm
cv2.drawContours(img, [bluebox.astype(int)], 0, (0, 0, 0), 5)

for cnt in cnts:
    if cv2.contourArea(cnt) < 100:
        continue
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.array(box)
    box = perspective.order_points(box.astype(int))
    (tl, tr, br, bl) = box
    tm = midpoint(tl, tr)
    tb = midpoint(bl, br)
    wid = euclidean(tm, tb)/pixel_per_cm
    print(wid)
    cv2.drawContours(img, [box.astype("int")], -1, (0, 0, 255), 2)
    cv2.putText(img, "{:.1f}cm".format(wid), (int(tm[0] - 15), int(tb[1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

# for cnt in blueContours:
#     if cv2.contourArea(cnt) < 100:
#         continue
#     rect = cv2.minAreaRect(cnt)
#     box = cv2.boxPoints(rect)
#     box = np.array(box)
#     box = perspective.order_points(box.astype(int))
#     (tl, tr, br, bl) = box
#     tm = midpoint(tl, tr)
#     tb = midpoint(bl, br)
#     wid = euclidean(tm, tb) / pixel_per_cm
#     print(wid)
#     cv2.drawContours(img, [box.astype("int")], -1, (0, 0, 255), 2)
#     cv2.putText(img, "{:.1f}cm".format(wid), (int(tm[0] - 15), int(tb[1] - 10)),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

cv2.imshow('original', img)
# cv2.imshow('blue canny', blueMask)
# cv2.imshow('red canny', redMask)
# cv2.imshow('gray', gray)
cv2.waitKey(0)
