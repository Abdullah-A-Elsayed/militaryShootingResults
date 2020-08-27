import cv2
import numpy as np

# convertToOpenCVHSV():
#   converts from HSV range (H: 0-360, S: 0-100, V: 0-100)
#   to what OpenCV expects: (H: 0-179, S: 0-255, V: 0-255)
def convertToOpenCVHSV(H, S, V):
    return np.array([H // 2, S * 2.55, V * 2.55], np.uint8)

green = np.uint8([[[0, 255, 0]]]) #here insert the bgr values which you want to convert to hsv
hsvGreen = cv2.cvtColor(green, cv2.COLOR_BGR2HSV)
#print(hsvGreen[0][0])

lowerLimit = hsvGreen[0][0][0] - 10, 100, 100
upperLimit = hsvGreen[0][0][0] + 10, 255, 255

#print(upperLimit)
#print(lowerLimit)

# 1. Load input image
img = cv2.imread('C:/Users/Abdelrahman Ezzat/Desktop/project_vc/results/pistol1/1_full_after - Copy.jpg')

# 2. Preprocess: quantize the image to reduce the number of colors
div = 6
img = img // div * div + div // 2
cv2.imwrite('C:/Users/Abdelrahman Ezzat/Desktop/project_vc/results/pistol1/1_full_after_quantized.jpg', img)
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
low_yellow = np.array([0,0,198])
high_yellow = np.array([179,255,255])
yellow_seg_img = cv2.inRange(hsv_img, low_yellow, high_yellow)
cv2.imwrite('C:/Users/Abdelrahman Ezzat/Desktop/project_vc/results/pistol1/mask.jpg', yellow_seg_img)
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
yellow_seg_img = cv2.morphologyEx(yellow_seg_img, cv2.MORPH_CLOSE, k)
cv2.imwrite('C:/Users/Abdelrahman Ezzat/Desktop/project_vc/results/pistol1/mask_closed.jpg', yellow_seg_img)
print(hsv_img.shape, yellow_seg_img.shape)
cv2.imwrite('C:/Users/Abdelrahman Ezzat/Desktop/project_vc/results/pistol1/difference.jpg', cv2.subtract(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),yellow_seg_img))

'''
# 3. Convert to HSV color space
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


# 4. Segment the image using predefined values of yellow (min and max colors)
low_yellow = convertToOpenCVHSV(40, 35, 52)
high_yellow = convertToOpenCVHSV(56, 95, 93)
yellow_seg_img = cv2.inRange(hsv_img, low_yellow, high_yellow)
#cv2.imshow('yellow_seg_img', yellow_seg_img)
cv2.imwrite('lego4_yellow_seg_img.jpg', yellow_seg_img)

# 5. Identify and count the number of yellow bricks and create a mask with just the yellow objects
bricks_list = []
min_size = 5

contours, hierarchy = cv2.findContours(yellow_seg_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
for contourIdx, cnt in enumerate(contours):
    # filter out tiny segments
    x, y, w, h = cv2.boundingRect(cnt)
    if (w < min_size) or (h < min_size):
        continue

    #print('contourIdx=', contourIdx, 'w=', w, 'h=', h)

    bricks_list.append(cnt)

    # debug: draw green contour in the original image
    #cv2.drawContours(img, cnt, -1, (0, 255, 0), 2) # green

print('Detected', len(bricks_list), 'yellow pieces.')

# Iterate the list of bricks and draw them (filled) on a new image to be used as a mask
yellow_mask_img = np.zeros((img.shape[0], img.shape[1]), np.uint8)
for cnt in bricks_list:
    cv2.fillPoly(yellow_mask_img, pts=[cnt], color=(255,255,255))

cv2.imshow('yellow_mask_img', yellow_mask_img)
cv2.imwrite('lego5_yellow_mask_img.jpg', yellow_mask_img)

# debug: display only the original yellow bricks found
bricks_img = cv2.bitwise_and(img, img, mask=yellow_mask_img)
#cv2.imshow('bricks_img', bricks_img)
cv2.imwrite('lego5_bricks_img.jpg', bricks_img)

# 6. Identify holes in each Lego brick
diff_img = yellow_mask_img - yellow_seg_img
cv2.imshow('diff_img', diff_img)
cv2.imwrite('lego6_diff_img.jpg', diff_img)

# debug: create new BGR image for debugging purposes
dbg_img = cv2.cvtColor(yellow_mask_img, cv2.COLOR_GRAY2RGB)
#dbg_img = bricks_img

holes_list = []
min_area_size = 10
max_area_size = 24
contours, hierarchy = cv2.findContours(yellow_seg_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
for contourIdx, cnt in enumerate(contours):
    # filter out tiny segments by area
    area = cv2.contourArea(contours[contourIdx])

    if (area < min_area_size) or (area > max_area_size):
        #print('contourIdx=', contourIdx, 'w=', w, 'h=', h, 'area=', area, '(ignored)')
        #cv2.drawContours(dbg_img, cnt, -1, (0, 0, 255), 2) # red
        continue

    #print('contourIdx=', contourIdx, 'w=', w, 'h=', h, 'area=', area)
    holes_list.append(cnt)

# debug: draw a blue-ish contour on any BGR image to show the holes of the bricks
for cnt in holes_list:
    cv2.fillPoly(dbg_img, pts=[cnt], color=(255, 128, 0))
    cv2.fillPoly(img, pts=[cnt], color=(255, 128, 0))

cv2.imwrite('lego6_dbg_img.jpg', dbg_img)
cv2.imwrite('lego6_img.jpg', img)

# 7. Iterate though the list of holes and associate them with a particular brick
# TODO

cv2.imshow('img', img)
cv2.imshow('dbg_img', dbg_img)
cv2.waitKey(0)
'''