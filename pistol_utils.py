import cv2
import numpy as np
from matplotlib.colors import hsv_to_rgb

import cvUtils
from cvUtils import avgBrightness
#from count_holes import count_holes
import time

def __getFirstPistolTargetShapeContour(contours, shape_area):
    """
    This method removes the contours that are iterated
    :param contours:
    :param shape_area:
    :return:
    """

    for i in range(len(contours)):
        cnt = contours[i]
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if (area < shape_area and area > shape_area / 10):
            return cnt

    return None

def cropImage(img, numberOfShapes, removeBG = True):
    dark_orange = (100, 0, 140)
    light_orange = (225, 255, 255)
    #print(img.size, img.dtype)
    hsv_originalImage = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_originalImage, dark_orange, light_orange)
    result = cv2.bitwise_and(img, img, mask=mask)
    result_gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)

    # thresholding to remove background from foreground objects

    ret, thresholded_img = cv2.threshold(result_gray, 10, 255, cv2.THRESH_BINARY)
    thresholded_img = cv2.medianBlur(thresholded_img, 15)
    # kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(11, 11))
    # thresholded_img = cv2.morphologyEx(thresholded_img, cv2.MORPH_CLOSE, kernel, iterations=3)

    thresholded_img = cv2.GaussianBlur(thresholded_img, (5, 5), 0)
    #cv2.imwrite("C:/Users/Abdallah Reda/Desktop/test_pistol/pistol_thresh"+ str(idx) +".JPG", thresholded_img)
    contours, hierar = cv2.findContours(thresholded_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img_copy = img.copy()

    contours, bndRects = cvUtils.sort_contours(contours)
    cv2.drawContours(img_copy, contours, -1, (255, 0, 255), 16)
    contours = list(contours)
    print(len(contours))
    #print(contours.shape)
    target_contours = []
    img_area = img.shape[0] * img.shape[1]
    if (numberOfShapes > 1):

        for i in range(numberOfShapes):
            contour = __getFirstPistolTargetShapeContour(contours, img_area / (numberOfShapes))
            i -= 1
            contours.remove(contour)
            target_contours.append(contour)
    else:
        contour = __getFirstPistolTargetShapeContour(contours, img_area / (numberOfShapes))
        target_contours.append(contour)

    if (len(target_contours) != numberOfShapes): raise Exception("Couldn't find all shapes")
    croppedOutImages = []
    positions_in_original_image = []

    for i in range(numberOfShapes):
        if(removeBG):
            imgCropped= cvUtils.__cropContourMaskingOutInfo(img, target_contours[i], 10, 0)
        else:imgCropped= cvUtils.cropContourFromImage(img, target_contours[i])
        croppedOutImages.append(imgCropped)

    cvUtils.drawBndRects(target_contours, img_copy)
    # cvUtils.createFigure(croppedOutImages)
    # cvUtils.createFigure([img, thresholded_img, img_copy],
    #                      ["original Image", "threshold_img", "Result"])
    return croppedOutImages, target_contours
'''
img = "C:/Users/Abdallah Reda/Desktop/test_pistol/DSC_0010 - Copy.JPG"
img = cv2.imread(img)
cropped, cont = __getThePistolTargetShapes(img,10,True)
#mask = cv2.drawContours(cropped[0], cont[0], -1, 0, cv2.FILLED)
#cv2.imwrite("C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/pistol/mask.png",mask)
#print(len(cropped))
for i,img in enumerate(cropped):
    #print(img.shape, img.dtype)
    #print(img)
    #mask = cvUtils.__cropContourMaskingOutInfo(img,cont[i])
    print("img", img.shape)
    #kernel = cv2.getStructuringElement(cv2.MORPH_DILATE,(19,19))
    #img = cv2.dilate(img,kernel,iterations = 1)
    cv2.imwrite("C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/pistol/test"+str(i)+".jpg",img)
    #cv2.imwrite("C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/pistol/test_nobg"+str(i)+".jpg",mask)
'''
'''
#img = cv2.imread("C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/pistol/test1.jpg", 0)
#cv2.imwrite("C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/pistol/test1_gray.jpg",img)
img = cv2.imread("C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/pistol/test1_gray.jpg", 0)
thresh = 170

bw1 = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)[1]
cv2.imwrite("C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/pistol/test1_bw1.jpg",bw1)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(19,19))
bw1 = cv2.erode(bw1,kernel,iterations = 1)
cv2.imwrite("C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/pistol/test1_bw1_eroded.jpg",bw1)

(_, bw2) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2.imwrite("C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/pistol/test1_bw2.jpg",bw2)

img = cv2.imread("C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/pistol/test1_gray_before.jpg", 0)

bw1_before = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)[1]
cv2.imwrite("C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/pistol/test1_bw1_before.jpg",bw1_before)

bw1_before = cv2.erode(bw1_before,kernel,iterations = 1)
cv2.imwrite("C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/pistol/test1_bw1_before_eroded.jpg",bw1_before)

(_, bw2_before) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2.imwrite("C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/pistol/test1_bw2_before.jpg",bw2_before)
cv2.imwrite("C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/pistol/test1_diff.jpg",cv2.subtract(bw1,bw1_before))

'''

def get_diff_align(bef_img, aft_img, idx=None):
    # before_image=cv2.imread("C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/pistol/cropped_11.jpg")
    # after_image=cv2.imread("C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/pistol/cropped_21.jpg")
    before_image=bef_img.copy()
    after_image=aft_img.copy()
    before_image = cv2.cvtColor(before_image, cv2.COLOR_GRAY2BGR)
    after_image = cv2.cvtColor(after_image, cv2.COLOR_GRAY2BGR)
    after_image_aligned, h = cvUtils.alignImages(after_image,before_image)
    after_image_aligned = cv2.cvtColor(after_image_aligned, cv2.COLOR_BGR2GRAY)
    before_image = cv2.cvtColor(before_image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("aft", after_image_aligned)
    # cv2.waitKey(0)
    eq = cv2.equalizeHist(after_image_aligned)
    cv2.imwrite("C:/Users/Abdelrahman Ezzat/Desktop/New folder/test_pistol/equalized_"+str(idx)+".jpg", eq)
    eq = cv2.Canny(after_image_aligned, 40, 120)
    cv2.imwrite("C:/Users/Abdelrahman Ezzat/Desktop/New folder/test_pistol/edged_"+str(idx)+".jpg", eq)
    ret, i1 = cv2.threshold(after_image_aligned, 10, 255, cv2.THRESH_BINARY)
    i1 = cv2.medianBlur(i1, 15)
    # kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(11, 11))
    # thresholded_img = cv2.morphologyEx(thresholded_img, cv2.MORPH_CLOSE, kernel, iterations=3)

    i1 = cv2.GaussianBlur(i1, (5, 5), 0)
    #i1 = cv2.Canny(before_image, 30,100)

    cnts1, _ = cv2.findContours(i1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    ret, i2 = cv2.threshold(before_image, 10, 255, cv2.THRESH_BINARY)
    i2 = cv2.medianBlur(i2, 15)
    # kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(11, 11))
    # thresholded_img = cv2.morphologyEx(thresholded_img, cv2.MORPH_CLOSE, kernel, iterations=3)

    i2 = cv2.GaussianBlur(i2, (5, 5), 0)

    cnts2, _ = cv2.findContours(i2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cnts2, _ = cv2.findContours(before_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts1.sort(key=cv2.contourArea)
    cnts2.sort(key=cv2.contourArea)
    #print(len(cnts1))
    im_copy = np.zeros_like(after_image_aligned)
    im_copy = cv2.cvtColor(im_copy, cv2.COLOR_GRAY2BGR)
    large_cnt1 = cnts1[-1]
    cv2.drawContours(im_copy, [large_cnt1], 0, (255, 0, 0), 3)
    #cv2.imshow('orig', im_copy)
    #cv2.waitKey(0)
    #im_copy = after_image_aligned.copy()
    large_cnt2 = cnts2[-1]
    cv2.drawContours(im_copy, [large_cnt2], 0, (0, 255, 0), 3)
    # cv2.imshow('new', im_copy)
    # cv2.waitKey(0)
    #large_cnt2 = scale_contour(cnts2[-2], 0.9)
    min_cnt = large_cnt1 if cv2.contourArea(large_cnt1) < cv2.contourArea(large_cnt2) else large_cnt2
    r1 = cvUtils.__cropContourMaskingOutInfo(after_image_aligned, min_cnt, 0, 255)
    r2 = cvUtils.__cropContourMaskingOutInfo(before_image, min_cnt, 0, 255)
    print('shapes',i1.shape,after_image_aligned.shape, r1.shape)
    #cv2.imshow("aft", r1)
    #cv2.waitKey(0)
    # cv2.imshow('orig', r2)
    # cv2.waitKey(0)
    # cv2.imshow('new', r1)
    # cv2.waitKey(0)
    #exit()

    after_image_aligned = r1
    before_image = r2
    #print(after_image_aligned.shape, h.shape)
    #img1 = cv2.imread("C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/pistol/cropped_21.jpg")
    #img2 = IDMatcher(img1, "C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/pistol/cropped_11.jpg")
    #cv2.imwrite("C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/pistol/cropped_21_edit.jpg",img2)
    diff = cv2.absdiff(after_image_aligned,before_image)
    cv2.imwrite("C:/Users/Abdelrahman Ezzat/Desktop/New folder/test_pistol/absdiff_"+ str(idx) +".jpg",diff)
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    #diff = cv2.threshold(diff,25,255,cv2.THRESH_BINARY)[1]
    diff = cv2.inRange(diff,50,140)
    #diff = cv2.threshold(diff,127, 255, cv2.THRESH_OTSU)[1]
    diff = cv2.morphologyEx(diff, cv2.MORPH_DILATE, k1, iterations=1)
    diff = cv2.GaussianBlur(diff, (5,5), 2)
    diff = cv2.threshold(diff,70,255,cv2.THRESH_BINARY)[1]
    cv2.imwrite("C:/Users/Abdelrahman Ezzat/Desktop/New folder/test_pistol/thresh_"+ str(idx) +".jpg",diff)
    diff = cv2.morphologyEx(diff, cv2.MORPH_ERODE, k1, iterations=1)
    # cv2.imshow('diff_eroded', diff)
    # cv2.waitKey(0)
    diff = cv2.morphologyEx(diff, cv2.MORPH_DILATE, k2, iterations=2)
    cv2.imwrite("C:/Users/Abdelrahman Ezzat/Desktop/New folder/test_pistol/diff_dilated_"+ str(idx) +".jpg",diff)
    # cv2.imshow('diff_dilated', diff)
    # cv2.waitKey(0)
    '''
    circles = cv2.HoughCircles(diff,cv2.HOUGH_GRADIENT,1,minDist=15, param1=118,param2=8,minRadius=3,maxRadius=25) # 10,15
    if circles is not None:
        print(circles)
        print("len,",len(circles[0]))
        output = diff.copy()
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(output, (x, y), r, (0, 0, 255), 2)
            #cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
        # show the output image
        #cv2.imshow("output", np.hstack([image, output]))
        #cv2.waitKey(0)
            #print("Dasdaoajoijnsss")
        cv2.imwrite("C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/pistol/diff_res.jpg",output)

        '''
    output = after_image_aligned.copy()
    output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
    op = cv2.connectedComponentsWithStats(diff, connectivity=8, ltype= cv2.CV_32S)
    centroids = op[3]
    bodies = op[2]
    #print(len(centroids))
    centroids = np.round(centroids).astype("int")
    #print("after", centroids)
    for i,c in enumerate(centroids):
        #print(c)
        area = bodies[i][4]
        width = bodies[i][2]
        height = bodies[i][3]
        print(area)
        #280-900 for (9,9) dilate kernel
        sx = c[0]-width//2
        sy = c[1]-height//2
        if(cv2.pointPolygonTest(min_cnt, (sx,sy), True) > 0):
            print("yes")
        score = 0
        if(800 <= area <= 1200): 
            cv2.circle(output, (c[0],c[1]), width//2, (0,0,255), 3)
            score+=1
    cv2.imwrite("C:/Users/Abdelrahman Ezzat/Desktop/New folder/test_pistol/res_"+str(idx) +".jpg",output)
    return output, score, after_image

def get_diff_pistol(img1, img2, index, thresh=10):
    #img1 = cv2.imread("C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/pistol/test4_before.jpg", 0)
    #img2 = cv2.imread("C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/pistol/test4.jpg", 0)
    print("img1 dtype",img1.dtype)
    img1 = np.array(img1, dtype=np.uint8)
    img2 = np.array(img2, dtype=np.uint8)
    print("img1 dtype",img1.dtype)
    if(len(img1.shape)>2):
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if(len(img2.shape)>2):
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('img1',img1)
    #cv2.waitKey(0)
    #cv2.imshow('img2',img2)
    #cv2.waitKey(0)
    max_shape = max(img1.shape, img2.shape)[::-1]
    img1 = cv2.resize(img1, max_shape)
    img2 = cv2.resize(img2, max_shape)
    diff = cv2.absdiff(img2,img1)
    cv2.imwrite("C:/Users/Abdallah Reda/Desktop/test_pistol/diff_gray"+str(index)+".jpg",diff)
    bw1 = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)[1]
    cv2.imwrite("C:/Users/Abdallah Reda/Desktop/test_pistol/diff_bin"+str(index)+".jpg",bw1)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(19,19))
    bw1 = cv2.dilate(bw1,kernel,iterations = 1)
    cv2.imwrite("C:/Users/Abdallah Reda/Desktop/test_pistol/diff_bin_dilated"+str(index)+".jpg",bw1)

    circles = cv2.HoughCircles(bw1,cv2.HOUGH_GRADIENT,1,minDist=15, param1=118,param2=8,minRadius=9,maxRadius=17 , ) # 10,15
    if circles is not None:
        output = img2.copy()
        
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(output, (x, y), r, (0, 0, 255), 2)
            #cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
        # show the output image
        #cv2.imshow("output", np.hstack([image, output]))
        #cv2.waitKey(0)
            #print("Dasdaoajoijnsss")

    print("circles = ", len(circles))
    cv2.imwrite("C:/Users/Abdallah Reda/Desktop/test_pistol/result"+str(index)+".jpg", output) #np.hstack([image, output]))
img1 = "C:\\Users\\Abdelrahman Ezzat\\Desktop\\project_vc\\results\\results2\\pistol1\\result1.jpg"
img1 = cv2.imread(img1)
imgs1 = cropImage(img1, 5)
i=1
for image in imgs1[0]:
    #print(pair[0].shape, pair[1].shape)
    cv2.imwrite("C:/Users/Abdelrahman Ezzat/Desktop/New folder/test_pistol/cropped_abd_"+str(i)+".jpg", image) #np.hstack([image, output]))
    #cv2.imwrite("C:/Users/Abdelrahman Ezzat/Desktop/New folder/test_pistol/cropped_2"+str(i)+".jpg", pair[1]) #np.hstack([image, output]))
    #print(imgs2[1][i-1] - imgs1[1][i-1])
    #get_diff_align(*pair, i)
    i+=1
    #break
'''
img1 = "C:\\Users\\Abdelrahman Ezzat\\Desktop\\project_vc\\results\\pistol6\\3_before.jpg"
img2 = "C:\\Users\\Abdelrahman Ezzat\\Desktop\\project_vc\\results\\pistol6\\3_after.jpg"
img1 = cv2.imread(img1,0)
img2 = cv2.imread(img2,0)
get_diff_align(img1, img2,4)
'''
'''
#print(img.shape)
imgs2 = cropImage(img2, 5)
imgs1 = cropImage(img1, 5)
i=1
for pair in zip(imgs1[0], imgs2[0]):
    print(pair[0].shape, pair[1].shape)
    cv2.imwrite("C:/Users/Abdelrahman Ezzat/Desktop/New folder/test_pistol/cropped_1"+str(i)+".jpg", pair[0]) #np.hstack([image, output]))
    cv2.imwrite("C:/Users/Abdelrahman Ezzat/Desktop/New folder/test_pistol/cropped_2"+str(i)+".jpg", pair[1]) #np.hstack([image, output]))
    #print(imgs2[1][i-1] - imgs1[1][i-1])
    get_diff_align(*pair, i)
    i+=1
    #break
#get_diff_pistol(10)

'''