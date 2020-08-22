import cv2
import imutils
import numpy as np
from draw_circles import draw_circles
'''
template1 = cv2.imread("C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/pistol/cropped_21.jpg")
template1 = cv2.cvtColor(template1, cv2.COLOR_BGR2GRAY)
template = cv2.Canny(template1, 50, 200)
(tH, tW) = template.shape[:2]
#cv2.imshow("Template", template)
#cv2.waitKey(0)
image = cv2.imread("C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/pistol/cropped_11.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
found = None
for scale in np.linspace(0.2, 1.0, 20)[::-1]:
    # resize the image according to the scale, and keep track
    # of the ratio of the resizing
    resized = imutils.resize(gray, width = int(gray.shape[1] * scale))
    #print(resized.shape, gray.shape)
    r = gray.shape[1] / float(resized.shape[1])
    # if the resized image is smaller than the template, then break
    # from the loop
    if resized.shape[0] < tH or resized.shape[1] < tW:
        print("yes")
        break
    edged = cv2.Canny(resized, 50, 200)
    result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
    (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
    clone = np.dstack([edged, edged, edged])
    cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
    (maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
    #cv2.imshow("Visualize", clone)
    #cv2.waitKey(0)
    # if we have found a new maximum correlation value, then update
    # the bookkeeping variable
    if found is None or maxVal > found[0]:
        found = (maxVal, maxLoc, r)

(_, maxLoc, r) = found
(startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
(endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
# draw a bounding box around the detected result and display the image
img2 = image[startY:endY, startX:endX]
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
print(template1.shape, img2.shape)
cv2.imwrite("C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/pistol/cropped_11_edit.jpg", img2)
cv2.imwrite("C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/pistol/cropped_11_diff.jpg", cv2.subtract(img2,template1))
cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
#cv2.imshow("Image", image)
'''


#cv2.waitKey(0)

from ImportLib import *
from HelperFunc import *
from IDNumberParser import *
import time

def IDMatcher(image, refImagePath):
        

    MIN_MATCH_COUNT= 20
    detector=cv2.xfeatures2d.SIFT_create()

    FLANN_INDEX_KDITREE=0
    flannParam=dict(algorithm=FLANN_INDEX_KDITREE,tree=5)
    flann=cv2.FlannBasedMatcher(flannParam,{})
    
    #refImagePath = "C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/REF@.jpg"
    trainImg  = cv2.imread(refImagePath,0)
    trainKP,trainDesc=detector.detectAndCompute(trainImg,None)
    
    QueryImgBGR = image
    QueryImg=cv2.cvtColor(QueryImgBGR, cv2.COLOR_BGR2GRAY)
    queryKP,queryDesc=detector.detectAndCompute(QueryImg,None)
    matches=flann.knnMatch(queryDesc,trainDesc,k=2)
    goodMatch=[]

    for m,n in matches:
        if(m.distance<0.75*n.distance):
            goodMatch.append(m)
    print("good matches:", len(goodMatch))
    if(len(goodMatch)>MIN_MATCH_COUNT):
        tp=[]
        qp=[]
        for m in goodMatch:
            tp.append(trainKP[m.trainIdx].pt)
            qp.append(queryKP[m.queryIdx].pt)
        tp,qp=np.float32((tp,qp))
        H,status=cv2.findHomography(tp,qp,cv2.RANSAC,3.0)
        h,w=trainImg.shape
        trainBorder=np.float32([[[0,0],[0,h-1],[w-1,h-1],[w-1,0]]])
        queryBorder=cv2.perspectiveTransform(trainBorder,H)
    else:
        raise Exception('Not Enough match found,Make sure its an ID card and recapture the image again')
        
    Edged = four_point_transform(QueryImgBGR, queryBorder.reshape(4, 2))
    Edged_Resized = cv2.resize(Edged,trainImg.shape[::-1])
    return Edged_Resized
MAX_FEATURES = 10000
GOOD_MATCH_PERCENT = 0.99
def alignImages(im1, im2):
    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    cv2.imwrite("matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    if (h is None):
        return im1, h
    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))

    return im1Reg, h
def get_center(cnt):
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    return (cx,cy)
def scale_contour(cnt, scale):
    cx, cy = get_center(cnt)

    cnt_norm = cnt - [cx, cy]
    cnt_scaled = cnt_norm * scale
    cnt_scaled = cnt_scaled + [cx, cy]
    cnt_scaled = cnt_scaled.astype(np.int32)

    return cnt_scaled
def translate_image(image, cx, cy):
    height,width = image.shape
    T = np.float32([[1, 0, cx], [0, 1, cy]])
    
    # We use warpAffine to transform 
    # the image using the matrix, T 
    img_translation = cv2.warpAffine(image, T, (width, height)) 
    return img_translation

def __cropContourMaskingOutInfo(img, cnt, margin=0):
    contour = cnt.copy()
    x, y, w, h = cv2.boundingRect(contour)
    #print(x,y,w,h)
    ROI = img[y - margin : y + h + margin, x - margin : x + w + margin]
    # cv2.imshow("roi",ROI)
    # cv2.waitKey(0)
    mask = np.ones(ROI.shape)
    contour-=[x-margin,y-margin]
    mask = cv2.drawContours(mask, [contour], -1, 0, cv2.FILLED)
    #cv2.fillConvexPoly(mask, contour, (0,0,0))
    # Generate output
    output = ROI.copy()
    #print(output.shape, mask.shape)
    output[mask.astype(np.bool)] = 255
    #print("op",output.shape)
    return output

def get_diff(before_path, after_path, idx):
    # trainImg=cv2.imread("C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/pistol/cropped_11.jpg")
    # queryImg=cv2.imread("C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/pistol/cropped_21.jpg")
    trainImg=cv2.imread(before_path)
    queryImg=cv2.imread(after_path)
    im1Reg, h = alignImages(queryImg,trainImg)
    im1Reg = cv2.cvtColor(im1Reg, cv2.COLOR_BGR2GRAY)
    trainImg = cv2.cvtColor(trainImg, cv2.COLOR_BGR2GRAY)

    ret, i1 = cv2.threshold(im1Reg, 10, 255, cv2.THRESH_BINARY)
    i1 = cv2.medianBlur(i1, 15)
    # kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(11, 11))
    # thresholded_img = cv2.morphologyEx(thresholded_img, cv2.MORPH_CLOSE, kernel, iterations=3)

    i1 = cv2.GaussianBlur(i1, (5, 5), 0)

    cnts1, _ = cv2.findContours(i1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    ret, i2 = cv2.threshold(trainImg, 10, 255, cv2.THRESH_BINARY)
    i2 = cv2.medianBlur(i2, 15)
    # kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(11, 11))
    # thresholded_img = cv2.morphologyEx(thresholded_img, cv2.MORPH_CLOSE, kernel, iterations=3)

    i2 = cv2.GaussianBlur(i2, (5, 5), 0)

    cnts2, _ = cv2.findContours(i2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cnts2, _ = cv2.findContours(trainImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts1.sort(key=cv2.contourArea)
    cnts2.sort(key=cv2.contourArea)
    #print(len(cnts1))
    im_copy = np.zeros_like(im1Reg)
    im_copy = cv2.cvtColor(im_copy, cv2.COLOR_GRAY2BGR)
    large_cnt1 = cnts1[-1]
    cv2.drawContours(im_copy, [large_cnt1], 0, (255, 0, 0), 3)
    #cv2.imshow('orig', im_copy)
    #cv2.waitKey(0)
    #im_copy = im1Reg.copy()
    large_cnt2 = cnts2[-1]
    cv2.drawContours(im_copy, [large_cnt2], 0, (0, 255, 0), 3)
    # cv2.imshow('new', im_copy)
    # cv2.waitKey(0)
    #large_cnt2 = scale_contour(cnts2[-2], 0.9)
    min_cnt = large_cnt1 if cv2.contourArea(large_cnt1) < cv2.contourArea(large_cnt2) else large_cnt2
    r1 = __cropContourMaskingOutInfo(im1Reg, min_cnt, 10, 255)
    r2 = __cropContourMaskingOutInfo(trainImg, min_cnt, 10, 255)
    # cv2.imshow('orig', r2)
    # cv2.waitKey(0)
    # cv2.imshow('new', r1)
    # cv2.waitKey(0)
    #exit()

    im1Reg = r1
    trainImg = r2
    #print(im1Reg.shape, h.shape)
    #img1 = cv2.imread("C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/pistol/cropped_21.jpg")
    #img2 = IDMatcher(img1, "C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/pistol/cropped_11.jpg")
    #cv2.imwrite("C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/pistol/cropped_21_edit.jpg",img2)
    diff = cv2.absdiff(im1Reg,trainImg)
    cv2.imwrite("C:/Users/Abdallah Reda/Desktop/test_pistol/absdiff_"+ str(idx) +".jpg",diff)
    diff = cv2.threshold(diff,25,255,cv2.THRESH_BINARY)[1]
    diff = cv2.GaussianBlur(diff, (5,5), 2)
    diff = cv2.threshold(diff,70,255,cv2.THRESH_BINARY)[1]
    cv2.imwrite("C:/Users/Abdallah Reda/Desktop/test_pistol/thresh_"+ str(idx) +".jpg",diff)
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    diff = cv2.morphologyEx(diff, cv2.MORPH_ERODE, k1, iterations=1)
    # cv2.imshow('diff_eroded', diff)
    # cv2.waitKey(0)
    diff = cv2.morphologyEx(diff, cv2.MORPH_DILATE, k2, iterations=2)
    cv2.imwrite("C:/Users/Abdallah Reda/Desktop/test_pistol/diff_dilated_"+ str(idx) +".jpg",diff)
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
    output = im1Reg.copy()
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
        print(area)
        #280-900 for (9,9) dilate kernel
        if(800 <= area <= 1200): 
            cv2.circle(output, (c[0],c[1]), width//2, (0,0,255), 3)
    cv2.imwrite("C:/Users/Abdallah Reda/Desktop/test_pistol/res_"+str(idx) +".jpg",output)

def get_diff_ak(before_path, after_path, idx):
    # trainImg=cv2.imread("C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/pistol/cropped_11.jpg")
    # queryImg=cv2.imread("C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/pistol/cropped_21.jpg")
    trainImg=cv2.imread(before_path)
    queryImg=cv2.imread(after_path)
    im1Reg, h = alignImages(queryImg,trainImg)
    im1Reg = cv2.cvtColor(im1Reg, cv2.COLOR_BGR2GRAY)
    trainImg = cv2.cvtColor(trainImg, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
    i1 = cv2.morphologyEx(im1Reg, cv2.MORPH_ERODE, kernel,iterations=3)
    i1 = cv2.morphologyEx(i1, cv2.MORPH_DILATE, kernel,iterations=4)
    i1 = cv2.threshold(i1, 85, 255, cv2.THRESH_BINARY)[1] #to be parameterized
    i1 = cv2.Canny(i1,30,100)
    #i1 = cv2.Canny(im1Reg, 30, 90)
    
    cnts1, _ = cv2.findContours(i1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #i2 = cv2.Canny(trainImg, 30, 90)   #10 for pistol
    i2 = cv2.morphologyEx(trainImg, cv2.MORPH_ERODE, kernel,iterations=3)
    i2 = cv2.morphologyEx(i2, cv2.MORPH_DILATE, kernel,iterations=4)
    i2 = cv2.threshold(i2, 85, 255, cv2.THRESH_BINARY)[1] #to be parameterized
    i2 = cv2.Canny(i2,30,100)
    #cv2.imshow("i1",i1)
    #cv2.imshow("i2",i2)
    #cv2.waitKey()
    cnts2, _ = cv2.findContours(i2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cnts2, _ = cv2.findContours(trainImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts1.sort(key=cv2.contourArea)
    cnts2.sort(key=cv2.contourArea)
    #print(len(cnts1))
    im_copy = np.zeros_like(im1Reg)
    im_copy = cv2.cvtColor(im_copy, cv2.COLOR_GRAY2BGR)
    large_cnt1 = cnts1[-2]
    cv2.drawContours(im_copy, [large_cnt1], 0, (255, 0, 0), 3)
    #cv2.imshow('orig', im_copy)
    #cv2.waitKey(0)
    #im_copy = im1Reg.copy()
    large_cnt2 = cnts2[-2]
    cv2.drawContours(im_copy, [large_cnt2], 0, (0, 255, 0), 3)
    #cv2.imshow('new', im_copy)
    #cv2.waitKey(0)
    #large_cnt2 = scale_contour(cnts2[-2], 0.9)
    center1 = get_center(large_cnt1)
    center2 = get_center(large_cnt2)
    if cv2.contourArea(large_cnt1) < cv2.contourArea(large_cnt2):
        min_cnt = large_cnt1
        cx,cy = center1[0]-center2[0], center1[1]-center2[1]
        trainImg = translate_image(trainImg, cx, cy)
    else:
        min_cnt = large_cnt2
        cx,cy = center2[0]-center1[0], center2[1]-center1[1]
        im1Reg = translate_image(im1Reg, cx, cy)
    #print(cx, cy)
    #cv2.imshow('new', im1Reg)
    #cv2.imshow('new', trainImg)
    #cv2.waitKey(0)
    r1 = __cropContourMaskingOutInfo(im1Reg, min_cnt, 10, 255)
    r2 = __cropContourMaskingOutInfo(trainImg, min_cnt, 10, 255)
    # cv2.imshow('orig', r2)
    # cv2.waitKey(0)
    # cv2.imshow('new', r1)
    # cv2.waitKey(0)
    #exit()

    im1Reg = r1
    trainImg = r2
    cv2.imwrite("C:/Users/Abdallah Reda/Desktop/test_ak/n1_"+ str(idx) +".jpg",r1)
    cv2.imwrite("C:/Users/Abdallah Reda/Desktop/test_ak/n2_"+ str(idx) +".jpg",r2)
    #print(im1Reg.shape, h.shape)
    #img1 = cv2.imread("C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/pistol/cropped_21.jpg")
    #img2 = IDMatcher(img1, "C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/pistol/cropped_11.jpg")
    #cv2.imwrite("C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/pistol/cropped_21_edit.jpg",img2)
    diff = cv2.subtract(im1Reg,trainImg)
    cv2.imwrite("C:/Users/Abdallah Reda/Desktop/test_ak/diff_align_"+ str(idx) +".jpg",diff)
    diff = cv2.threshold(diff,25,255,cv2.THRESH_BINARY)[1]
    diff = cv2.GaussianBlur(diff, (5,5), 2)
    diff = cv2.threshold(diff,70,255,cv2.THRESH_BINARY)[1]
    cv2.imwrite("C:/Users/Abdallah Reda/Desktop/test_ak/diff_thresh_"+ str(idx) +".jpg",diff)
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
    diff = cv2.morphologyEx(diff, cv2.MORPH_ERODE, k1)
    # cv2.imshow('diff_eroded', diff)
    # cv2.waitKey(0)
    diff = cv2.morphologyEx(diff, cv2.MORPH_DILATE, k2, iterations=2)
    cv2.imwrite("C:/Users/Abdallah Reda/Desktop/test_ak/diff_dilated"+ str(idx) +".jpg",diff)
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
    output = im1Reg.copy()
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
        print(area)
        #280-900 for (9,9) dilate kernel
        if(600 <= area <= 900): 
            cv2.circle(output, (c[0],c[1]), width//2, (0,0,255), 3)
    cv2.imwrite("C:/Users/Abdallah Reda/Desktop/test_ak/res_"+str(idx) +".jpg",output)
    return output

# save_path = "C:/Users/Abdallah Reda/Desktop/test_pistol/"
# for i in range(1,11):
#     bef = save_path+"cropped_1"+str(i)+".jpg"
#     aft = save_path+"cropped_2"+str(i)+".jpg"
#     get_diff(bef, aft, i)

bef = "C:\\Users\\Abdallah Reda\\Downloads\\CVC-19-Documnet-Wallet-\\BackEnd\\visionapp\\Natinal_ID\\158\\friday14-8\\1"
aft = "C:\\Users\\Abdallah Reda\\Downloads\\CVC-19-Documnet-Wallet-\\BackEnd\\visionapp\\Natinal_ID\\158\\friday14-8\\2.jpg"
for i in range(1,5):
    #bef = process(ss)
    op = get_diff_ak(bef+"_"+str(i)+".jpg",aft,55+i)
    op = draw_circles(op)
    cv2.imwrite("C:/Users/Abdallah Reda/Desktop/test_ak/res_with_circles_"+str(55+i) +".jpg", op)