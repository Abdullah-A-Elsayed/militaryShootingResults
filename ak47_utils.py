from ImportLib import *
from HelperFunc import *
from IDNumberParser import *
import cvUtils
from draw_circles import draw_circles
from cvUtils import *
import time

def IDMatcher(image, refImagePath):
        

    MIN_MATCH_COUNT= 20
    detector=cv2.xfeatures2d.SIFT_create()

    FLANN_INDEX_KDITREE=0
    flannParam=dict(algorithm=FLANN_INDEX_KDITREE,tree=5)
    flann=cv2.FlannBasedMatcher(flannParam,{})
    
    #refImagePath = "C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/REF@.jpg"
    before_image  = cv2.imread(refImagePath,0)
    trainKP,trainDesc=detector.detectAndCompute(before_image,None)
    
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
        h,w=before_image.shape
        trainBorder=np.float32([[[0,0],[0,h-1],[w-1,h-1],[w-1,0]]])
        queryBorder=cv2.perspectiveTransform(trainBorder,H)
    else:
        raise Exception('Not Enough match found,Make sure its an ID card and recapture the image again')
        
    Edged = four_point_transform(QueryImgBGR, queryBorder.reshape(4, 2))

    return Edged
def __getTheTargetImageSift(image, refImagePath):
    img2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    MIN_TARGET_MATCH_COUNT = 20

    # Initiate SIFT detector
    sifttime = time.time()
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    targetReferenceShape = cv2.imread(refImagePath,0)
    kp1, des1 = sift.detectAndCompute(targetReferenceShape, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    siftend = time.time()
    sifttime = siftend - sifttime
    print('sift time = ', sifttime)
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=1)
    search_params = dict(checks=6)
    flanntime = time.time()
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)
    flannend = time.time()
    flanntime = flannend - flanntime
    print('flann time : %f' % flanntime)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    if len(good) > MIN_TARGET_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h, w = targetReferenceShape.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    else:
        print("Not enough matches are found - %d/%d" % (len(good), MIN_TARGET_MATCH_COUNT))
        matchesMask = None
    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)

    img3 = cv2.drawMatches(targetReferenceShape, kp1, img2, kp2, good, None, **draw_params)
    ## (9) Crop the matched region from scene
    h, w = targetReferenceShape.shape[:2]
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)
    perspectiveM = cv2.getPerspectiveTransform(np.float32(dst), pts)
    found = cv2.warpPerspective(image, perspectiveM, (w, h))
    return found

def IDCutter(image, refImagePath, method=False):
    print("shape:",image.shape)
    Kernel = np.ones((2,2), np.uint8)
    Kernel_Vertical = np.ones((2,1), np.uint8)
    Kernel_sharpen = np.array([[-1,-1,-1], [-1, 9,-1],[-1,-1,-1]])
    if(method):
        Edged = __getTheTargetImageSift(image, refImagePath)
    else:
        Edged = IDMatcher(image, refImagePath)


    Edged_Resized = cv2.resize(Edged,(1400,1000))
    Gaussian = cv2.GaussianBlur(Edged_Resized,(29,29),2)
    Edged_Gray = cv2.cvtColor(Gaussian, cv2.COLOR_BGR2GRAY)
    Edged_Gray = cv2.fastNlMeansDenoising(Edged_Gray,10,10,7,21) 
    Edged_Gray = cv2.filter2D(Edged_Gray, -1, Kernel_sharpen)

    #dilate image to widen bullet holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(19,19))
    Edged_Gray_dilated = cv2.dilate(Edged_Gray,kernel,iterations = 1)
    ##cv2.imwrite(newImagePath, Edged_Gray_dilated)
    
    #convert to binary for processing
    #thresh = self.shooting_params.THRESH_BINARY
    #Edged_bin = cv2.threshold(Edged_Gray_dilated, thresh, 255, cv2.THRESH_BINARY)[1]
    #(_, Edged_Gray2) = cv2.threshold(Edged_Gray2, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    ##cv2.imwrite(newImagePath[:-4]+"_binary.jpg", Edged_bin)
    return Edged_Gray_dilated
    #return Edged_Gray2
    #Edged_Gray = cv2.dilate(Edged_Gray , )
    #Edged_Gray2 = cv2.morphologyEx(Edged_Gray, cv2.MORPH_CLOSE, kernel)
    #Edged_Gray2 = cv2.dilate(Edged_Gray , kernel)
    #Edged_Gray = cv2.dilate(Edged_Gray , )
    ##cv2.imwrite("C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/ResultOpen.jpg", Edged_Gray2)
    #Edged_Gray2 = cv2.threshold(Edged_Gray2, thresh, 255, cv2.THRESH_BINARY)[1]
    #(_, Edged_Gray2) = cv2.threshold(Edged_Gray2, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    ##cv2.imwrite(newImagePath[:-4]+"_binary1.jpg", Edged_Gray2)
    #cv2.imshow(Edged_Gray)
def process(img):
    # cv2.imshow("img",img)
    # cv2.waitKey(0)
    #img = cv2.imread(imgPath)
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    ##cv2.imwrite("C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/output/blue.jpg",gray)
    ## convert to hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ## mask of blue (140,70,65) ~ (210, 44, 35)
    mask_shapeblue = cv2.inRange(hsv, (75,0,0), (180, 255, 255))
    #mask_shapeblue = cv2.inRange(hsv, (0,25,0), (180, 255, 255))

    ## final mask and masked
    #mask = cv2.bitwise_or(mask1, mask2)
    mask_shape_inverted = cv2.bitwise_not(mask_shapeblue)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
    mask_shape_eroded = cv2.morphologyEx(mask_shape_inverted, cv2.MORPH_DILATE, kernel,iterations=1)
    mask_shape_eroded = cv2.morphologyEx(mask_shape_eroded, cv2.MORPH_ERODE, kernel,iterations=2)
    #cv2.imshow("mask",mask_shape_eroded)
    #cv2.waitKey(0)
    #crop white paper from whole image
    contours, hierarchy = cv2.findContours(mask_shape_eroded, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #print("contours length in ak47 process whole shape",len(contours))
    largest_area = sorted(contours, key=cv2.contourArea)[-2]
    x, y, w, h = cv2.boundingRect(largest_area)
    detectionImage_paper = img[y:y+h, x:x+w]
    # cv2.imshow("paper",detectionImage_paper)
    # cv2.waitKey(0)

    #crop black shape from white paper
    detectionImage_paper_gray = cv2.cvtColor(detectionImage_paper, cv2.COLOR_BGR2GRAY)
    detectionImage_paper_gray = cv2.morphologyEx(detectionImage_paper_gray, cv2.MORPH_ERODE, kernel,iterations=3)
    detectionImage_paper_gray = cv2.morphologyEx(detectionImage_paper_gray, cv2.MORPH_DILATE, kernel,iterations=4)
    detectionImage_paper_bin = cv2.threshold(detectionImage_paper_gray, 85, 255, cv2.THRESH_BINARY)[1] #to be parameterized
    detectionImage_paper_bin = cv2.Canny(detectionImage_paper_bin,30,100)
    #cv2.imshow("paper_edged",detectionImage_paper_bin)
    #cv2.waitKey(0)

    contours, hierarchy = cv2.findContours(detectionImage_paper_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #print("contours length in ak47 process paper ",len(contours))

    largest_areas = sorted(contours, key=cv2.contourArea)
    #print("contours length in ak47 process whole shape,",len(contours))
    largest_area = largest_areas[-1]
    #for l in largest_areas:
    #    print(cv2.contourArea(l))
    x, y, w, h = cv2.boundingRect(largest_area)
    detectionImage_shape = detectionImage_paper[y-50:y+h+50, x-50:x+w+50]
    largest_area-=[x-50,y-50]
    epsilon = 0.001*cv2.arcLength(largest_area,True)
    approx = cv2.approxPolyDP(largest_area,epsilon,True)

    mask_shape = np.ones(detectionImage_shape.shape, dtype=np.uint8)
    #print(mask_shape)
    cv2.fillConvexPoly(mask_shape, largest_area, (0,0,0))
    #cv2.imshow("mask_shape", mask_shape)
    #cv2.waitKey(0)
    #mask_shape = cv2.GaussianBlur(mask_shape,(11,11),0)
    #print(mask_shape)
    ##cv2.imwrite("C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/output/jagged.jpg",mask_shape)
    #output = cv2.bitwise_and(detectionImage_shape, mask_shape)
    output = detectionImage_shape.copy()
    #print(output.shape, mask_shape.shape)
    output[mask_shape.astype(np.bool)] = 255
    ##cv2.imwrite("C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/output/final_crop.jpg",detectionImage_paper)
    ##cv2.imwrite("C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/output/final_crop_shape1.jpg",output)
    ##cv2.imwrite(savePath,detectionImage_shape)
    return detectionImage_shape


def process_and_get_diff_ak_kk(bef_img, aft_img, idx=None):
    before_image = cv2.cvtColor(bef_img, cv2.COLOR_BGR2GRAY)
    after_image = cv2.cvtColor(aft_img, cv2.COLOR_BGR2GRAY)
    #print("both_shape:",after_image.shape,before_image.shape)
    before_image = cv2.threshold(before_image, 40, 255, cv2.THRESH_BINARY)[1] #to be parameterized
    cv2.imwrite("C:/Users/Abdelrahman Ezzat/Desktop/New folder/single_processed"+ str(idx) +".jpg",before_image)
    #after_image = cv2.cvtColor(after_image, cv2.COLOR_BGR2GRAY)
    after_image = cv2.threshold(after_image, 40, 255, cv2.THRESH_BINARY)[1] #to be parameterized
    cv2.imwrite("C:/Users/Abdelrahman Ezzat/Desktop/New folder/single_processed_after"+ str(idx) +".jpg",after_image)
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    before_image = cv2.morphologyEx(before_image, cv2.MORPH_DILATE, k1)
    after_image = cv2.morphologyEx(after_image, cv2.MORPH_DILATE, k1)
    output = after_image.copy()
    cv2.imwrite("C:/Users/Abdelrahman Ezzat/Desktop/New folder/single_processed_after_dilated"+ str(idx) +".jpg",after_image)
    print("output shape",output.shape)
    output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
    #output = draw_circles(output)   #takes gray image, returns BGR image

    #print("after_shape:",after_image.shape)
    op = cv2.connectedComponentsWithStats(after_image, connectivity=8, ltype= cv2.CV_32S)
    centroids = op[3]
    bodies = op[2]
    #print(len(centroids))
    centroids = np.round(centroids).astype("int")
    score = 0
    #print("after", centroids)
    for i,c in enumerate(centroids):
        #print(c)
        area = bodies[i][4]
        width = bodies[i][2]
        print("area",i,"=",area)
        #280-900 for (9,9) dilate kernel
        if(60 <= area <= 150):
            score += 1
            cv2.circle(output, (c[0],c[1]), 10, (0,0,255), 3) #radius of width//2
    #cv2.imwrite("C:/Users/Abdallah Reda/Desktop/test_ak/res_"+str(idx) +".jpg",output)
    cv2.imwrite("C:/Users/Abdelrahman Ezzat/Desktop/New folder/res_after"+ str(idx) +".jpg",output)


    output = before_image.copy()
    output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
    #output = draw_circles(output)   #takes gray image, returns BGR image

    #print("after_shape:",after_image.shape)
    op = cv2.connectedComponentsWithStats(before_image, connectivity=8, ltype= cv2.CV_32S)
    centroids = op[3]
    bodies = op[2]
    #print(len(centroids))
    centroids = np.round(centroids).astype("int")
    score_bef = 0
    #print("after", centroids)
    for i,c in enumerate(centroids):
        #print(c)
        area = bodies[i][4]
        width = bodies[i][2]
        print("area",i,"=",area)
        #280-900 for (9,9) dilate kernel
        if(60 <= area <= 150):
            score_bef += 1
            cv2.circle(output, (c[0],c[1]), 10, (0,0,255), 3) #radius of width//2
    #cv2.imwrite("C:/Users/Abdallah Reda/Desktop/test_ak/res_"+str(idx) +".jpg",output)
    cv2.imwrite("C:/Users/Abdelrahman Ezzat/Desktop/New folder/res_before"+ str(idx) +".jpg",output)
    print(score - score_bef)
    return output
idx=50
def process_and_get_diff_ak(before_image, after_image):#, idx=None):
    """
    Takes two BGR images for an AK47 shooting targets, and calculates the difference in bullets

    Arguments:
    -----------
        before_image: np.ndarray
            Begin image; image before shooting
        after_image: np.ndarray
            End image; image after shooting
    Returns:
    -----------
        diff: np.ndarray
            The difference between both images
        new_image: np.ndarray
            The new image to be used to plot results
        contour:
            The used contour to reduce the area of detection
    """
    global idx
    # before_image=cv2.imread("C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/pistol/cropped_11.jpg")
    # queryImg=cv2.imread("C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/pistol/cropped_21.jpg")
    # before_image=cv2.imread(before_path)
    # after_image=cv2.imread(after_path)
    after_image_aligned, h = alignImages(after_image,before_image)
    after_image_aligned = cv2.cvtColor(after_image_aligned, cv2.COLOR_BGR2GRAY)
    before_image = cv2.cvtColor(before_image, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
    
    #cv2.imshow("i1_aligned", before_image)
    #cv2.imshow("i2_aligned", after_image_aligned)
    #cv2.waitKey(0)
    after_image_processed = cv2.morphologyEx(after_image_aligned, cv2.MORPH_ERODE, kernel,iterations=3)
    after_image_processed = cv2.morphologyEx(after_image_processed, cv2.MORPH_DILATE, kernel,iterations=4)
    after_image_processed = cv2.threshold(after_image_processed, 85, 255, cv2.THRESH_BINARY)[1] #to be parameterized
    after_image_processed = cv2.Canny(after_image_processed,30,100)
    #after_image_processed = cv2.Canny(after_image_aligned, 30, 90)
    

    #before_image_processed = cv2.Canny(before_image, 30, 90)   #10 for pistol
    before_image_processed = cv2.morphologyEx(before_image, cv2.MORPH_ERODE, kernel,iterations=3)
    before_image_processed = cv2.morphologyEx(before_image_processed, cv2.MORPH_DILATE, kernel,iterations=4)
    before_image_processed = cv2.threshold(before_image_processed, 85, 255, cv2.THRESH_BINARY)[1] #to be parameterized
    before_image_processed = cv2.Canny(before_image_processed,30,100)
    #cv2.imshow("after_image_processed",after_image_processed)
    #cv2.imshow("before_image_processed",before_image_processed)
    #cv2.waitKey()
    cnts1, _ = cv2.findContours(after_image_processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts2, _ = cv2.findContours(before_image_processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cnts2, _ = cv2.findContours(before_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts1.sort(key=cv2.contourArea)
    cnts2.sort(key=cv2.contourArea)
    print("diff cnts1 len in ak get diff:",len(cnts1))
    print("diff cnts2 len in ak get diff:",len(cnts2))
    large_cnt1 = cnts1[-2]
    large_cnt2 = cnts2[-2]

    '''
    # test contours alignment
    im_copy = np.zeros_like(after_image_aligned)
    im_copy = cv2.cvtColor(im_copy, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(im_copy, [large_cnt1], 0, (255, 0, 0), 3)
    cv2.drawContours(im_copy, [large_cnt2], 0, (0, 255, 0), 3)
    #cv2.imshow('new', im_copy)
    #cv2.waitKey(0)
    '''
    #large_cnt2 = scale_contour(cnts2[-2], 0.9)

    #translate shifted image to center of contour
    center1 = get_center(large_cnt1)
    center2 = get_center(large_cnt2)
    
    if cv2.contourArea(large_cnt1) < cv2.contourArea(large_cnt2):
        min_cnt = large_cnt1
        if center1!=(0,0) and center2!=(0,0):
            cx,cy = center1[0]-center2[0], center1[1]-center2[1]
            before_image = translate_image(before_image, cx, cy)
    else:
        min_cnt = large_cnt2
        if center1!=(0,0) and center2!=(0,0):
            cx,cy = center2[0]-center1[0], center2[1]-center1[1]
            after_image_aligned = translate_image(after_image_aligned, cx, cy)
    #print(cx, cy)
    #cv2.imshow('new', after_image_aligned)
    #cv2.imshow('new', before_image)
    #cv2.waitKey(0)
    r1 = cvUtils.__cropContourMaskingOutInfo(after_image_aligned, min_cnt, 10, 255)
    r2 = cvUtils.__cropContourMaskingOutInfo(before_image, min_cnt, 10, 255)
    x, y, w, h = cv2.boundingRect(min_cnt)
    min_cnt -= [x-10,y-10]
    # cv2.imshow('orig', r2)
    # cv2.waitKey(0)
    # cv2.imshow('new', r1)
    # cv2.waitKey(0)
    #exit()

    after_image_aligned = r1
    before_image = r2
    #r1 = cv2.drawContours(after_image_aligned, [min_cnt], -1, (0,255,0),3)
    #r2 = cv2.drawContours(before_image, [min_cnt], -1, (0,255,0),3)
    # cv2.imshow("before_diff_1", r1)
    # cv2.imshow("before_diff_2", r2)
    # cv2.waitKey(0)
    #cv2.imwrite("C:/Users/Abdallah Reda/Desktop/test_ak/n1_"+ str(idx) +".jpg",after_image_aligned)
    #cv2.imwrite("C:/Users/Abdallah Reda/Desktop/test_ak/n2_"+ str(idx) +".jpg",before_image)
    #print(after_image_aligned.shape, h.shape)
    #img1 = cv2.imread("C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/pistol/cropped_21.jpg")
    #img2 = IDMatcher(img1, "C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/pistol/cropped_11.jpg")
    ##cv2.imwrite("C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/pistol/cropped_21_edit.jpg",img2)
    ''' CALCULATE DIFFERENCE'''
    diff = cv2.subtract(after_image_aligned,before_image)
    #cv2.imshow("diff", diff)
    #cv2.waitKey(0)
    cv2.imwrite("C:/Users/Abdelrahman Ezzat/Desktop/New folder/diff_align_"+ str(idx) +".jpg",diff)
    diff = cv2.threshold(diff,10,255,cv2.THRESH_BINARY)[1]
    #cv2.imshow("diff_bef_gauss", diff)
    #cv2.waitKey(0)
    diff = cv2.GaussianBlur(diff, (15,15), 2)
    cv2.imwrite("C:/Users/Abdelrahman Ezzat/Desktop/New folder/diff_blurred_"+ str(idx) +".jpg",diff)
    #Kernel_sharpen = np.array([[-1,-1,-1], [-1, 9,-1],[-1,-1,-1]])
    #diff = cv2.filter2D(diff, -1, Kernel_sharpen)
    print("diff shape",diff.shape)
    n_diff = np.zeros_like(diff)
    alpha = 2.0 # Simple contrast control
    beta = 50    # Simple brightness control
    #for y in range(diff.shape[0]):
    #    for x in range(diff.shape[1]):
    #        n_diff[y,x] = np.clip(alpha*diff[y,x] + beta, 0, 255)
    lookUpTable = np.empty((1,256), np.uint8)
    gamma = 0.4
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    n_diff = cv2.LUT(diff, lookUpTable)
    cv2.convertScaleAbs(diff, n_diff, alpha, beta)
    cv2.imwrite("C:/Users/Abdelrahman Ezzat/Desktop/New folder/diff_sharpened_"+ str(idx) +".jpg",n_diff)
    diff = cv2.threshold(diff,50,255,cv2.THRESH_BINARY)[1]
    cv2.imwrite("C:/Users/Abdelrahman Ezzat/Desktop/New folder/diff_blurred_threshed_"+ str(idx) +".jpg",diff)
    #cv2.imshow("diff_after_gauss", diff)
    #cv2.waitKey(0)
    #cv2.imwrite("C:/Users/Abdallah Reda/Desktop/test_ak/diff_thresh_"+ str(idx) +".jpg",diff)
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    #diff = cv2.morphologyEx(diff, cv2.MORPH_ERODE, k1)
    #cv2.imshow('diff_eroded', diff)
    #cv2.waitKey(0)
    #diff = cv2.morphologyEx(diff, cv2.MORPH_DILATE, k2, iterations=1)
    #cv2.imwrite("C:/Users/Abdallah Reda/Desktop/test_ak/diff_dilated"+ str(idx) +".jpg",diff)
    #cv2.imshow('diff_dilated', diff)
    #cv2.waitKey(0)
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
        #cv2.imwrite("C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/pistol/diff_res.jpg",output)

        '''
    #output = after_image_aligned.copy()
    #print("output shape",output.shape)
    #output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
    #output = draw_circles(output)   #takes gray image, returns BGR image
    '''
    op = cv2.connectedComponentsWithStats(diff, connectivity=8, ltype= cv2.CV_32S)
    centroids = op[3]
    bodies = op[2]
    #print(len(centroids))
    centroids = np.round(centroids).astype("int")
    score = 0
    #print("after", centroids)
    for i,c in enumerate(centroids):
        #print(c)
        area = bodies[i][4]
        width = bodies[i][2]
        print(area)
        #280-900 for (9,9) dilate kernel
        if(600 <= area <= 900):
            score += 1
            cv2.circle(output, (c[0],c[1]), 10, (0,0,255), 3) #radius of width//2
    #cv2.imwrite("C:/Users/Abdallah Reda/Desktop/test_ak/res_"+str(idx) +".jpg",output)
    return output, score
    '''
    idx+=1
    return diff , after_image_aligned, scale_contour(min_cnt, 0.95)

def process_and_get_diff_ak_loop(before_image, after_image):#, idx=None):
    """
    Takes two BGR images for an AK47 shooting targets, and calculates the difference in bullets

    Arguments:
    -----------
        before_image: np.ndarray
            Begin image; image before shooting
        after_image: np.ndarray
            End image; image after shooting
    Returns:
    -----------
        diff: np.ndarray
            The difference between both images
        new_image: np.ndarray
            The new image to be used to plot results
        contour:
            The used contour to reduce the area of detection
    """
    global idx
    # before_image=cv2.imread("C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/pistol/cropped_11.jpg")
    # queryImg=cv2.imread("C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/pistol/cropped_21.jpg")
    # before_image=cv2.imread(before_path)
    # after_image=cv2.imread(after_path)
    after_image_aligned, h = alignImages(after_image,before_image)
    after_image_aligned = cv2.cvtColor(after_image_aligned, cv2.COLOR_BGR2GRAY)
    before_image = cv2.cvtColor(before_image, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
    
    #cv2.imshow("i1_aligned", before_image)
    #cv2.imshow("i2_aligned", after_image_aligned)
    #cv2.waitKey(0)
    after_image_processed = cv2.morphologyEx(after_image_aligned, cv2.MORPH_ERODE, kernel,iterations=3)
    after_image_processed = cv2.morphologyEx(after_image_processed, cv2.MORPH_DILATE, kernel,iterations=4)
    after_image_processed = cv2.threshold(after_image_processed, 85, 255, cv2.THRESH_BINARY)[1] #to be parameterized
    after_image_processed = cv2.Canny(after_image_processed,30,100)
    #after_image_processed = cv2.Canny(after_image_aligned, 30, 90)
    

    #before_image_processed = cv2.Canny(before_image, 30, 90)   #10 for pistol
    before_image_processed = cv2.morphologyEx(before_image, cv2.MORPH_ERODE, kernel,iterations=3)
    before_image_processed = cv2.morphologyEx(before_image_processed, cv2.MORPH_DILATE, kernel,iterations=4)
    before_image_processed = cv2.threshold(before_image_processed, 85, 255, cv2.THRESH_BINARY)[1] #to be parameterized
    before_image_processed = cv2.Canny(before_image_processed,30,100)
    #cv2.imshow("after_image_processed",after_image_processed)
    #cv2.imshow("before_image_processed",before_image_processed)
    #cv2.waitKey()
    cnts1, _ = cv2.findContours(after_image_processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts2, _ = cv2.findContours(before_image_processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cnts2, _ = cv2.findContours(before_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts1.sort(key=cv2.contourArea)
    cnts2.sort(key=cv2.contourArea)
    print("diff cnts1 len in ak get diff:",len(cnts1))
    print("diff cnts2 len in ak get diff:",len(cnts2))
    large_cnt1 = cnts1[-2]
    large_cnt2 = cnts2[-2]

    '''
    # test contours alignment
    im_copy = np.zeros_like(after_image_aligned)
    im_copy = cv2.cvtColor(im_copy, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(im_copy, [large_cnt1], 0, (255, 0, 0), 3)
    cv2.drawContours(im_copy, [large_cnt2], 0, (0, 255, 0), 3)
    #cv2.imshow('new', im_copy)
    #cv2.waitKey(0)
    '''
    #large_cnt2 = scale_contour(cnts2[-2], 0.9)

    #translate shifted image to center of contour
    center1 = get_center(large_cnt1)
    center2 = get_center(large_cnt2)
    
    if cv2.contourArea(large_cnt1) < cv2.contourArea(large_cnt2):
        min_cnt = large_cnt1
        if center1!=(0,0) and center2!=(0,0):
            cx,cy = center1[0]-center2[0], center1[1]-center2[1]
            before_image = translate_image(before_image, cx, cy)
    else:
        min_cnt = large_cnt2
        if center1!=(0,0) and center2!=(0,0):
            cx,cy = center2[0]-center1[0], center2[1]-center1[1]
            after_image_aligned = translate_image(after_image_aligned, cx, cy)
    #print(cx, cy)
    #cv2.imshow('new', after_image_aligned)
    #cv2.imshow('new', before_image)
    #cv2.waitKey(0)
    r1 = cvUtils.__cropContourMaskingOutInfo(after_image_aligned, min_cnt, 10, 255)
    r2 = cvUtils.__cropContourMaskingOutInfo(before_image, min_cnt, 10, 255)
    x, y, w, h = cv2.boundingRect(min_cnt)
    min_cnt -= [x-10,y-10]
    # cv2.imshow('orig', r2)
    # cv2.waitKey(0)
    # cv2.imshow('new', r1)
    # cv2.waitKey(0)
    #exit()

    after_image_aligned = r1
    before_image = r2
    r1 = cv2.drawContours(after_image_aligned, [min_cnt], -1, (0,255,0),3)
    r2 = cv2.drawContours(before_image, [min_cnt], -1, (0,255,0),3)
    # cv2.imshow("before_diff_1", r1)
    # cv2.imshow("before_diff_2", r2)
    # cv2.waitKey(0)
    #cv2.imwrite("C:/Users/Abdallah Reda/Desktop/test_ak/n1_"+ str(idx) +".jpg",after_image_aligned)
    #cv2.imwrite("C:/Users/Abdallah Reda/Desktop/test_ak/n2_"+ str(idx) +".jpg",before_image)
    #print(after_image_aligned.shape, h.shape)
    #img1 = cv2.imread("C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/pistol/cropped_21.jpg")
    #img2 = IDMatcher(img1, "C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/pistol/cropped_11.jpg")
    ##cv2.imwrite("C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/pistol/cropped_21_edit.jpg",img2)
    ''' CALCULATE DIFFERENCE'''
    diff = cv2.subtract(after_image_aligned,before_image)
    #cv2.imshow("diff", diff)
    #cv2.waitKey(0)
    cv2.imwrite("C:/Users/Abdelrahman Ezzat/Desktop/New folder/diff_align_"+ str(idx) +".jpg",diff)
    diff = cv2.threshold(diff,10,255,cv2.THRESH_BINARY)[1]
    #cv2.imshow("diff_bef_gauss", diff)
    #cv2.waitKey(0)
    diff = cv2.GaussianBlur(diff, (15,15), 5)
    cv2.imwrite("C:/Users/Abdelrahman Ezzat/Desktop/New folder/diff_blurred_"+ str(idx) +".jpg",diff)
    #Kernel_sharpen = np.array([[-1,-1,-1], [-1, 9,-1],[-1,-1,-1]])
    #diff = cv2.filter2D(diff, -1, Kernel_sharpen)
    print("diff shape",diff.shape)
    n_diff = np.zeros_like(diff)
    alpha = 2.0 # Simple contrast control
    beta = 50    # Simple brightness control
    #for y in range(diff.shape[0]):
    #    for x in range(diff.shape[1]):
    #        n_diff[y,x] = np.clip(alpha*diff[y,x] + beta, 0, 255)
    lookUpTable = np.empty((1,256), np.uint8)
    gamma = 0.4
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    n_diff = cv2.LUT(diff, lookUpTable)
    cv2.convertScaleAbs(diff, n_diff, alpha, beta)
    cv2.imwrite("C:/Users/Abdelrahman Ezzat/Desktop/New folder/diff_sharpened_"+ str(idx) +".jpg",n_diff)
    diff = cv2.threshold(diff,43,255,cv2.THRESH_BINARY)[1]
    cv2.imwrite("C:/Users/Abdelrahman Ezzat/Desktop/New folder/diff_blurred_threshed_"+ str(idx) +".jpg",diff)
    #cv2.imshow("diff_after_gauss", diff)
    #cv2.waitKey(0)
    #cv2.imwrite("C:/Users/Abdallah Reda/Desktop/test_ak/diff_thresh_"+ str(idx) +".jpg",diff)
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    #diff = cv2.morphologyEx(diff, cv2.MORPH_ERODE, k1)
    #cv2.imshow('diff_eroded', diff)
    #cv2.waitKey(0)
    #diff = cv2.morphologyEx(diff, cv2.MORPH_DILATE, k2, iterations=1)
    #cv2.imwrite("C:/Users/Abdallah Reda/Desktop/test_ak/diff_dilated"+ str(idx) +".jpg",diff)
    #cv2.imshow('diff_dilated', diff)
    #cv2.waitKey(0)
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
        #cv2.imwrite("C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/pistol/diff_res.jpg",output)

        '''
    #output = after_image_aligned.copy()
    #print("output shape",output.shape)
    #output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
    #output = draw_circles(output)   #takes gray image, returns BGR image
    '''
    op = cv2.connectedComponentsWithStats(diff, connectivity=8, ltype= cv2.CV_32S)
    centroids = op[3]
    bodies = op[2]
    #print(len(centroids))
    centroids = np.round(centroids).astype("int")
    score = 0
    #print("after", centroids)
    for i,c in enumerate(centroids):
        #print(c)
        area = bodies[i][4]
        width = bodies[i][2]
        print(area)
        #280-900 for (9,9) dilate kernel
        if(600 <= area <= 900):
            score += 1
            cv2.circle(output, (c[0],c[1]), 10, (0,0,255), 3) #radius of width//2
    #cv2.imwrite("C:/Users/Abdallah Reda/Desktop/test_ak/res_"+str(idx) +".jpg",output)
    return output, score
    '''
    idx+=1
    return diff , after_image_aligned, scale_contour(min_cnt, 0.95)

def count_and_plot_connectedComponents(detectionImage, plotImage, saveName, min_cnt):
        if len(detectionImage.shape)==3:
            detectionImage = cv2.cvtColor(detectionImage, cv2.COLOR_BGR2GRAY)
        op = cv2.connectedComponentsWithStats(detectionImage, connectivity=8, ltype= cv2.CV_32S)
        centroids = op[3]
        bodies = op[2]
        #print(len(centroids))
        centroids = np.round(centroids).astype("int")
        score = 0
        #print("after", centroids)
        for i,c in enumerate(centroids):
            #print(c)
            area = bodies[i][4]
            width = bodies[i][2]
            height = bodies[i][3]
            print("area of body",i,":",area)
            sx = c[0]-width//2
            sy = c[1]-height//2
            if(cv2.pointPolygonTest(min_cnt, (sx,sy), True) > 0):
                print("body",i,"inside contour")
            #plotImage = cv2.drawContours(plotImage, [min_cnt], -1, (0,255,0),3)
            #280-900 for (9,9) dilate kernel
            #print(self.shooting_params.connected_components_min_area,area,self.shooting_params.connected_components_max_area)
            if(20 <= area <= 150 and cv2.pointPolygonTest(min_cnt, (sx,sy), True) > 0):
                score += 1
                cv2.circle(plotImage, (c[0],c[1]), 8, (0,0,255), 2) #radius of width//2
        cv2.imwrite(saveName,plotImage)
        #imwrite_unicode(self.save_path, saveName, plotImage)
        return score

def cropImage(img, numberOfShapes):
    """

    :type img: BGR Image
    :type numberOfShapes: number of shapes to be found that are equally separated horizontally
    METHOD ASSUMES UNIFORM SHAPES DISTRIBUTION ACROSS THE IMAGE
    """

    height, width, colors = img.shape
    # print("Width of the image = ",width)
    start = 0
    width_cutoff = width // numberOfShapes
    end = width_cutoff
    imgList = []
    #n = 2
    while end <= width:
        image_crop = img[1300:2800, start:end, :]       #may need vertical cutting
        # cvUtils.plotCVImage(n, targetImage)
        imgList.append(image_crop)
        start = end
        end += width_cutoff
        # print("current width = ",start)
        # n += 1
    print(len(imgList))
    return imgList
#img2_path = "C:\\Users\\Abdallah Reda\\Downloads\\CVC-19-Documnet-Wallet-\\BackEnd\\visionapp\\Natinal_ID\\158\\friday14-8\\1"
#img1 = "C:\\Users\\Abdallah Reda\\Downloads\\CVC-19-Documnet-Wallet-\\BackEnd\\visionapp\\Natinal_ID\\158\\friday14-8\\2.jpg"
'''
img1 = "C:/Users/Abdelrahman Ezzat/Desktop/project_vc/results/results2/teste - Copy/3_before.jpg"
img2 = "C:/Users/Abdelrahman Ezzat/Desktop/project_vc/results/results2/teste - Copy/3_after.jpg"
img1 = "C:/Users/Abdelrahman Ezzat/Desktop/project_vc/results/results/testh/3_before.jpg"
img2 = "C:/Users/Abdelrahman Ezzat/Desktop/project_vc/results/results/testh/3_after.jpg"
resultPath = "C:/Users/Abdelrahman Ezzat/Desktop/New folder/result4.jpg"
img1 = cv2.imread(img1)
img2 = cv2.imread(img2)

#img2 = cv2.fastNlMeansDenoising(img2,None,10,21,21)
#cv2.imwrite("C:/Users/Abdelrahman Ezzat/Desktop/New folder/denoise3.jpg", img2)
diff_img,toPlotImg, min_cnt = process_and_get_diff_ak_loop(img1, img2)
count_and_plot_connectedComponents(diff_img, toPlotImg, resultPath, min_cnt)
'''
'''
save_path = "C:/Users/Abdallah Reda/Desktop/test_ak/"
#img2 = cv2.imread(img2)
img1 = cv2.imread(img1)
img1 = process(img1)
#cropped1 = cropImage(img1, 5)
#cropped2 = cropImage(img2, 5)
for i in range(1,5):
    #bef = process(ss)
    img2 = cv2.imread(img2_path+"_"+str(i)+".jpg")
    img2 = process(img2)
    output, score = process_and_get_diff_ak(img2, img1, 65+i)
    print("score:",  score)
    #op = draw_circles(op)
    cv2.imwrite("C:/Users/Abdallah Reda/Desktop/test_ak/res_with_circles_"+str(65+i) +".jpg", output)
for i in range(len(cropped1)):
    processed1 = process(cropped1[i])
    processed2 = process(cropped2[i])
    #cv2.imwrite(save_path+"proc1_"+str(i)+".jpg", processed1)
    #cv2.imwrite(save_path+"proc2_"+str(i)+".jpg", processed2)

#processed1 = process(cropped2[4])
##cv2.imwrite(save_path+"proc1_4.jpg", processed1)
'''