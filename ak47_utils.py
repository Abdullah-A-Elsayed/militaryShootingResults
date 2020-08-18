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
    #cv2.imwrite(newImagePath, Edged_Gray_dilated)
    
    #convert to binary for processing
    #thresh = self.shooting_params.THRESH_BINARY
    #Edged_bin = cv2.threshold(Edged_Gray_dilated, thresh, 255, cv2.THRESH_BINARY)[1]
    #(_, Edged_Gray2) = cv2.threshold(Edged_Gray2, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #cv2.imwrite(newImagePath[:-4]+"_binary.jpg", Edged_bin)
    return Edged_Gray_dilated
    #return Edged_Gray2
    #Edged_Gray = cv2.dilate(Edged_Gray , )
    #Edged_Gray2 = cv2.morphologyEx(Edged_Gray, cv2.MORPH_CLOSE, kernel)
    #Edged_Gray2 = cv2.dilate(Edged_Gray , kernel)
    #Edged_Gray = cv2.dilate(Edged_Gray , )
    #cv2.imwrite("C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/ResultOpen.jpg", Edged_Gray2)
    #Edged_Gray2 = cv2.threshold(Edged_Gray2, thresh, 255, cv2.THRESH_BINARY)[1]
    #(_, Edged_Gray2) = cv2.threshold(Edged_Gray2, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #cv2.imwrite(newImagePath[:-4]+"_binary1.jpg", Edged_Gray2)
    #cv2.imshow(Edged_Gray)
def process(img):
    #img = cv2.imread(imgPath)
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    #cv2.imwrite("C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/output/blue.jpg",gray)
    ## convert to hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ## mask of blue (140,70,65) ~ (210, 44, 35)
    #mask_shapeblue = cv2.inRange(hsv, (75,0,0), (180, 255, 255))
    mask_shapeblue = cv2.inRange(hsv, (0,25,0), (180, 255, 255))

    ## final mask and masked
    #mask = cv2.bitwise_or(mask1, mask2)
    mask_shape_inverted = mask_shapeblue#cv2.bitwise_not(mask_shapeblue)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
    mask_shape_eroded = cv2.morphologyEx(mask_shape_inverted, cv2.MORPH_DILATE, kernel,iterations=1)
    mask_shape_eroded = cv2.morphologyEx(mask_shape_eroded, cv2.MORPH_ERODE, kernel,iterations=2)

    #crop white paper from whole image
    contours, hierarchy = cv2.findContours(mask_shape_eroded, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    print("contours lenghth ",len(contours))
    largest_area = sorted(contours, key=cv2.contourArea)[-2]
    x, y, w, h = cv2.boundingRect(largest_area)
    detectionImage_paper = img[y:y+h, x:x+w]

    #crop black shape from white paper
    detectionImage_paper_gray = cv2.cvtColor(detectionImage_paper, cv2.COLOR_BGR2GRAY)
    detectionImage_paper_gray = cv2.morphologyEx(detectionImage_paper_gray, cv2.MORPH_ERODE, kernel,iterations=3)
    detectionImage_paper_gray = cv2.morphologyEx(detectionImage_paper_gray, cv2.MORPH_DILATE, kernel,iterations=4)
    detectionImage_paper_bin = cv2.threshold(detectionImage_paper_gray, 85, 255, cv2.THRESH_BINARY)[1] #to be parameterized
    detectionImage_paper_bin = cv2.Canny(detectionImage_paper_bin,30,100)
    

    contours, hierarchy = cv2.findContours(detectionImage_paper_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    print("contours lenghth ",len(contours))

    largest_areas = sorted(contours, key=cv2.contourArea)
    largest_area = largest_areas[-2]
    #for l in largest_areas:
    #    print(cv2.contourArea(l))
    x, y, w, h = cv2.boundingRect(largest_area)
    detectionImage_shape = detectionImage_paper[y-10:y+h+10, x-10:x+w+10]
    largest_area-=[x-10,y-10]
    epsilon = 0.001*cv2.arcLength(largest_area,True)
    approx = cv2.approxPolyDP(largest_area,epsilon,True)

    mask_shape = np.ones(detectionImage_shape.shape, dtype=np.uint8)
    #print(mask_shape)
    cv2.fillConvexPoly(mask_shape, largest_area, (0,0,0))
    #cv2.imshow("mask_shape", mask_shape)
    #cv2.waitKey(0)
    #mask_shape = cv2.GaussianBlur(mask_shape,(11,11),0)
    #print(mask_shape)
    #cv2.imwrite("C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/output/jagged.jpg",mask_shape)
    #output = cv2.bitwise_and(detectionImage_shape, mask_shape)
    output = detectionImage_shape.copy()
    print(output.shape, mask_shape.shape)
    output[mask_shape.astype(np.bool)] = 255
    #cv2.imwrite("C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/output/final_crop.jpg",detectionImage_paper)
    #cv2.imwrite("C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/output/final_crop_shape1.jpg",output)
    #cv2.imwrite(savePath,detectionImage_shape)
    return detectionImage_shape
    
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
        image_crop = img[:, start:end, :]       #may need vertical cutting
        # cvUtils.plotCVImage(n, targetImage)
        imgList.append(image_crop)
        start = end
        end += width_cutoff
        # print("current width = ",start)
        # n += 1
    print(len(imgList))
    return imgList
    