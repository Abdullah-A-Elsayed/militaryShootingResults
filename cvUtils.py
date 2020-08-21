from __future__ import print_function

import cv2
import numpy as np
from matplotlib import pyplot as plt

_x = 0
_y = 0

import cv2
import numpy as np

debug = True
##TODO: change this to dynamic setting
max_number_of_bullets = 3

MAX_FEATURES = 550
GOOD_MATCH_PERCENT = 0.2


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


def Find_HSV_Value(img):
    pass


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def avgBrightness(BGRimg):
    gray_img = cv2.cvtColor(BGRimg, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(gray_img)
    sum = np.sum(y)
    brightness = sum / len(y) / len(y[0])
    return brightness


def binaryThresholdFinder(avgBrightness):
    pass


def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)


def plotCVImage(plotNumber, image, label="", rgbColor=False):
    if (not rgbColor):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.subplot(x, y, plotNumber), plt.title(label)
    plt.imshow(image)


def initializePlot(noRows, noColumns):
    """

    :param noRows:
    :param noColumns:
    """
    global x
    global y
    x = noRows
    y = noColumns


def plotCVImage(plotNumber, image, label="", rgbColor=False):
    if image.dtype != np.uint:
        image = np.array(image, dtype=np.uint8)

    if (not rgbColor and len(image.shape) > 2):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.subplot(x, y, plotNumber), plt.title(label)
    if (len(image.shape) < 3):
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)


def createFigure(listImages, listImageNames=None, numberOfRows=0, numberOfCols=0):
    number_of_images = len(listImages)
    if (numberOfCols != 0 and numberOfRows != 0):
        initializePlot(numberOfRows, numberOfCols)
    elif (number_of_images > 3):
        initializePlot((number_of_images) // 3 + number_of_images % 3, 3)
    else:
        initializePlot(number_of_images, 1)
    plt.figure()
    n = 1
    for image in listImages:
        if (listImageNames == None):
            plotCVImage(n, image)
        else:
            plotCVImage(n, image, listImageNames[n - 1])

        n += 1


def show():
    if debug:
        plt.show()
    pass


def findTargetContours(contours, target_max_area, target_min_area):
    contour_list = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        # print("approx = ", len(approx))
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        print(area)
        if ((len(approx) > 4) and (area < target_max_area) and checkIfContourRectIsSquare(contour) and (
                area > target_min_area) and (len(approx) < 12)):
            contour_list.append(contour)
    return contour_list


def convertRGBToHSV(color):
    """
    :type color: list
    """
    return cv2.cvtColor([[color]], cv2.COLOR_BGR2HSV)


def checkIfContourRectIsSquare(contour):
    x, y, w, h = cv2.boundingRect(contour)
    difference = abs(w - h)
    smaller_dim = min(w, h)
    if (difference < smaller_dim):
        return True
        return False


def blobDetect(differenceImage, imageWithDetectedObjects, params=None, position=None, contour=None):
    """

    :param imageWithDetectedObjects:  this is to draw on it the detected objects
    :type differenceImage: BGR image or gray
    """
    print("blob detection")
    # Read image
    # im = cv2.imread("blob.jpg", cv2.IMREAD_GRAYSCALE)
    if len(differenceImage) == 3:
        differenceImage = cv2.cvtColor(differenceImage, cv2.COLOR_BGR2GRAY)
    # differenceImage= cv2.GaussianBlur(differenceImage,3)
    # Setup SimpleBlobDetector parameters.
    if params is None:
        params = cv2.SimpleBlobDetector_Params()

        # Change thresholds
        params.minThreshold = 0
        params.maxThreshold = 200

        # Filter by Area.
        params.filterByArea = True
        params.minArea = 30

        # Filter by Circularity
        params.filterByCircularity = False
        params.minCircularity = 0.3

        # Filter by Convexity
        params.filterByConvexity = False
        params.minConvexity = 0.4

        # Filter by Inertia
        params.filterByInertia = False
        params.minInertiaRatio = 0.1

        # Create a detector with the parameters
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3:
        detector = cv2.SimpleBlobDetector(params)
    else:
        detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs.
    keypoints = detector.detect(differenceImage)
    numberOfDetectedShapes = len(keypoints)
    if max_number_of_bullets != -1:
        if numberOfDetectedShapes > max_number_of_bullets:
            params.maxThreshold = params.maxThreshold - 5
            return blobDetect(differenceImage, imageWithDetectedObjects, params)
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
    # the size of the circle corresponds to the size of blob
    # print(imageWithDetectedObjects.dtype)
    im_with_keypoints = drawKeyPts(imageWithDetectedObjects, keypoints, (255, 0, 255), 2)

    return im_with_keypoints, numberOfDetectedShapes, keypoints


def drawKeyPts(im, keyp, col, th):
    for curKey in keyp:
        x = np.int(curKey.pt[0])
        y = np.int(curKey.pt[1])
        size = np.int(curKey.size) // 2
        cv2.circle(im, (x, y), size, col, thickness=th, lineType=8, shift=0)
    return im


def show2():
    if debug:
        plt.show()


def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0
    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
    return cnts, boundingBoxes


def drawBndRects(contours, img):
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 30)

    return None


def __cropContourMaskingOutInfo(img, contour):
    x, y, w, h = cv2.boundingRect(contour)
    print(x,y)
    ROI = img[y - 50:y + h + 50, x - 50:x + w + 50]
    # cv2.imshow("roi",ROI)
    # cv2.waitKey(0)
    mask = np.ones(ROI.shape)
    contour-=[x-50,y-50]
    #mask = cv2.drawContours(mask, contour, -1, 0, cv2.FILLED)
    cv2.fillConvexPoly(mask, contour, (0,0,0))
    # Generate output
    output = ROI.copy()
    print(output.shape, mask.shape)
    output[mask.astype(np.bool)] = 0
    print("op",output.shape)
    return output
def __cropContourMaskingOutInfo_white(img, contour):
    x, y, w, h = cv2.boundingRect(contour)
    print(x,y)
    ROI = img[y - 50:y + h + 50, x - 50:x + w + 50]
    # cv2.imshow("roi",ROI)
    # cv2.waitKey(0)
    mask = np.ones(ROI.shape)
    contour-=[x-50,y-50]
    #mask = cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
    #mask = cv2.drawContours(mask, [contour], -1, (255,255,255), -1)

    cv2.fillConvexPoly(mask, contour, (255,255,255))
    # Generate output
    output = ROI.copy()
    print(output.shape, mask.shape)
    output[mask.astype(np.bool)] = 0
    print("op",output.shape)
    return mask

def cropContourFromImage(img, contour):
    # mask = np.zeros_like(img)  # Create mask where white is what we want, black otherwise
    # cv2.drawContours(mask, [contour], -1, 255, -1)  # Draw filled contour in mask
    # out = np.zeros_like(img)  # Extract out the object and place into output image
    # out[mask == 255] = img[mask == 255]
    #
    # # Now crop
    # (y, x, _) = np.where(mask == 255)
    # (topy, topx) = (np.min(y), np.min(x))
    # (bottomy, bottomx) = (np.max(y), np.max(x))
    # out = out[topy:bottomy + 1, topx:bottomx + 1]
    # print(out[500][250])
    x, y = [], []

    x, y, w, h = cv2.boundingRect(contour)
    #cont_new = cv2.
    #cv2.fillConvexPoly(mask, max_contour[0], (255))

    # This is simple slicing to get the "Region of Interest"
    ROI = img[y - 50:y + h + 50, x - 50:x + w + 50]

    return ROI


from win32api import GetSystemMetrics


def showOpenCVWindow(img):
    widthScreen = GetSystemMetrics(0)
    heightScreen = GetSystemMetrics(1)
    img_height = img.shape[0]
    img_width = img.shape[1]
    if (img_width > img_height):
        img_resized = image_resize(img, width=widthScreen - widthScreen // 5)
    else:
        img_resized = image_resize(img, height=heightScreen - heightScreen // 5)

    height = img_resized.shape[0]
    width = img_resized.shape[1]
    winname = ""
    cv2.namedWindow(winname)  # Create a named window
    cv2.moveWindow(winname, (widthScreen - width) // 2, (heightScreen - height) // 2)  # Move it to (40,30)
    cv2.imshow(winname, img_resized)
    cv2.waitKey()
    cv2.destroyAllWindows()
