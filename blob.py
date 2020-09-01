#!/usr/bin/python

# Standard imports
import cv2
import numpy as np


# Read image
im = cv2.imread("H://toblob//33.jpg")
# Setup SimpleBlobDetector parameters.
kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
im=cv2.morphologyEx(im,cv2.MORPH_ERODE,kernel,iterations=2)
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
#params.minThreshold = 10
#params.maxThreshold = 220

# Filter by Area.
params.filterByArea = True
params.minArea = 300
params.maxArea = 500

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.7

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.8
    
# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.4

# Create a detector with the parameters
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3 :
	detector = cv2.SimpleBlobDetector(params)
else : 
	detector = cv2.SimpleBlobDetector_create(params)


# Detect blobs.
keypoints = detector.detect(im)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
# the size of the circle corresponds to the size of blob

im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

print("Number of circular Blobs: " + str(len(keypoints)))
# Show blobs
blob_path = "C://Users//Abdelrahman Ezzat//Desktop//toblob//blob.jpg" #" +""+
cv2.imwrite(blob_path, im_with_keypoints)
#cv2.waitKey(0)