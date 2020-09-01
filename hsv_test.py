import cv2
import numpy as np
from cvUtils import scale_contour
from sklearn.cluster import DBSCAN
from math import sqrt
from itertools import combinations
# convertToOpenCVHSV():
#   converts from HSV range (H: 0-360, S: 0-100, V: 0-100)
#   to what OpenCV expects: (H: 0-179, S: 0-255, V: 0-255)
def convertToOpenCVHSV(H, S, V):
    return np.array([H // 2, S * 2.55, V * 2.55], np.uint8)
def kmeans_cluster(image, clusters=3, rounds=1):
    h, w = image.shape[:2]
    #samples = np.zeros([h*w,3], dtype=np.float32)
    #count = 0

    #for x in range(h):
    #    for y in range(w):
    #        samples[count] = image[x][y]
    #        count += 1
    samples = image.reshape(-1, image.shape[-1]).astype(np.float32)
    #print((samples==samples2).all())
    #print(samples2)
    compactness, labels, centers = cv2.kmeans(samples,
            clusters, 
            None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0001), 
            rounds, 
            cv2.KMEANS_RANDOM_CENTERS)
    #print(len(labels))
    #print(image.shape[0]*image.shape[1])
    centers = np.uint8(centers)
    print(centers)
    sorted_centers = centers.tolist()
    sorted_centers.sort(key = lambda x: sum(x)/len(x))
    print(sorted_centers)
    shape_color = sorted_centers[-1]
    res = centers[labels.flatten()]
    return res.reshape((image.shape)), sorted_centers

#image = cv2.imread('2.jpg')
def inRangeBGR(img, low, high):
    l = np.array([[low]], np.uint8)
    h = np.array([[high]], np.uint8)
    l = cv2.cvtColor(l, cv2.COLOR_BGR2HSV)
    h = cv2.cvtColor(h, cv2.COLOR_BGR2HSV)
    return cv2.inRange(img, l, h)
def are_contours_near(cnt1_dimensions, cnt2_dimensions, thresh=20):
    def doOverlap(cnt1_dimensions, cnt2_dimensions): 
        x1,y1,w1,h1 = cnt1_dimensions
        x2,y2,w2,h2 = cnt2_dimensions
        sx = max(x1, x2)
        sy = max(y1, y2)
        fw = min(x1+w1, x2+w2) - sx
        fh = min(y1+h1, y2+h2) - sy

        return fw>=0 and fh>=0
    
    def rect_distance(cnt1_dimensions, cnt2_dimensions):
        def dist(x1,y1,x2,y2):
            return sqrt((y2-y1)**2 + (x2-x1)**2)
        x1,y1,w1,h1 = cnt1_dimensions
        x2,y2,w2,h2 = cnt2_dimensions
        x1b = x1+w1
        x2b = x2+w2
        y1b = y1+h1
        y2b = y2+h2
        left = x2b < x1
        right = x1b < x2
        bottom = y2b < y1
        top = y1b < y2
        if top and left:
            return dist(x1, y1b, x2b, y2)
        elif left and bottom:
            return dist(x1, y1, x2b, y2b)
        elif bottom and right:
            return dist(x1b, y1, x2, y2b)
        elif right and top:
            return dist(x1b, y1b, x2, y2)
        elif left:
            return x1 - x2b
        elif right:
            return x2 - x1b
        elif bottom:
            return y1 - y2b
        elif top:
            return y2 - y1b
        else:             # rectangles intersect
            return 0.
    # x1,y1,w1,h1 = cnt1_dimensions
    # x2,y2,w2,h2 = cnt2_dimensions
    # cx1 = x1+w1/2
    # cy1 = y1+h1/2
    # cx2 = x2+w2/2
    # cy2 = y2+h2/2
    # dist = sqrt((cy2-cy1)**2 + (cx2-cx1)**2)

    included = doOverlap(cnt1_dimensions, cnt2_dimensions)
    dist = rect_distance(cnt1_dimensions, cnt2_dimensions)
    #print(dist)
    #print(dist < thresh, included)
    return dist<thresh or included
def merge_contours(cnt1_dimensions, cnt2_dimensions):
    x1,y1,w1,h1 = cnt1_dimensions
    x2,y2,w2,h2 = cnt2_dimensions
    #w=h=0
    
    #if (dist < thresh):
    sx = min(x1, x2)
    sy = min(y1, y2)
    fw = max(x1+w1, x2+w2) - sx
    fh = max(y1+h1, y2+h2) - sy
    #fh = fy - min(y1,y2)
    return sx, sy, fw, fh
def dist(cnt):
    x,y,w,h = cv2.boundingRect(cnt)
    cx = x+w/2
    cy = y+h/2
    return (cx*cx) + (cy*cy)

def merge_rects(rects, thresh=20):
    while (1):
        found = 0
        for ra, rb in combinations(rects, 2):
            if are_contours_near(ra, rb, thresh):
                if ra in rects:
                    rects.remove(ra)
                if rb in rects:
                    rects.remove(rb)
                rects.append(merge_contours(ra, rb))
                found = 1
                break
        if found == 0:
            break

    return rects
def test_lines_original():
    image = cv2.imread('C:\\Users\\Abdelrahman Ezzat\\Desktop\\project_vc\\results\\results2\\pistol1\\1_cropped.jpg')
    image2 = cv2.imread('C:\\Users\\Abdelrahman Ezzat\\Desktop\\project_vc\\results\\results2\\pistol1\\1_cropped_after.jpg')

    #original = image.copy()
    kmeans1, colors = kmeans_cluster(image, clusters=3)
    kmeans2, colors = kmeans_cluster(image2, clusters=3)
    cv2.imwrite('C:\\Users\\Abdelrahman Ezzat\\Desktop\\project_vc\\results\\results2\\testp\\1_before_kmeans_clusterd.jpg', kmeans1)
    cv2.imwrite('C:\\Users\\Abdelrahman Ezzat\\Desktop\\project_vc\\results\\results2\\testp\\1_after_kmeans_clusterd.jpg', kmeans2)
    gray = cv2.cvtColor(kmeans1, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,21,2)
    cv2.imwrite('C:\\Users\\Abdelrahman Ezzat\\Desktop\\project_vc\\results\\results2\\testp\\thresh.jpg', thresh)

    cdstP = kmeans1.copy()
    im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # edged=cv2.threshold(im, 10, 255, cv2.THRESH_BINARY_INV)[1]
    low,high = colors[:2]
    edged = cv2.inRange(im, 112, 180)
    cv2.imwrite("C:\\Users\\Abdelrahman Ezzat\\Desktop\\project_vc\\results\\results2\\testp\\edged_threshed.jpg", edged)
    edged = cv2.Canny(edged,30, 100,apertureSize=3)
    cv2.imwrite("C:\\Users\\Abdelrahman Ezzat\\Desktop\\project_vc\\results\\results2\\testp\\edged.jpg", edged)
    edged2=cv2.threshold(im, 180, 255, cv2.THRESH_BINARY)[1]
    cv2.imwrite("C:\\Users\\Abdelrahman Ezzat\\Desktop\\project_vc\\results\\results2\\testp\\edged2_threshed.jpg", edged2)
    edged2 = cv2.Canny(edged2,30, 100,apertureSize=3)
    cv2.imwrite("C:\\Users\\Abdelrahman Ezzat\\Desktop\\project_vc\\results\\results2\\testp\\edged2.jpg", edged2)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    edged_eroded = cv2.morphologyEx(edged, cv2.MORPH_ERODE, kernel, iterations=1)
    cv2.imwrite("C:\\Users\\Abdelrahman Ezzat\\Desktop\\project_vc\\results\\results2\\testp\\eroded.jpg", edged_eroded)
    edged_dilated = cv2.morphologyEx(edged, cv2.MORPH_DILATE, kernel, iterations=1)
    cv2.imwrite("C:\\Users\\Abdelrahman Ezzat\\Desktop\\project_vc\\results\\results2\\testp\\dilated.jpg", edged_dilated)

    cnts, _ = cv2.findContours(edged_dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    largest_area = sorted(cnts, key=cv2.contourArea)[-1]
    mask = np.zeros_like(cdstP)
    largest_area = scale_contour(largest_area, 0.99)
    #cv2.drawContours(cdstP, [largest_area], -1, (0,255,0), 12)
    cv2.imwrite("C:\\Users\\Abdelrahman Ezzat\\Desktop\\project_vc\\results\\results2\\testp\\cont.jpg", mask)
    #edged=cv2.bitwise_not(edged)
    lines = cv2.HoughLinesP(edged, 1, np.pi/180, 30, minLineLength=30, maxLineGap=20)
    if lines is not None:
        print(len(lines))
        for l in lines:
            #for x1, y1, x2, y2 in line:
            l = l[0]
            cy = (l[3]+l[1])//2
            cx = (l[2]+l[0])//2
            if(cv2.pointPolygonTest(largest_area, (cx,cy), True) > 0):# or cv2.pointPolygonTest(largest_area, (l[2],l[3]), True) > 0):
                cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), colors[-1], 3)
            #else:
            #    cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (255,0,0), 3)
        cv2.imwrite("C:\\Users\\Abdelrahman Ezzat\\Desktop\\project_vc\\results\\results2\\testp\\lines.jpg", cdstP)

def process_single(image):
    # image = cv2.imread('C:\\Users\\Abdelrahman Ezzat\\Desktop\\project_vc\\results\\results2\\pistol1\\1_cropped.jpg')
    # image2 = cv2.imread('C:\\Users\\Abdelrahman Ezzat\\Desktop\\project_vc\\results\\results2\\pistol1\\1_cropped_after.jpg')

    #original = image.copy()
    kmeans1, colors = kmeans_cluster(image, clusters=3)
    # cv2.imwrite('C:\\Users\\Abdelrahman Ezzat\\Desktop\\project_vc\\results\\results2\\pistol1\\1_before_kmeans_clusterd.jpg', kmeans1)
    # cv2.imwrite('C:\\Users\\Abdelrahman Ezzat\\Desktop\\project_vc\\results\\results2\\pistol1\\1_after_kmeans_clusterd.jpg', kmeans2)
    # gray = cv2.cvtColor(kmeans1, cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(gray, (5,5), 0)
    # thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,21,2)
    # cv2.imwrite('C:\\Users\\Abdelrahman Ezzat\\Desktop\\project_vc\\results\\results2\\pistol1\\thresh.jpg', kmeans2)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    # morphs = [cv2.MORPH_CLOSE, cv2.MORPH_OPEN, cv2.MORPH_DILATE, cv2.MORPH_ERODE, cv2.MORPH_BLACKHAT, cv2.MORPH_TOPHAT]
    # morphs_s = ['MORPH_CLOSE', 'MORPH_OPEN', 'MORPH_DILATE', 'MORPH_ERODE', 'MORPH_BLACKHAT', 'MORPH_TOPHAT']
    # for i in range(len(morphs)):
    #     nimg = cv2.morphologyEx(kmeans1, morphs[i], kernel)
    #     cv2.imwrite('C:\\Users\\Abdelrahman Ezzat\\Desktop\\project_vc\\results\\results2\\pistol1\\'+ morphs_s[i] +'.jpg', nimg)
    nimg = cv2.morphologyEx(kmeans1, cv2.MORPH_ERODE, kernel)
    
    # cv2.imwrite('C:\\Users\\Abdelrahman Ezzat\\Desktop\\project_vc\\results\\results2\\pistol1\\MORPH_ERODE.jpg', nimg)
    cdstP = kmeans1.copy()
    mask = np.zeros(kmeans1.shape, dtype=np.uint8)
    top = int(0.6 * kmeans1.shape[0])
    bottom = int(0.66 * kmeans1.shape[0])
    print(mask[top][0])
    mask[top:bottom,:] = colors[-1]
    

    mask_w = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    #print(mask_w[top][0])
    masked = cv2.bitwise_and(cdstP, mask, mask=mask_w)
    masked_w = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    cnts,_ = cv2.findContours(masked_w, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts.sort(key=cv2.contourArea)
    cv2.drawContours(masked, [cnts[-1]], -1, colors[-1], cv2.FILLED)
    # cv2.imwrite('C:\\Users\\Abdelrahman Ezzat\\Desktop\\project_vc\\results\\results2\\pistol1\\middle_mask.jpg', masked)
    # cv2.imwrite('C:\\Users\\Abdelrahman Ezzat\\Desktop\\project_vc\\results\\results2\\pistol1\\middle_masked.jpg',cv2.bitwise_or(nimg, masked))
    cdstP = cv2.max(cdstP, masked)
    cv2.imwrite('C:\\Users\\Abdelrahman Ezzat\\Desktop\\project_vc\\results\\results2\\pistol1\\middle_masked.jpg',cdstP)

    im = cv2.cvtColor(cdstP, cv2.COLOR_BGR2HSV)
    # edged=cv2.threshold(im, 10, 255, cv2.THRESH_BINARY_INV)[1]
    low,high = colors[:2]
    edged = inRangeBGR(im, low, high)
    cv2.imwrite("C:\\Users\\Abdelrahman Ezzat\\Desktop\\project_vc\\results\\results2\\pistol1\\edged_threshed.jpg", edged)
    edged = cv2.Canny(edged,30, 100,apertureSize=3)
    cv2.imwrite("C:\\Users\\Abdelrahman Ezzat\\Desktop\\project_vc\\results\\results2\\pistol1\\edged.jpg", edged)
    #edged2=cv2.threshold(im, 180, 255, cv2.THRESH_BINARY)[1]
    # cv2.imwrite("C:\\Users\\Abdelrahman Ezzat\\Desktop\\project_vc\\results\\results2\\pistol1\\edged2_threshed.jpg", edged2)
    #edged2 = cv2.Canny(edged2,30, 100,apertureSize=3)
    # cv2.imwrite("C:\\Users\\Abdelrahman Ezzat\\Desktop\\project_vc\\results\\results2\\pistol1\\edged2.jpg", edged2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    edged_dilated = cv2.morphologyEx(edged, cv2.MORPH_DILATE, kernel, iterations=1)
    edged_dilated = cv2.bitwise_not(edged_dilated)
    cv2.imwrite("C:\\Users\\Abdelrahman Ezzat\\Desktop\\project_vc\\results\\results2\\pistol1\\dilated.jpg", edged_dilated)
    #edged_eroded = cv2.morphologyEx(edged, cv2.MORPH_ERODE, kernel, iterations=1)
    #cv2.imwrite("C:\\Users\\Abdelrahman Ezzat\\Desktop\\project_vc\\results\\results2\\pistol1\\eroded.jpg", edged_eroded)
    cdstP_g = cv2.cvtColor(cdstP, cv2.COLOR_BGR2GRAY)
    cnts, _ = cv2.findContours(cdstP_g, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    largest_area = sorted(cnts, key=cv2.contourArea)[-1]
    largest_area = scale_contour(largest_area, 0.99)
    #cdstP = nimg
    #cv2.drawContours(cdstP, [largest_area], -1, (0,255,0), 3)
    #edged=cv2.bitwise_not(edged)
    lines = cv2.HoughLinesP(edged, 1, np.pi/180, 30, minLineLength=30, maxLineGap=20)
    if lines is not None:
        print(len(lines))
        for l in lines:
            #for x1, y1, x2, y2 in line:
            l = l[0]
            cy = (l[3]+l[1])//2
            cx = (l[2]+l[0])//2
            if(cv2.pointPolygonTest(largest_area, (cx,cy), True) > 0):# or cv2.pointPolygonTest(largest_area, (l[2],l[3]), True) > 0):
                cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), colors[-1], 3)
            #else:
            #    cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3)
        cv2.imwrite("C:\\Users\\Abdelrahman Ezzat\\Desktop\\project_vc\\results\\results2\\pistol1\\lines.jpg", cdstP)
    print(cdstP.dtype)
    threshed = cv2.threshold(cdstP,180, 255, cv2.THRESH_BINARY)[1]
    #cv2.imwrite("C:\\Users\\Abdelrahman Ezzat\\Desktop\\project_vc\\results\\results2\\pistol1\\no_lines_threshed.jpg", threshed)
    threshed = cv2.cvtColor(threshed, cv2.COLOR_BGR2GRAY)

    # kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (13,1))
    # kernel_v = np.array([[0,1,0], [0, 1, 0],[0, 1, 0]], dtype=np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    # threshed_v = cv2.morphologyEx(threshed, cv2.MORPH_DILATE, kernel_v, iterations=1)
    eroded_cdstP = cv2.morphologyEx(cdstP, cv2.MORPH_ERODE, kernel, iterations=1)
    cv2.imwrite("C:\\Users\\Abdelrahman Ezzat\\Desktop\\project_vc\\results\\results2\\pistol1\\eroded_cdstp.jpg", eroded_cdstP)
    # cv2.imwrite("C:\\Users\\Abdelrahman Ezzat\\Desktop\\project_vc\\results\\results2\\pistol1\\no_lines_threshed_dilated_v.jpg", threshed_v)
    # threshed_h = cv2.bitwise_not(threshed)
    # threshed_h = cv2.morphologyEx(threshed, cv2.MORPH_ERODE, kernel_h, iterations=1)
    # cv2.imwrite("C:\\Users\\Abdelrahman Ezzat\\Desktop\\project_vc\\results\\results2\\pistol1\\no_lines_threshed_dilated_h.jpg", threshed_h)
    # threshed_h = cv2.morphologyEx(threshed_h, cv2.MORPH_DILATE, kernel_h, iterations=1)
    # cv2.imwrite("C:\\Users\\Abdelrahman Ezzat\\Desktop\\project_vc\\results\\results2\\pistol1\\no_lines_threshed_d_eroded_h.jpg", threshed_h)

    # threshed_v = cv2.morphologyEx(threshed_v, cv2.MORPH_ERODE, kernel, iterations=1)
    #threshed = cv2.cvtColor(threshed, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite("C:\\Users\\Abdelrahman Ezzat\\Desktop\\project_vc\\results\\results2\\pistol1\\no_lines_threshed_eroded_v.jpg", threshed_v)
    # threshed_h = cv2.morphologyEx(threshed_h, cv2.MORPH_ERODE, kernel, iterations=1)
    #threshed = cv2.cvtColor(threshed, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite("C:\\Users\\Abdelrahman Ezzat\\Desktop\\project_vc\\results\\results2\\pistol1\\no_lines_threshed_eroded_h.jpg", threshed_h)
    
    output = image.copy()
    eroded_cdstP = cv2.threshold(eroded_cdstP, 180, 255, cv2.THRESH_BINARY)[1]
    eroded_cdstP = cv2.cvtColor(eroded_cdstP, cv2.COLOR_BGR2GRAY)
    cnts,_ = cv2.findContours(eroded_cdstP, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print("len contours",len(cnts))
    cnts.sort(key=cv2.contourArea)
    i=0
    cnt=0
    rects = []
    #cv2.drawContours(cdstP, cnts, -1, (0,255,0), 3)
    while i<len(cnts)-1:
        #cv2.drawContours(cdstP, cnts, i, (0,0,255), 3)
        #x,y,_,_ = cv2.boundingRect(cnts[i])
        #print(x,y)
        x,y,w,h = cv2.boundingRect(cnts[i])
        j=i+1
        rects.append(cv2.boundingRect(cnts[i]))
        # while j<len(cnts) and are_contours_near([x,y,w,h], cv2.boundingRect(cnts[j]), 30):
        #    x,y,w,h = merge_contours([x,y,w,h], cv2.boundingRect(cnts[j]))
        #    i=j
        #    j+=1
        #cv2.rectangle(cdstP, (x,y), (x+w, y+h), (0,0,255), 1)
        #cnt+=1
        i+=1
        #cv2.putText(cdstP, str(cv2.contourArea(cnts[i])), (x,y), cv2.FONT_HERSHEY_SIMPLEX,0.25, (0, 255, 0), 1)
    print("len rects before", len(rects))
    rects = merge_rects(rects, 10)
    print("len rects merged", len(rects))
    for rect in rects:
        x,y,w,h = rect
        max_dim, min_dim = max(w,h), min(w,h)
        if(w*h>=180 and max_dim/min_dim<=1.4): #200,1.5
            cnt+=1
            cv2.rectangle(output, (x,y), (x+w, y+h), (0,0,255), 1)
    print("len rects threshed", cnt)
    # points = np.array(points)
    # db = DBSCAN(eps=0.3, min_samples=10).fit(points)
    # core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    # core_samples_mask[db.core_sample_indices_] = True
    # labels = db.labels_
    # n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # print(labels)
    # print(points)
    # print(n_clusters_)
    cv2.imwrite("C:\\Users\\Abdelrahman Ezzat\\Desktop\\project_vc\\results\\results2\\pistol1\\ccomp7x7_180.jpg", output)
    return output
    '''
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    threshed = cv2.morphologyEx(threshed, cv2.MORPH_ERODE, kernel, iterations=1)
    threshed = cv2.cvtColor(threshed, cv2.COLOR_BGR2GRAY)
    print(threshed.shape)
    db = DBSCAN(eps=0.03, min_samples=10).fit(threshed)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    print(labels.shape)
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print(n_clusters_)
    n_noise_ = list(labels).count(-1)
    print(n_noise_)
    import matplotlib.pyplot as plt

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
            for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = threshed[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=14)

        xy = threshed[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()
    cv2.imwrite("C:\\Users\\Abdelrahman Ezzat\\Desktop\\project_vc\\results\\results2\\pistol1\\no_lines_eroded.jpg", threshed)
    cv2.imwrite("C:\\Users\\Abdelrahman Ezzat\\Desktop\\project_vc\\results\\results2\\pistol1\\edged.jpg", edged)
    cv2.imwrite("C:\\Users\\Abdelrahman Ezzat\\Desktop\\project_vc\\results\\results2\\pistol1\\edged2.jpg", edged2)
'''
#op1,sc1 = process_single(cv2.imread('C:\\Users\\Abdelrahman Ezzat\\Desktop\\project_vc\\results\\results2\\pistol1\\1_cropped.jpg'))
#op2,sc2 = process_single(cv2.imread('C:\\Users\\Abdelrahman Ezzat\\Desktop\\project_vc\\results\\results2\\pistol1\\1_cropped_after.jpg'))
# rects = [ [1,1,5,5],
# [2,2,6,6],
# [10,2,4,4],
# [9,1, 5,5],
# [20,20,9,9]
# ]
# rects = merge_rects(rects, 2)
# print(rects)
'''
lowerLimit = np.uint8([[color - [2,2,2]]])
upperLimit = np.uint8([[color + [2,2,2]]])
lowerLimit = cv2.cvtColor(lowerLimit, cv2.COLOR_BGR2HSV)[0][0]
upperLimit = cv2.cvtColor(upperLimit, cv2.COLOR_BGR2HSV)[0][0]
print(color,lowerLimit,upperLimit)
hsv = cv2.cvtColor(kmeans, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, lowerLimit, upperLimit)
print(hsv.shape, mask.shape)
cv2.imwrite('C:/Users/Abdelrahman Ezzat/Desktop/project_vc/results/results2/pistol1/1_full_after_quantized_threshed.jpg', thresh)
cv2.imwrite('C:/Users/Abdelrahman Ezzat/Desktop/project_vc/results/results2/pistol1/1_full_after_mask.jpg', mask)
cv2.imwrite('C:/Users/Abdelrahman Ezzat/Desktop/project_vc/results/results2/pistol1/1_full_after_quantized_masked.jpg', cv2.bitwise_and(kmeans, kmeans, mask=mask))

green = np.uint8([[[169, 212, 245]]]) #here insert the bgr values which you want to convert to hsv
green = np.uint8([[[195, 202, 221]]]) #here insert the bgr values which you want to convert to hsv
hsvGreen = cv2.cvtColor(green, cv2.COLOR_BGR2HSV)
#print(hsvGreen[0][0])

lowerLimit = hsvGreen[0][0][0] - 10, 100, 100
upperLimit = hsvGreen[0][0][0] + 10, 255, 255

print(hsvGreen[0][0])
print(upperLimit)
print(lowerLimit)
'''
'''
# 1. Load input image
img = cv2.imread('C:/Users/Abdelrahman Ezzat/Desktop/project_vc/results/results2/pistol1/1_full_after - Copy.jpg')

# 2. Preprocess: quantize the image to reduce the number of colors
div = 6
img = img // div * div + div // 2
cv2.imwrite('C:/Users/Abdelrahman Ezzat/Desktop/project_vc/results/results2/pistol1/1_full_after_quantized.jpg', img)
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
low_yellow = np.array([0,0,198])
high_yellow = np.array([179,255,255])
yellow_seg_img = cv2.inRange(hsv_img, low_yellow, high_yellow)
cv2.imwrite('C:/Users/Abdelrahman Ezzat/Desktop/project_vc/results/results2/pistol1/mask.jpg', yellow_seg_img)
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
yellow_seg_img = cv2.morphologyEx(yellow_seg_img, cv2.MORPH_CLOSE, k)
cv2.imwrite('C:/Users/Abdelrahman Ezzat/Desktop/project_vc/results/results2/pistol1/mask_closed.jpg', yellow_seg_img)
print(hsv_img.shape, yellow_seg_img.shape)
cv2.imwrite('C:/Users/Abdelrahman Ezzat/Desktop/project_vc/results/results2/pistol1/difference.jpg', cv2.subtract(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),yellow_seg_img))
'''

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

'''
sawaf
30/5/2020 -> 3/9/2020
'''