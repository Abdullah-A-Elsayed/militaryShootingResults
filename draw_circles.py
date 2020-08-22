import cv2

def get_center(cnt):
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    return (cx,cy)


def draw_circles(img):
    #cv2.imshow("proc", img)
    #cv2.waitKey(0)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
    if(len(img.shape)>2):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # proc_img = cv2.morphologyEx(proc_img, cv2.MORPH_ERODE, kernel,iterations=3)
    # proc_img = cv2.morphologyEx(proc_img, cv2.MORPH_DILATE, kernel,iterations=4)
    # proc_img = cv2.threshold(proc_img, 85, 255, cv2.THRESH_BINARY)[1] #to be parameterized
    proc_img = cv2.Canny(img,30,100)
    #cv2.imshow("proc", proc_img)
    #cv2.waitKey(0)
    contours, hierarchy = cv2.findContours(proc_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    print("contours lenghth ",len(contours))

    largest_areas = sorted(contours, key=cv2.contourArea)
    largest_area = largest_areas[-1]
    #cx, cy = get_center(largest_area)
    x, y, w, h = cv2.boundingRect(largest_area)
    cx = x+w//2
    cy = y+h//2
    output = img.copy()
    output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
    END_RADIUS = w//2
    COUNT = 5
    scores = ["50","40","30","20","10"]
    for r in range(1, COUNT + 1):
        radius = r*END_RADIUS // 5
        tx = (r-1)*(END_RADIUS//5) + 30*(bool(r-1))
        cv2.circle(output, (cx,cy), radius, (255,255,255), 2)
        cv2.putText(output, scores[r-1], (cx , cy+ tx), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 255, 255), 1)
    return output
# img = cv2.imread("C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/ak47/Ref&.jpg")
# op = draw_circles( img)
# cv2.imwrite("C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/ak47/Ref&_circles.jpg", op)
# cv2.imshow("res", op)
# cv2.waitKey(0)