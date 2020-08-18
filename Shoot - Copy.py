from ImportLib import *
from HelperFunc import *
from IDNumberParser import *
from enum import Enum

class ShootingTypes(Enum):
    AK47 = 1
    PISTOL = 2
    MORRIS = 3
    def get_params_object(sType):
        if(sType==ShootingTypes.AK47):
            return AK47_params()
        if(sType==ShootingTypes.PISTOL):
            return Pistol_params()
        if(sType==ShootingTypes.MORRIS):
            return Morris_params()
        return None

class mParams:
    REF_IMAGE = None
    HORIZONTAL_SEPARATOR_BEGIN = None
    HORIZONTAL_SEPARATOR_END = None
    VERTICAL_SEPARATOR = None
    MAX_SHOOTERS = None
    THRESH_BINARY = None
    hough_dp = None
    hough_minDist = None
    hough_param1 = None
    hough_param2 = None
    hough_minRadius = None
    hough_maxRadius = None

class AK47_params(mParams):
    REF_IMAGE = "C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/ak47/Ref$.jpg"
    HORIZONTAL_SEPARATOR_BEGIN = 1250
    HORIZONTAL_SEPARATOR_END = 2600
    VERTICAL_SEPARATOR = 1200
    MAX_SHOOTERS = 5
    THRESH_BINARY = 90
    hough_dp = 1
    hough_minDist = 15
    hough_param1 = 118
    hough_param2 = 8
    hough_minRadius = 10
    hough_maxRadius = 24 # 10,15

class Pistol_params(mParams):
    REF_IMAGE = "C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/pistol/Ref_bg.jpg"
    HORIZONTAL_SEPARATOR_BEGIN = 1250
    HORIZONTAL_SEPARATOR_END = 2600
    VERTICAL_SEPARATOR = 1200
    MAX_SHOOTERS = 5
    THRESH_BINARY = 90
    hough_dp = 1
    hough_minDist = 15
    hough_param1 = 118
    hough_param2 = 8
    hough_minRadius = 10
    hough_maxRadius = 24 # 10,15

class Morris_params(mParams):
    REF_IMAGE = "C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/REF2.jpg"
    HORIZONTAL_SEPARATOR_BEGIN = 1250
    HORIZONTAL_SEPARATOR_END = 2600
    VERTICAL_SEPARATOR = 1200
    MAX_SHOOTERS = 5
    THRESH_BINARY = 90
    hough_dp = 1
    hough_minDist = 15
    hough_param1 = 118
    hough_param2 = 8
    hough_minRadius = 10
    hough_maxRadius = 24 # 10,15

class ShootingResults:

    def __init__(self, shooting_type=ShootingTypes.AK47):
        self.num_shooters = 3                           #Number of targets that will be processed at once
        self.current_id = 1                             #ID to determine currently processing shooter
        self.save_path = "Katiba1"                      #Name of the unit that is shooting, to be used for creating a folder in which all images will be saved
        self.shooting_type = shooting_type         #Type of shooting for parameters tuning
        self.begin_images = [None]*self.num_shooters    #A list that will hold the pre-shooting images for all num shooters
        self.shooting_params = ShootingTypes.get_params_object(self.shooting_type)

    '''reads image path and matches it with a reference and returns matched reference in the image, TODO: pass the reference as a parameter'''
    def IDMatcher(self, imagePath, refImagePath):
        

        MIN_MATCH_COUNT= 30
        detector=cv2.xfeatures2d.SIFT_create()

        FLANN_INDEX_KDITREE=0
        flannParam=dict(algorithm=FLANN_INDEX_KDITREE,tree=5)
        flann=cv2.FlannBasedMatcher(flannParam,{})
        
        #refImagePath = "C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/REF@.jpg"
        trainImg  = cv2.imread(refImagePath,0)
        trainKP,trainDesc=detector.detectAndCompute(trainImg,None)
        
        QueryImgBGR = cv2.imread(imagePath)
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

    def IDCutter(self, imagePath, refImagePath, newImagePath):
        
        Kernel = np.ones((2,2), np.uint8)
        Kernel_Vertical = np.ones((2,1), np.uint8)
        Kernel_sharpen = np.array([[-1,-1,-1], [-1, 9,-1],[-1,-1,-1]])

        Edged = self.IDMatcher(imagePath, refImagePath)


        Edged_Resized = cv2.resize(Edged,(1400,1000))
        Gaussian = cv2.GaussianBlur(Edged_Resized,(29,29),2)
        Edged_Gray = cv2.cvtColor(Gaussian, cv2.COLOR_BGR2GRAY)
        Edged_Gray = cv2.fastNlMeansDenoising(Edged_Gray,10,10,7,21) 
        Edged_Gray = cv2.filter2D(Edged_Gray, -1, Kernel_sharpen)

        #dilate image to widen bullet holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(19,19))
        Edged_Gray2 = cv2.dilate(Edged_Gray,kernel,iterations = 1)
        cv2.imwrite(newImagePath, Edged_Gray2)
        
        #convert to binary for processing
        thresh = self.shooting_params.THRESH_BINARY
        Edged_Gray2 = cv2.threshold(Edged_Gray2, thresh, 255, cv2.THRESH_BINARY)[1]
        #(_, Edged_Gray2) = cv2.threshold(Edged_Gray2, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        cv2.imwrite(newImagePath[:-4]+"_binary.jpg", Edged_Gray2)
        #return Edged_Gray2
        #Edged_Gray = cv2.dilate(Edged_Gray , )
        Edged_Gray2 = cv2.morphologyEx(Edged_Gray, cv2.MORPH_CLOSE, kernel)
        Edged_Gray2 = cv2.dilate(Edged_Gray , kernel)
        #Edged_Gray = cv2.dilate(Edged_Gray , )
        cv2.imwrite("C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/ResultOpen.jpg", Edged_Gray2)
        Edged_Gray2 = cv2.threshold(Edged_Gray2, thresh, 255, cv2.THRESH_BINARY)[1]
        #(_, Edged_Gray2) = cv2.threshold(Edged_Gray2, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        cv2.imwrite(newImagePath[:-4]+"_binary1.jpg", Edged_Gray2)
        #cv2.imshow(Edged_Gray)

        

    
    '''counts number of circle from a read image, returns (centers:List of tuples)'''
    def count_circles_true_image(self, image):
        #cnt=0
        #image = cv2.imread(imgPath)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #image = black_and_white(image,100)
        #with open("C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/arr.txt", 'w') as content_file:
        #   for i in image:
            #    for j in i:
            #      content_file.write(str(j))
                #    content_file.write(", ")
            #  content_file.write("\n")
        #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
        #image = cv2.erode(image,kernel,iterations = 1)
        #print(image[524,217])
        nx,ny=image.shape
        #print(nx,ny)
        vis = np.zeros((nx,ny))

        def get_neighbours(x,y,width,height):
            dx=[-1,0,1,0]
            dy=[0,-1,0,1]
            neighbours = []
            for i in range(4):
                newx,newy = x+dx[i],y+dy[i]
                if(newx>=0 and newx<width and newy>=0 and newy<height):
                    neighbours.append((newx,newy))
            return neighbours
            
        '''takes two points and returns whichever is nearest to the bottom right'''
        def compare_points(p1, p2):
            p1_y, p1_x = p1
            p2_y, p2_x = p2
            if(p2_y>p1_y):
                return p2
            elif(p2_y==p1_y):
                if(p2_x>p1_x):
                    return p2
                else:
                    return p1
            else:
                return p1
        '''loop on all neighbours to mark them as one connected components and return rightmost top point in the component'''
        def dfs(x,y):
            nonlocal vis
            nonlocal image
            nonlocal nx
            nonlocal ny
            vis[x,y] = 1
            neighbours = get_neighbours(x,y,nx,ny)
            p_ret = (x,y)
            for elem in neighbours:
                newx,newy = elem
                if(image[newx,newy]>0 and vis[newx,newy]==0):
                    p_new = dfs(newx,newy)
                    p_ret = compare_points(p_new, p_ret)
            return p_ret
        circles = []
        for i in range(nx):
            for j in range(ny):
                if(image[i,j]>0 and vis[i,j]==0):
                    top_most = (i,j) #get leftmost point in the circle
                    #cnt+=1
                    bottom_most = dfs(i,j)
                    y1, y2 = top_most[0], bottom_most[0]
                    center_y = (y1+y2)//2
                    center = (center_y, j)
                    print(top_most, bottom_most, center)
                    circles.append(center)
        return circles
    '''takes a gray image and a list of centers, return new image with plotted circles'''
    def plot_circles(self, image, circles):
        if circles is not None:
            print("sogrOGJORIjgdoijgorjgoergoperjojoi")
            if(image.shape[1]==1):
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            output = image.copy()
            # convert the (x, y) coordinates and radius of the circles to integers
            #circles = np.round(circles[0, :]).astype("int")
            # loop over the (x, y) coordinates and radius of the circles
            for (x, y) in circles:
                # draw the circle in the output image
                #cv2.circle(output, (y, x), 15, (0,0,255), 3)
                cv2.circle(output, (y, x), 7, (0,0,255), 3)
                #cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
            # show the output image
            #cv2.imshow("output", np.hstack([image, output]))
            #cv2.waitKey(0)
            #cv2.imwrite("C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/FinalResult_ours.jpg", np.hstack([image, output]))
            return output
        return None



    def detectCircles(self, InputImage):

        # load the image, clone it for output, and then convert it to grayscale
        image = cv2.imread(InputImage)
        output = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # detect circles in the image
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 21, minDist= 50, minRadius= 8 , maxRadius = 100)
        # ensure at least some circles were found
        if circles is not None:
            print("sogrOGJORIjgdoijgorjgoergoperjojoi")
            # convert the (x, y) coordinates and radius of the circles to integers
            circles = np.round(circles[0, :]).astype("int")
            # loop over the (x, y) coordinates and radius of the circles
            for (x, y, r) in circles:
                # draw the circle in the output image, then draw a rectangle
                # corresponding to the center of the circle
                cv2.circle(output, (x, y), r, (0, 0, 255), 3)
                #cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
            # show the output image
            #cv2.imshow("output", np.hstack([image, output]))
            #cv2.waitKey(0)
            cv2.imwrite("C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/FinalResult1.jpg", output) #np.hstack([image, output]))

    def __cropImage_helper(self, image, h_begin, h_end, v_begin, v_end):
        return image[h_begin:h_end, v_begin:v_end, :]
    def cropImage(self, image):
        v_begin = 0
        cropped_images = []
        for i in range(self.shooting_params.MAX_SHOOTERS):
            cropped_img = self.__cropImage_helper(image, self.shooting_params.HORIZONTAL_SEPARATOR_BEGIN, self.shooting_params.HORIZONTAL_SEPARATOR_END, v_begin, v_begin + self.shooting_params.VERTICAL_SEPARATOR)
            cv2.imwrite("C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/cropped_"+ str(i+1) + ".jpg", cropped_img)
            cropped_images.append(cropped_img)
            v_begin += self.shooting_params.VERTICAL_SEPARATOR
        return cropped_images

    def prepare_image(self, imagePath, newImagePath):
        #image = "C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/Shoot2.jpg"
        self.IDCutter(imagePath, self.shooting_params.REF_IMAGE, newImagePath)
        #InputImage = "C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/difference_before.jpg"
    def prepare_collected_images(self, c):
        pass
    def count_and_plot(self, detectionImage, plotImage, savePath):
        circles = cv2.HoughCircles(detectionImage,cv2.HOUGH_GRADIENT,self.shooting_params.hough_dp,minDist=self.shooting_params.hough_minDist, param1=self.shooting_params.hough_param1,param2=self.shooting_params.hough_param2,minRadius=self.shooting_params.hough_minRadius,maxRadius=self.shooting_params.hough_maxRadius) # 10,15
        #cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if circles is not None:
            #output = plotImage
            
            # convert the (x, y) coordinates and radius of the circles to integers
            circles = np.round(circles[0, :]).astype("int")
            # loop over the (x, y) coordinates and radius of the circles
            for (x, y, _) in circles:
                # draw the circle in the output image, then draw a rectangle
                # corresponding to the center of the circle
                cv2.circle(plotImage, (x, y), 13,  	(34,100,251), 14)
                #cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
            # show the output image
            #cv2.imshow("output", np.hstack([image, output]))
            #cv2.waitKey(0)
            #print("Dasdaoajoijnsss")
            print("circles = ", len(circles))
            cv2.imwrite(savePath, plotImage) #np.hstack([image, output]))
            return len(circles)
        return None

    def calculate_difference(self, previousImagePath, newImagePath, toPlotImagePath, resultPath):
        #prev = cv2.imread("C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/prev.jpg",0)
        #new = cv2.imread("C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/new.jpg",0)
        prev = cv2.imread(previousImagePath,0)
        new = cv2.imread(newImagePath,0)
        #difference = self.diff_images(prev,new)
        difference = cv2.subtract(new,prev)
        #difference = black_and_white(difference)
        #cv2.imwrite(InputImage, difference)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
        difference = cv2.morphologyEx(difference, cv2.MORPH_OPEN, kernel)
        cv2.imwrite("C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/diff_demo.jpg", difference)
        #count_results = self.count_circles_true_image(difference)
        #if len(count_results)>0:
        #    plotImage = cv2.imread(toPlotImagePath)
        #    final_image = self.plot_circles(plotImage, count_results)
        #    #TODO change saving procedure
        #    cv2.imwrite("C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/result_from_class_demo.jpg", final_image)
        count_results = self.count_and_plot(difference, cv2.imread(toPlotImagePath), resultPath)
        #InputImage = "C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/difference_after.jpg"
        print(count_results)
        #cv2.imwrite(InputImage, difference)
        #detectCircles(InputImage)

    def begin_shooting(self):
        #full_image = capture_Nikon()
        full_image = ""
        cropped_images = self.cropImage(full_image)
        self.begin_images = []
        for cropped_image in cropped_images[:-1-self.num_shooters:-1]: #get rightmost {num_shooters} images from right to left
            #self.begin_images.append()
            pass
            
            
#sr_pistol = ShootingResults(ShootingTypes.PISTOL)
#pistol_image = "C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/pistol/test.jpg"
#n_pistol_image = "C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/pistol/processed.jpg"
#sr_pistol.prepare_image(pistol_image, n_pistol_image)
#exit()
sr = ShootingResults()
prev = "C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/processed_1_binary_before.jpg"
new = "C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/processed_1_binary_after.jpg"
plot = "C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/processed_1_ after.jpg"
res = "C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/final_demo.jpg"

testImage_2b = "C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/ak47/_DSC0009.jpg"
testImage_4b = "C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/ak47/_DSC0012.jpg"
new_2b = "C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/ak47/2b_test.jpg"
new_4b = "C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/ak47/4b_test.jpg"
sr.prepare_image(testImage_2b,new_2b)
sr.prepare_image(testImage_4b,new_4b)
print("done")

'''
path = "C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/demo_@/"
imgs = ["O","O1","O4","O6 - Copy"]
#for i in imgs:
 #   sr.prepare_image(path+i+".jpg",path+i+"_processed.jpg")
for i in range(len(imgs)-1):
    prev = path+imgs[i]+"_processed_binary.jpg"
    new = path+imgs[i+1]+"_processed_binary.jpg"
    plot = path+imgs[i+1]+"_processed.jpg"
    res = path+imgs[i+1]+"_result.jpg"
    sr.calculate_difference(prev,new,plot,res)


img = "C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/cropped_3.jpg"
img_a = "C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/cropped_"
nimg = "C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/processed_3.jpg"
nimg_a = "C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/demo_@/processed_"
#for i in range(1,6):
#    img = img_a+str(i)+".jpg"
#    nimg = nimg_a+str(i)+".jpg"
#    sr.prepare_image(img,nimg)

#sr.cropImage(img)

def plot_c(cimg, index):
    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,minDist=15, param1=118,param2=8,minRadius=9,maxRadius=17 , ) # 10,15
    if circles is not None:
        output = cimg.copy()
        
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
    cv2.imwrite("C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/"+str(index)+".jpg", output) #np.hstack([image, output]))

img = cv2.imread("C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/inter.jpg",0)
#img = cv2.threshold(img, 90, 255, cv2.THRESH_BINARY)[1]
#cv2.imwrite("C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/sharp3_bin.jpg",img)
cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
plot_c(cimg,51)

img = cv2.imread("C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/processed_1_binary1.jpg",0)
cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
plot_c(cimg, 4)
'''