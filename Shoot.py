from ImportLib import *
import ak47_utils
import pistol_utils
import cap
from enum import Enum
from draw_circles import draw_circles

class ShootingStringTypes():
    AK47 =  "آلي نهاري"
    MORRIS = "موريس نهاري"
    PISTOL = "طبنجة (۹ ملي)"

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
    connected_components_min_area = None
    connected_components_max_area = None
    res_plot_radius = None
    res_plot_thickness = None

    def cropImage(self, img, num_shooters):
        return None
    def process_image(self, img):
        return None
    def get_difference(self, prevImg, newImg):
        return None
    def get_plot_image(self, newImg):
        return newImg

class AK47_params(mParams):
    # REF_IMAGE = "C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/REF2.jpg"
    HORIZONTAL_SEPARATOR_BEGIN = 1200
    HORIZONTAL_SEPARATOR_END = 4000-1600
    VERTICAL_SEPARATOR = 1200
    MAX_SHOOTERS = 5
    THRESH_BINARY = 100#90
    hough_dp = 1
    hough_minDist = 15
    hough_param1 = 118
    hough_param2 = 8
    hough_minRadius = 10
    hough_maxRadius = 24 # 10,15
    connected_components_min_area = 20
    connected_components_max_area = 150
    res_plot_radius = 8        #13
    res_plot_thickness = 2     #14

    def cropImage(self, img, num_shooters):
        height, width, colors = img.shape
        # print("Width of the image = ",width)
        start = 0
        width_cutoff = width // num_shooters
        end = width_cutoff
        imgList = []
        #n = 2
        while end <= width:
            #image_crop = img[AK47_params.HORIZONTAL_SEPARATOR_BEGIN:AK47_params.HORIZONTAL_SEPARATOR_END, start:end, :]       #may need vertical cutting
            image_crop = img[AK47_params.HORIZONTAL_SEPARATOR_BEGIN:AK47_params.HORIZONTAL_SEPARATOR_END, start:end, :]       #may need vertical cutting
            # cvUtils.plotCVImage(n, targetImage)
            imgList.append(image_crop)
            start = end
            end += width_cutoff
            # print("current width = ",start)
            # n += 1
        print("number of cropped images:", len(imgList))
        return imgList
    def process_image(self, img):
        #return ak47_utils.IDCutter(img, AK47_params.REF_IMAGE)
        return ak47_utils.process(img)
        #return img
    def get_difference(self, prevImg, newImg):
        '''
        #prevImg = cv2.threshold(prevImg, AK47_params.THRESH_BINARY, 255, cv2.THRESH_BINARY)[1]
        #newImg = cv2.threshold(newImg, AK47_params.THRESH_BINARY, 255, cv2.THRESH_BINARY)[1]
        #difference = cv2.subtract(newImg,prevImg)
        #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
        #difference = cv2.morphologyEx(difference, cv2.MORPH_OPEN, kernel)
        #cv2.imwrite("C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/diff_demo.jpg", difference)
        #count_results = self.count_and_plot_hough(difference, toPlotImg, resultPath)
        prevImg = cv2.cvtColor(prevImg, cv2.COLOR_BGR2GRAY)
        newImg = cv2.cvtColor(newImg, cv2.COLOR_BGR2GRAY)
        maxshape = max(prevImg.shape, newImg.shape)
        prevImg = cv2.resize(prevImg, maxshape)
        newImg = cv2.resize(newImg, maxshape)
        diff = cv2.subtract(newImg, prevImg)
        cv2.imshow("diff",diff)
        cv2.waitKey(0)
        diff = cv2.threshold(diff, 100, AK47_params.THRESH_BINARY, cv2.THRESH_BINARY)[1] #to be parameterized
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (14,14))
        diff = cv2.morphologyEx(diff, cv2.MORPH_DILATE, kernel,iterations=1)
        return diff
        '''
        return ak47_utils.process_and_get_diff_ak(prevImg, newImg)


    def get_plot_image(self, newImg):
        output = newImg.copy()
        print("output shape",output.shape)
        #output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
        output = draw_circles(output)   #takes gray image, returns BGR image
        return output

class Pistol_params(mParams):
    # REF_IMAGE = "C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/pistol/Ref_bg.jpg"
    HORIZONTAL_SEPARATOR_BEGIN = 1250
    HORIZONTAL_SEPARATOR_END = 2600
    VERTICAL_SEPARATOR = 1200
    MAX_SHOOTERS = 5
    THRESH_BINARY = 10
    hough_dp = 1
    hough_minDist = 15
    hough_param1 = 118
    hough_param2 = 8
    hough_minRadius = 10
    hough_maxRadius = 24 # 10,15
    connected_components_min_area = 400
    connected_components_max_area = 600
    res_plot_radius = 13
    res_plot_thickness = 14
    #index=0
    def cropImage(self, img, num_shooters):
        return pistol_utils.cropImage(img, num_shooters)[0]
    def process_image(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img_gray
    def get_difference(self, prevImg, newImg):
        '''
        if(len(prevImg.shape)>2):
            prevImg = cv2.cvtColor(prevImg, cv2.COLOR_BGR2GRAY)
        if(len(newImg.shape)>2):
            newImg = cv2.cvtColor(newImg, cv2.COLOR_BGR2GRAY)
        #cv2.imshow('img1',img1)
        #cv2.waitKey(0)
        #cv2.imshow('img2',img2)
        #cv2.waitKey(0)
        #self.index+=1
        diff = cv2.absdiff(newImg,prevImg)
        #cv2.imwrite('C:\\Users\\Abdelrahman Ezzat\\Desktop\\New folder\\diff_'+str(self.index)+'.jpg',diff)
        #cv2.imwrite("C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/pistol/diff_gray"+str(self.index)+".jpg",diff)
        bw1 = cv2.threshold(diff, Pistol_params.THRESH_BINARY, 255, cv2.THRESH_BINARY)[1]
        #cv2.imwrite("C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/pistol/diff_bin"+str(self.index)+".jpg",bw1)
        #cv2.imwrite('C:\\Users\\Abdelrahman Ezzat\\Desktop\\New folder\\diff_thresh'+str(self.index)+'.jpg',bw1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
        bw1 = cv2.dilate(bw1,kernel,iterations = 1)
        #cv2.imwrite('C:\\Users\\Abdelrahman Ezzat\\Desktop\\New folder\\diff_dilated'+str(self.index)+'.jpg',bw1)
        #cv2.imwrite("C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/pistol/diff_bin_dilated"+str(index)+".jpg",bw1)
        return bw1,prevImg, newImg
        '''
        return pistol_utils.get_diff_align(prevImg, newImg)
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
    connected_components_min_area = 600
    connected_components_max_area = 900
    res_plot_radius = 13
    res_plot_thickness = 14

    def cropImage(self, img, num_shooters):
        pass
    def process_image(self, img):
        return None
    def get_difference(self, prevImg, newImg, toPlotImg, resultPath):
        return None

class ShootingResults:

    def __init__(self, save_path, num_shooters, shooting_type=ShootingTypes.AK47):
        self.num_shooters = num_shooters                           #Number of targets that will be processed at once
        self.current_id = 1                             #ID to determine currently processing shooter
        self.save_path = save_path                      #Name of the unit that is shooting, to be used for creating a folder in which all images will be saved
        self.shooting_type = shooting_type         #Type of shooting for parameters tuning
        self.begin_images = [None]*self.num_shooters    #A list that will hold the pre-shooting images for all num shooters
        self.shooting_params = ShootingTypes.get_params_object(shooting_type)
        print("exists", os.path.exists(save_path), os.getcwd())
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    

    
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



    
    def cropImage(self, image):
        #v_begin = 0
        #cropped_images = []
        #for i in range(self.shooting_params.MAX_SHOOTERS):
        #    cropped_img = self.__cropImage_helper(image, self.shooting_params.HORIZONTAL_SEPARATOR_BEGIN, self.shooting_params.HORIZONTAL_SEPARATOR_END, v_begin, v_begin + self.shooting_params.VERTICAL_SEPARATOR)
        #    cv2.imwrite("C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/cropped_"+ str(i+1) + ".jpg", cropped_img)
        #    cropped_images.append(cropped_img)
        #    v_begin += self.shooting_params.VERTICAL_SEPARATOR
        return self.shooting_params.cropImage(image, self.shooting_params.MAX_SHOOTERS) #self.num_shooters

    def prepare_image(self, img, newImageName):
        #image = "C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/Shoot2.jpg"
        #self.IDCutter(imagePath, self.shooting_params.REF_IMAGE, newImagePath)
        #InputImage = "C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/difference_before.jpg"
        #img = cv2.imread(imagePath)
        processed_image = self.shooting_params.process_image(img)
        #print("processed image,",processed_image)
        imwrite_unicode(self.save_path, newImageName, processed_image)
        return processed_image
    def prepare_collected_images(self, c):
        pass
    def count_and_plot_hough(self, detectionImage, plotImage, saveName):
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
                cv2.circle(plotImage, (x, y), self.shooting_params.res_plot_radius, (34,100,251), self.shooting_params.res_plot_thickness)
                #cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
            # show the output image
            #cv2.imshow("output", np.hstack([image, output]))
            #cv2.waitKey(0)
            #print("Dasdaoajoijnsss")
            print("circles = ", len(circles))
            #cv2.imwrite(savePath, plotImage) #np.hstack([image, output]))
            imwrite_unicode(self.save_path, saveName, plotImage)
            return len(circles)
        return 0
        
    def count_and_plot_connectedComponents(self, detectionImage, plotImage, saveName, min_cnt):
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
            if(self.shooting_params.connected_components_min_area <= area <= self.shooting_params.connected_components_max_area and cv2.pointPolygonTest(min_cnt, (sx,sy), True) > 0):
                score += 1
                cv2.circle(plotImage, (c[0],c[1]), self.shooting_params.res_plot_radius, (0,0,255), self.shooting_params.res_plot_thickness) #radius of width//2
        #cv2.imwrite("C:/Users/Abdallah Reda/Desktop/test_ak/res_"+str(idx) +".jpg",output)
        imwrite_unicode(self.save_path, saveName, plotImage)
        return score

    def calculate_difference(self, previousImagePath, newImagePath, toPlotImagePath, resultPath):
        prev = cv2.imread(previousImagePath,0)
        new = cv2.imread(newImagePath,0)
        difference = cv2.subtract(new,prev)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
        difference = cv2.morphologyEx(difference, cv2.MORPH_OPEN, kernel)
        # cv2.imwrite("C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/diff_demo.jpg", difference)
        count_results = self.count_and_plot_hough(difference, cv2.imread(toPlotImagePath), resultPath)
        print(count_results)

    def calculate_difference_images(self, prevImg, newImg, toPlotImg, resultPath):
        #prev = cv2.imread(previousImagePath,0)
        #new = cv2.imread(newImagePath,0)
        diff_img,toPlotImg, min_cnt = self.shooting_params.get_difference(prevImg, newImg)
        toPlotImg = self.shooting_params.get_plot_image(toPlotImg)
        return self.count_and_plot_connectedComponents(diff_img, toPlotImg, resultPath, min_cnt)
        imwrite_unicode(self.save_path, resultPath, diff_img)
        return score
        
    
    def begin_shooting(self):
        #full_image_path = cap.func_TakeNikonPicture(self.save_path+str(self.current_id)+"_full_before.jpg")
        #full_image_path = self.save_path+str(self.current_id)+"_full_before.jpg"
        if self.shooting_type == ShootingTypes.PISTOL:
           full_image_path =  get_configuration("PATHS","Sample1")
        else:
           full_image_path =  get_configuration("PATHS","AKSample1")
        full_image = cv2.imread(full_image_path)
        #cv2.imshow('png',full_image)
        cropped_images = self.cropImage(full_image)
        print("length=",len(cropped_images))
        #ASSERT len(cropped_images) == self.num_shooters
        if(len(cropped_images) >= self.num_shooters):
            self.begin_images = []
            for i in range(-1, -1 - min(self.num_shooters, len(cropped_images)), -1): #get rightmost {num_shooters} images from right to left
                index = - 1 - i
                #cv2.imwrite( (self.save_path+str(self.current_id+index)+"_cropped.jpg").encode('UTF-8'), cropped_images[i])
                imwrite_unicode(self.save_path, str(self.current_id+index)+"_cropped.jpg", cropped_images[i])
                resPath = str(self.current_id+index)+"_before.jpg"
                processed = self.prepare_image(cropped_images[i], resPath)
                self.begin_images.append(processed)
        else:
            print("cropped images doesn't match required targets")
    ''' returns two lists,
    scores: list of ints which are the score of each target
    path: save path of resulting image for each target'''
    def end_shooting(self):
        #full_image_path = cap.func_TakeNikonPicture(self.save_path+str(self.current_id)+"_full_after.jpg")
        #full_image_path = self.save_path+str(self.current_id)+"_full_after.jpg"
        if self.shooting_type == ShootingTypes.PISTOL:
           full_image_path =  get_configuration("PATHS","Sample2")
        else:
           full_image_path =  get_configuration("PATHS","AKSample2")
        full_image = cv2.imread(full_image_path)
        cropped_images = self.cropImage(full_image)
        end_images = []
        if(len(cropped_images) >= self.num_shooters):
            for i in range(-1, -1 - min(self.num_shooters, len(cropped_images)), -1): #get rightmost {num_shooters} images from right to left
                index = -1 - i
                imwrite_unicode(self.save_path, str(self.current_id+index)+"_cropped_after.jpg", cropped_images[i])
                resPath = str(self.current_id+index)+"_after.jpg"
                processed = self.prepare_image(cropped_images[i], resPath)
                end_images.append(processed)
        else:
            print("cropped images doesn't match required targets")
            return
        scores = []
        paths = []
        for i in range(len(end_images)):
            prev, new = self.begin_images[i], end_images[i]
            resPath = str(self.current_id)+"_result.jpg"
            self.current_id+=1
            score = self.calculate_difference_images(prev, new, self.shooting_params.get_plot_image(new), resPath)
            # score = self.calculate_difference_images(prev, new, cropped_images[i], resPath)

            scores.append(score)
            paths.append(self.save_path+resPath)
        return (scores, paths)
            
            
#sr_pistol = ShootingResults(ShootingTypes.PISTOL)
#pistol_image = "C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/pistol/test.jpg"
#n_pistol_image = "C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/pistol/processed.jpg"
#sr_pistol.prepare_image(pistol_image, n_pistol_image)
#exit()


'''
sr.begin_images = []
for i in range(1,6): #get rightmost {num_shooters} images from right to left
    index = - 1 - i
    #cv2.imwrite( (self.save_path+str(self.current_id+index)+"_cropped.jpg").encode('UTF-8'), cropped_images[i])
    img1 = cv2.imread(sr.save_path+ str(i) +"_cropped.jpg")
    #imwrite_unicode(sr.save_path, str(self.current_id+index)+"_cropped.jpg", cropped_images[i])
    resPath = str(i)+"_before.jpg"
    processed = sr.prepare_image(img1, resPath)
    sr.begin_images.append(processed)
end_images = []
for i in range(1,6): #get rightmost {num_shooters} images from right to left
    img2 = cv2.imread(sr.save_path+ str(i) +"_cropped_after.jpg")
    
    resPath = str(i)+"_after.jpg"
    processed = sr.prepare_image(img2, resPath)
    end_images.append(processed)

scores = []

for i in range(3,4):
    prev, new = sr.begin_images[i-1], end_images[i-1]
    #print("prev,",prev)
    #print("new,",new)
    resPath = str(sr.current_id)+"_result.jpg"
    sr.current_id+=1
    score = sr.calculate_difference_images(prev, new, new, resPath)
    # score = self.calculate_difference_images(prev, new, cropped_images[i], resPath)

    scores.append(score)
    #paths.append(self.save_path+resPath)
print(scores)
'''
'''
sr = ShootingResults("C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/output/",1)
prev = "C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/processed_1_binary_before.jpg"
new = "C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/processed_1_binary_after.jpg"
plot = "C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/processed_1_ after.jpg"
res = "C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/final_demo.jpg"

testImage_2b = "C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/output/second_trial.jpg"
testImage_4b = "C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/output/second_trial_after.jpg"
new_2b = "C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/output/2b_test.jpg"
new_4b = "C:/Users/Abdallah Reda/Downloads/CVC-19-Documnet-Wallet-/BackEnd/visionapp/Natinal_ID/output/4b_test.jpg"
testImage_2b = cv2.imread(testImage_2b)
testImage_4b = cv2.imread(testImage_4b)
sr.begin_shooting()
sr.end_shooting()
print("done")

'''
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