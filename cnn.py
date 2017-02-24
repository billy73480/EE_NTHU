# Import the modules
import cv2
from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np
from keras.models import model_from_json
from PIL import Image                                                            
import matplotlib.pyplot as plt                                                  
import glob

# Load the classifier
#clf = joblib.load("digits_cls.pkl")
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print("model complie")

# Input the image 
img = cv2.imread("note_1.jpg")
img_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#img_gray = cv2.imread('note_1.jpg', cv2.IMREAD_GRAYSCALE)

# Read image into 3D array
'''
imageFolderPath = '/home/B/Pictures/'
imagePath = glob.glob(imageFolderPath+'/*.JPG') 

im_array = numpy.array( [numpy.array(Image.open(imagePath[i]).convert('L'), 'f') for i in range(len(imagePath))] )
'''

# Apply Gaussian filtering
img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)

# Threshold the image
img_thr = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 3)
#cv2.imshow(" cv2.threshold", img_thr)

# Find contours in the image
im_test ,ctrs, hier = cv2.findContours(img_thr.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get rectangles that contain each contour
rects = [cv2.boundingRect(ctr) for ctr in ctrs]

# For each rectangular region, calculate HOG features and predict
# the digit using Linear SVM.
for rect in rects:
    if ((rect[2]+rect[3])*2)>40:
        # Draw the rectangles
        cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
        roi = im_thr[rect[0]:rect[0]+rect[3], rect[1]:rect[1]+rect[2]]
        
        # Make the rectangular region around the digit
        '''
        leng = int(rect[3] * 1.6)
        pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
        pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
        roi = img_thr[pt1:pt1+leng, pt2:pt2+leng]
        '''
        
        # Resize the image
        if rect[2] > rect[3]:
            ratio = 24/rect[2]
            roi = cv2.resize(roi, (2*width, 2*height), interpolation = cv2.INTER_LINEAR)
            if 
        elif rect[2] < rect[3]:
            
        else:
            
            
        roi = cv2.dilate(roi, (3, 3))    
       
    
        #reshape for model input based on X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        roi = roi.reshape(1, 28, 28, 1)
        nbr = loaded_model.predict_classes(roi, verbose = 1)
        cv2.putText(im, str(int(nbr[0])), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)

cv2.imshow("Resulting Image with Rectangular ROIs", im)
cv2.waitKey()
