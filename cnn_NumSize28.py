# Import modules
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
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#img_gray = cv2.imread('note_1.jpg', cv2.IMREAD_GRAYSCALE)
# Read image into 3D array
'''
imageFolderPath = '/home/B/Pictures/'
imagePath = glob.glob(imageFolderPath+'/*.JPG') 

im_array = numpy.array( [numpy.array(Image.open(imagePath[i]).convert('L'), 'f') for i in range(len(imagePath))] )
'''

# Image pre-processing
img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
img_thr = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 4)
kernel = np.ones((2, 2), np.uint8)
img_ero = cv2.erode(img_thr, kernel, iterations = 1)
kernel = np.ones((4, 4), np.uint8)
img_dil = cv2.dilate(img_ero, kernel, iterations = 1)

# Find contours in the image
im_test ,ctrs, hier = cv2.findContours(img_thr.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get rectangles that contain each contour
rects = [cv2.boundingRect(ctr) for ctr in ctrs]

# For each rectangular region, calculate HOG features and predict
# the digit using Linear SVM.
long_side = 20

for rect in rects:
	if ((rect[2]+rect[3])>20):

		pt_col = rect[0]
		pt_row = rect[1]
        	width = rect[2]
		height = rect[3]

		# Draw the rectangles
		cv2.rectangle(img, (pt_col, pt_row), (pt_col + width, pt_row + height), (0, 255, 0), 3)
		roi = img_dil[pt_row:pt_row+height, pt_col:pt_col+width]
		
		# Create a black image
		roi_black = np.zeros((28, 28), np.uint8)
		
		# Resize the image
		
		if width < height:
			width = width * long_side / height
			height = long_side			
		elif width > height:
			width = long_side
			height = height * long_side / width			
		else:
			width = long_side
			height = long_side			

		width_st = (28-width)/2
		height_st = (28-height)/2		
		roi = cv2.resize(roi, (width, height), interpolation = cv2.INTER_LINEAR)
		roi_black[height_st : height_st+height, width_st : width_st+width] = roi
				
		#reshape for model input based on X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
		roi_black = roi_black.reshape(1, 28, 28, 1)
		nbr = loaded_model.predict_classes(roi_black, verbose = 1)
		cv2.putText(img, str(int(nbr[0])), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
		
cv2.imshow("Resulting Image with Rectangular ROIs", img)
cv2.imwrite('Result.png', img)
cv2.waitKey()
