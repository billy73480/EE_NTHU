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
img_thr = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 3, 3)

# Find contours in the image
im_test ,ctrs, hier = cv2.findContours(img_thr.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get rectangles that contain each contour
rects = [cv2.boundingRect(ctr) for ctr in ctrs]

# For each rectangular region, calculate HOG features and predict
# the digit using Linear SVM.
for rect in rects:
	if ((rect[2]+rect[3]))>20:
		# Draw the rectangles
		cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
		roi = img_thr[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
		roi = cv2.dilate(roi, (3, 3))
		#cv2.imwrite('roi_0%d.png'%n, roi)
		
		# Create a black image
		roi_black = np.zeros((28, 28), np.uint8)
		
		# Resize the image
		if rect[2] < rect[3]:
			width = rect[2]*20/rect[3]
			roi = cv2.resize(roi, (width, 20), interpolation = cv2.INTER_LINEAR)
			width_st = (28-width)/2
			roi_black[4:24, width_st:width_st+width] = roi
		elif rect[2] > rect[3]:
			height = rect[3]*20/rect[2]
			roi = cv2.resize(roi, (20, height), interpolation = cv2.INTER_LINEAR)
			height_st = (28-height)/2
			roi_black[height_st:height_st+height, 4:24] = roi
		else:
			roi = cv2.resize(roi, (20, 20), interpolation = cv2.INTER_LINEAR)
			roi_black[4:24, 4:24] = roi
			
		#cv2.imwrite('roi_1%d.png'%n, black)
		n += 1
		
		#reshape for model input based on X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
		roi_black = roi_black.reshape(1, 28, 28, 1)
		nbr = loaded_model.predict_classes(roi_black, verbose = 1)
		cv2.putText(img, str(int(nbr[0])), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
		
cv2.imshow("Resulting Image with Rectangular ROIs", img)
cv2.imwrite('Result.png', img)
cv2.waitKey()
