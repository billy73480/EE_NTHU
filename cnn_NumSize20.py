# Import modules
import cv2
import numpy as np
from sklearn.externals import joblib
from skimage.feature import hog
from keras.models import model_from_json
from PIL import Image
import matplotlib.pyplot as plt
import glob
from os import listdir
from os.path import isfile, join

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
src_path = '/home/billy73480/Handwriting_Recognition/Pictures/'
onlyfiles = [ f for f in listdir(src_path) if isfile(join(src_path, f)) ]
#imgs = np.empty(len(onlyfiles), dtype=object)
res_path = '/home/billy73480/Handwriting_Recognition/Results/'
long_side = 20
accuracy = 0
data_num = 0

for n in range(0, len(onlyfiles)):		
	img = cv2.imread( join(src_path, onlyfiles[n]) )
	print onlyfiles[n]
	data = onlyfiles[n].replace('.jpg', '')
	data_seg = len(data)
	data_num += len(data)
	verify = np.zeros((len(data), 3))


	# Image pre-processing
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
	seq = 0
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
				height = height * long_side / width
				width = long_side
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
			cv2.putText(img, str(nbr[0]), (pt_col, pt_row), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
			
			verify[seq] = [pt_col, pt_row, nbr[0]]
			seq += 1

	# sort
	def getKey(item):
		return item[0]
	verify =  sorted(verify, key = getKey)
	
	for n in range(0, len(data)):		
		if int(data[n]) == verify[n][2]:
			accuracy += 1			


	cv2.imwrite(res_path + data + '.png', img)
	#cv2.imshow("Resulting Image with Rectangular ROIs", img)
	#cv2.waitKey()


acu_rate = float(accuracy) / data_num
print 'Total number of data: ' + str(data_num)
print 'Total number of correctly recognized: ' + str(accuracy)
print 'Accuracy: ' + str(acu_rate)
