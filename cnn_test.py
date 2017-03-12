# Import modules
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

# Input the image
src_path = '/home/billy73480/Handwriting_Recognition/Pictures/'
onlyfiles = [ f for f in listdir(src_path) if isfile(join(src_path, f)) ]
imgs = np.empty(len(onlyfiles), dtype=object)
for n in range(0, len(onlyfiles)):
	print onlyfiles[n]
	imgs[n] = cv2.imread( join(src_path, onlyfiles[n]) )
	

res_path = '/home/billy73480/Handwriting_Recognition/Results/'
long_side = 20

for n in range(0, len(onlyfiles)):

	# Image pre-processing
	img_gray = cv2.cvtColor(imgs[n], cv2.COLOR_BGR2GRAY)	
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
	k = 1

	for rect in rects:
		if ((rect[2]+rect[3])>20):

			pt_col = rect[0]
			pt_row = rect[1]
        		width = rect[2]
			height = rect[3]

			# Draw the rectangles
			cv2.rectangle(imgs[n], (pt_col, pt_row), (pt_col + width, pt_row + height), (0, 255, 0), 3)
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
			
			cv2.imwrite(res_path + str(n) + '_roi_%d.png'%k, roi_black)
			k += 1
	
	cv2.imwrite(res_path + 'img_%d.png'%n, imgs[n])
