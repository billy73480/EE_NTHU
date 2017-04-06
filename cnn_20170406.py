# Import modules
import cv2
import numpy as np
from keras.models import model_from_json
from os import listdir
from os.path import isfile, join
from skimage.filters import threshold_adaptive
from openpyxl import Workbook


# Functions
def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")

	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	# return the ordered coordinates
	return rect


def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect

	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")

	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

	# return the warped image
	return warped


def form_extract(img):
	# convert the image to grayscale, blur it, and find edges
	# in the image
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(blur, 75, 200)


	# find the contours in the edged image, keeping only the
	# largest ones, and initialize the screen contour
	(test,cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

	# loop over the contours
	for c in cnts:
		# print c
		# approximate the contour
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02 * peri, True)

		# if our approximated contour has four points, then we
		# can assume that we have found our screen
		if len(approx) == 4:
			screenCnt = approx
			break
		 

	# apply the four point transform to obtain a top-down
	# view of the original image
	form = four_point_transform(img, screenCnt.reshape(4, 2))

	form = cv2.cvtColor(form, cv2.COLOR_BGR2GRAY)
	form = threshold_adaptive(form, 251, offset = 10)
	form = form.astype("uint8") * 255
	
	cv2.imshow("Form", form)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	return form


def cell_extract(form):
	gray = cv2.cvtColor(form, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray, (5, 5), 0)
	thr = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, 9, 4)

	lines = cv2.HoughLines(form, 1, (np.pi/180), 1000)
	y1_list = list();
	x1_list = list();


	for x in range(0,len(lines)):
		for rho,theta in lines[x]:
	    		a = np.cos(theta)
	    		b = np.sin(theta)
			#   print(rho, theta, a, b)
	    		x0 = a*rho
	    		y0 = b*rho
	    		x1 = int(x0 + 1000*(-b))
	    		y1 = int(y0 + 1000*(a))
	    		x2 = int(x0 - 1000*(-b))
	    		y2 = int(y0 - 1000*(a))
	    		cv2.line(form,(x1,y1),(x2,y2),(0,0,255),8)
	    		cv2.imwrite('line.jpg',form)
	    		# separate line in horizontal and vertical line
	    		if b == 1: 
				#print ('hor')
				y1_list.append(y1)
	    		else: 
				#print ('ver')
				x1_list.append(x1)
	    
	   		 # print line
	    		#print((x1,y1),(x2,y2))
	    
	y1_list.sort()
	x1_list.sort()
	#print y1_list
	#print x1_list


	# cut the form into segment, left some margin
	length_y = len(y1_list)-1
	length_x = len(x1_list)-1
	#print ('(length_x,length_y)', (length_x,length_y))
	y_gap = int(y1_list[1] - y1_list[0])
	x_gap = int(x1_list[1] - x1_list[0])

	num_of_form_seg =int((len(y1_list)+1) * (len(x1_list)+1))

	cell = []

	for y in range(0,length_y):
		for x in range(0, length_x+1):
			cell.append(form[y1_list[y]:y1_list[y]+y_gap,x1_list[x]:x1_list[x]+x_gap])
			cv2.imwrite('seg' + str(y) + '_' + str(x) +'.jpg', form[y1_list[y]:y1_list[y]+y_gap,x1_list[x]:x1_list[x]+x_gap])

	return cell


def num_segment(img):
	# Image pre-processing
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray, (5, 5), 0)
	thr = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 4)
	kernel = np.ones((2, 2), np.uint8)
	erode = cv2.erode(thr, kernel, iterations = 1)
	kernel = np.ones((4, 4), np.uint8)
	dilate = cv2.dilate(erode, kernel, iterations = 1)

	# Find contours in the image
	im_test, ctrs, hier = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	# Get rectangles that contain each contour
	rects = [cv2.boundingRect(ctr) for ctr in ctrs]

	# For each rectangular region, calculate HOG features and predict
	# the digit using Linear SVM.
	long_side = 20
	number_list = []

	for rect in rects:
		if ((rect[2]+rect[3])>20):

			pt_col = rect[0]
			pt_row = rect[1]
        		width = rect[2]
			height = rect[3]

			

			# Draw the rectangles
			cv2.rectangle(img, (pt_col, pt_row), (pt_col + width, pt_row + height), (0, 255, 0), 3)
			roi = dilate[pt_row:pt_row+height, pt_col:pt_col+width]

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
								
			number_list.append([roi_black, pt_col, pt_row])	# number_seq is a 2D list
			def getKey(item):
				return item[1]
			number_list = sorted(number_list, key = getKey)

	return number_list


def num_recognition(number_list):
	num_list_recog = []

	for n in range(0, len(number_list)):
		single_num = number_list[n][0].reshape(1, 28, 28, 1)
		nbr = loaded_model.predict_classes(single_num, verbose = 1)
		#cv2.putText(img, str(nbr[0]), (pt_col, pt_row), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
		num_list_recog.append(nbr[0])

	return num_list_recog

	

# Load Keras model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print('Kera model loaded.')
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print('Kera model compiled.')


# Loading image files
src_path = '/home/billy73480/Handwriting_Recognition/Images/'
onlyfiles = [ f for f in listdir(src_path) if isfile(join(src_path, f)) ]
#imgs = np.empty(len(onlyfiles), dtype=object)
#res_path = '/home/billy73480/Handwriting_Recognition/Results/'
print('Image files loaded.')

num_seg_recog = []

for n in range(0, len(onlyfiles)):
	'''
	paper = cv2.imread( join(src_path, onlyfiles[n]) )
	
	form = form_extract(paper)
	print('Form extracted.')

	cell = cell_extract(form)	# cell is an array
	print('Cell extracted.')

	for m in range(0, len(cell)):
		num_seq = num_segment(cell[m])

		num_seg_recog = num_recognition(num_seq)

		print num_seg_recog
		print '\n'

	'''

	cell = cv2.imread( join(src_path, onlyfiles[n]) )

	num_list = num_segment(cell)
	
	num_seq = ''
	for m in range(0, len(num_list)):
		num_seq = num_seq + str( num_recognition(num_list)[m] )
		
	num_seg_recog.append(num_seq)

print '\n'

wb = Workbook()
ws = wb.active
ws.title = "Number Recognition"

for n in range(0, len(onlyfiles)):
	ws.cell(row=n+1, column=1, value=num_seg_recog[n])
	

	print 'Image: ' + str(onlyfiles[n])
	print num_seg_recog[n]
	print '\n'
	

wb.save('number_recognition.xlsx')

	

'''
# READ FILES
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
'''
