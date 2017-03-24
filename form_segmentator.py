import cv2
import numpy as np

img = cv2.imread('form_7_extract.jpg')
print ('img' ,img.shape)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
# edges = cv2.Canny(gray,50,150,apertureSize = 3)
edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, 9, 4)

'''
print "STEP 1: Edge Detection"
cv2.imshow("Image", img)
cv2.imshow("Edged", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

lines = cv2.HoughLines(edges,1,(np.pi/180),1000)
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
	    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),8)
	    cv2.imwrite('line.jpg',img)
	    # separate line in horizontal and vertical line
	    if b == 1: 
		print ('hor')
		y1_list.append(y1)
	    else: 
		print ('ver')
		x1_list.append(x1)
	    
	    # print line
	    print((x1,y1),(x2,y2))
	    
y1_list.sort()
x1_list.sort()
print y1_list
print x1_list



# calculate horizontal_gap and vertical_gap 


# cut the form into segment, left some margin
length_y = len(y1_list)-1
length_x = len(x1_list)-1
print ('(length_x,length_y)', (length_x,length_y))
y_gap =int(y1_list[1] - y1_list[0])
x_gap =int(x1_list[1] - x1_list[0])

num_of_form_seg =int((len(y1_list)+1) * (len(x1_list)+1))



for y in range(0,length_y):
	for x in range(0, length_x+1):
		cv2.imwrite('seg' + str(y) + '_' + str(x) +'.jpg',img[y1_list[y]:y1_list[y]+y_gap,x1_list[x]:x1_list[x]+x_gap])





































