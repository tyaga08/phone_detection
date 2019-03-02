# including standard library headers
import numpy as np
import cv2
import matplotlib as plt
from joblib import dump, load
from sklearn import svm
import decimal as dc

# Include library from pyimagesearch
from include.pyimagesearch.helpers import pyramid
from include.pyimagesearch.helpers import sliding_window


# including headers containing various classes
from include import find_hog
from include import svm_classifier


with open('find_phone/labels.txt') as file:
	lines = file.readlines()

	rows = len(lines)
	file_name = ['']*rows
	xstr = ['']*rows
	ystr = ['']*rows
	
	for i in range(0,rows):
		cols = len(lines[i])-1
		space_num = 0
		for j in range(0,cols):
			if lines[i][j] == ' ':
				space_num += 1

			else:
				if space_num == 0:
					file_name[i] += lines[i][j]

(winW, winH) = (50, 50)

def main():

	test_hog = find_hog.find_hog_of_image((50,50),20,20,(10,10))
	phone_count = 0

	clf = load('classifier_model/HOG_SVM_training.joblib')
	
	
	for l in range(0,rows):
		img1 = cv2.imread('find_phone/'+file_name[l])
		print('find_phone/'+file_name[l])
		img = cv2.resize(img1, (500,500))
		prediction = 0
		count = 0
		clone = img.copy()
		max_x = 0.0000
		max_y = 0.0000
		x_diff = 0.0000
		y_diff = 0.0000
		max_probability = 0.0000
		for (x, y, window) in sliding_window(img, stepSize=10, windowSize=(winW, winH)):
		# if the window does not meet our desired window size, ignore it
			# print(x,y)
			
			if window.shape[0] != winH or window.shape[1] != winW:
				continue
			test_hog.compute_hog_descriptor(window)
			check_for_phone = np.reshape(test_hog.descriptor,(1,576))
			prediction = clf.predict(check_for_phone)
			if prediction:
				probability = clf.predict_proba(check_for_phone)
				if probability[0][1] > max_probability:
					max_x = x
					max_y = y
					max_probability = probability[0][1]
				if probability[0][1] > 0.9:
					x_diff += x - max_x
					y_diff += y -  max_y
		if max_x > 0 and max_y > 0:
			# print(max_x)
			# print(max_y)
			x_ans = (max_x+25+x_diff)/500
			y_ans = (max_y+25+y_diff)/500
			print(x_ans, y_ans )
			# print((dc.Decimal(max_x)+dc.Decimal(25)+dc.Decimal(x_diff))/dc.Decimal(500), dc.Decimal(max_y+25+y_diff)/dc.Decimal(500))
			cv2.rectangle(clone, (max_x, max_y), (max_x + winW, max_y + winH), (0, 0, 255), 2)
			cv2.imshow("Window", clone)
			cv2.waitKey(1)
			count+=1
		else:
			print("No Phone")
		
		if count:
			phone_count+=1
		
	print('phone_count',phone_count)
	# print("accuracy = ", phone_count/rows)

if __name__ == "__main__":
    main()