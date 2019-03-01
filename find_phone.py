# including standard library headers
import numpy as np
import cv2
import matplotlib as plt
from joblib import dump, load
from sklearn import svm

from pyimagesearch.helpers import pyramid
from pyimagesearch.helpers import sliding_window


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
	clf = load('HOG_SVM_training.joblib')
	
	for l in range(0,rows):
		img1 = cv2.imread('find_phone/'+file_name[l])
		print('find_phone/'+file_name[l])
		img = cv2.resize(img1, (500,500))
		prediction = 0
		count = 0
		clone = img.copy()
		for (x, y, window) in sliding_window(img, stepSize=10, windowSize=(winW, winH)):
		# if the window does not meet our desired window size, ignore it
			if window.shape[0] != winH or window.shape[1] != winW:
				continue
			test_hog.compute_hog_descriptor(window)
			check_for_phone = np.reshape(test_hog.descriptor,(1,576))
			prediction = clf.predict(check_for_phone)
			if prediction:
				count+=1
				cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
				cv2.imshow("Window", clone)
				cv2.waitKey(1)
		if count:
			phone_count+=1
		
	print('phone_count',phone_count)
	# print("accuracy = ", phone_count/rows)

if __name__ == "__main__":
    main()