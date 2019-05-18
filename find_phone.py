#############################################################################
############ Finding the phone using the pre-trained classifier #############
#############################################################################

# send the python command as  python find_phone.py ~/find_phone_test_images/image_name_and_format

# including standard library headers
import numpy as np
import sys
from os.path import expanduser
import cv2
from joblib import dump, load
from sklearn import svm

# Include library from pyimagesearch. credits: https://www.pyimagesearch.com/
from include.pyimagesearch.helpers import pyramid
from include.pyimagesearch.helpers import sliding_window

# including headers containing various classes
from include import find_hog
from include import svm_classifier

# main function
def main(file_name):
	home = expanduser(file_name)

# setting up the window size for sliding window
	(winW, winH) = (50, 50)

# Defining HOG object to finding phone
	test_hog = find_hog.find_hog_of_image((50,50),20,20,(10,10))
	
# Loading the classifier model that was trained in the train_phone_finder.py
	clf = load('classifier_model/HOG_SVM_training.joblib')

	img1 = cv2.imread(file_name)
	img = cv2.resize(img1, (500,500))

# Defining set of variables required to localize the phone
	prediction = 0
	clone = img.copy()
	max_x = 0.0
	max_y = 0.0
	x_diff = 0.0
	y_diff = 0.0
	max_probability = 0.0

# Sliding window function
	for (x, y, window) in sliding_window(img, stepSize=10, windowSize=(winW, winH)):
	# if the window does not meet our desired window size, ignore it
		if window.shape[0] != winH or window.shape[1] != winW:
			continue
		test_hog.compute_hog_descriptor(window)
		check_for_phone = np.reshape(test_hog.descriptor,(1,576))
	# The prediction of each window if the phone is present or not. If yes, the probability estimate is checked
	# and it is localized at the location of maximum probability estimate (Non-maximal supptression)
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

# Localizing the phone in the image
	if max_x > 0 and max_y > 0:
		x_ans = (max_x+25+x_diff)/500
		y_ans = (max_y+25+y_diff)/500
		print(x_ans, y_ans )

		#Uncomment these statementsto visualize the algorithm
		cv2.rectangle(clone, (max_x, max_y), (max_x + winW, max_y + winH), (0, 0, 255), 2)
		cv2.imshow("Window", clone)
		cv2.waitKey(0)

# if no phone is detected
	else:
		print("No Phone")
	
	
if __name__ == "__main__":
    main(sys.argv[1])