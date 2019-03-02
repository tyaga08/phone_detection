#################################################################################
################ PHONE DETECTION USING HOG FEATURES AND SVM #####################
####### This the training file where the training images are used to ############
####### develop an SVM classifier. The positive imgages are phones   ############
####### to be detected and the negative images are the background.   ############
####### HOG features are fed to the classifier and a model is        ############
####### generated. Then the training images are again tested to find ############
####### any false positives using the Hard Mining Technique. The     ############
####### is stored in a file.                                         ############
#################################################################################

# type the following command on the terminal
# python train_phone_finder.py ~/find_phone

# including standard library headers
import sys
from os.path import expanduser
import numpy as np
import cv2
from joblib import dump, load

# including headers containing various classes
from include import find_hog
from include import svm_classifier

# Include library from pyimagesearch. credits: https://www.pyimagesearch.com/
from include.pyimagesearch.helpers import pyramid
from include.pyimagesearch.helpers import sliding_window

# This is the size of the sliding window, which is used for Hard Negative Mining
(winW, winH) = (50,50)

# Main function
def main(folder_name):
	print("TRAINING HAS BEGUN. PLEASE WAIT UNTIL YOU RECEIVE A MESSAGE - TRAINING COMPLETED SUCCESSFULLY.")
# Get the data from the text file and store in 3 lists: names of the files, x-position of the centre, y-position of the centre
	print("This may take a while. press and hold ctrl + c if you wish to stop.")
	home = expanduser(folder_name)
	
	with open(home + '/labels.txt') as file:
		lines = file.readlines()

	rows = len(lines)
	file_name = ['']*rows
	xstr = ['']*rows
	ystr = ['']*rows
	x = [0.0]*rows
	y = [0.0]*rows

	for i in range(0,rows):
		cols = len(lines[i])-1
		space_num = 0
		for j in range(0,cols):
			if lines[i][j] == ' ':
				space_num += 1

			else:
				if space_num == 0:
					file_name[i] += lines[i][j]

				elif space_num == 1:
					xstr[i] += lines[i][j]

				elif space_num == 2:
					ystr[i] += lines[i][j]

		x[i] = float(xstr[i])
		y[i] = float(ystr[i])

# window size is the size of the HOG window size that we are interested in
	window_size = 50

# resize x and y are the values to which the image is resized before computing the HOG descriptors
	resize_x = 500
	resize_y = 500

# Define a list of positive and negative HOG feature vectors
	positive_train_sample = [find_hog.find_hog_of_image((window_size,window_size), window_size//5*2, window_size//5*2, (window_size//5,window_size//5))]*rows
	negative_hog = find_hog.find_hog_of_image((window_size,window_size), window_size//5*2, window_size//5*2, (window_size//5,window_size//5))
	negative_train_sample = np.zeros((rows*100, 576))

# Defining the data to be sent to the classifier. X_data is the array of HOG feature vectors. y_data is the array of corresponding labels
	X_data = np.zeros((rows*(1+100),len(negative_train_sample[0])))
	y_data = np.zeros((rows*(1+100)))

	j = 0

# Finding HOG Feature vector for all images. rows is the number of lines in the text file.
	for i in range(0,rows):	
		
		img2 = cv2.imread(home + '/' + file_name[i])
		img1 = cv2.resize(img2, (resize_x,resize_y))
		
		y_pix = int(y[i]*resize_x)
		x_pix = int(x[i]*resize_y)
		rows_img = len(img1)
		cols_img = len(img1[0])

	# setting window to crop the positive image (phone) from the actual image
		if y_pix < window_size//2:
			ycrop_min = 0
			ycrop_max = window_size
		elif y_pix >= rows_img:
			ycrop_min = rows_img-1-window_size
			ycrop_max = rows_img-1
		else:
			ycrop_min = y_pix - window_size//2
			ycrop_max = y_pix + window_size//2
		
		if x_pix < window_size//2:
			xcrop_min = 0
			xcrop_max = window_size
		elif x_pix >= cols_img:
			xcrop_min = cols_img-1-window_size
			xcrop_max = cols_img-1
		else:
			xcrop_min = x_pix - window_size//2
			xcrop_max = x_pix + window_size//2
	
	# cropping the image
		crop_img = img1[ycrop_min:ycrop_max, xcrop_min:xcrop_max].copy()

	# Finding and reshaping the HOG descriptor of the positive image
		positive_train_sample[i].compute_hog_descriptor(crop_img)
		temp = np.reshape(positive_train_sample[i].descriptor, (1,576))
		
	# Adding the HOG feature vector and label in the dataset
		X_data[i] = temp[0]
		y_data[i] = 1

	# Adding a mask covering the positive image in the actual image. This makes the negative image ready for training
		for a in range (int(y[i]*resize_y)-window_size//2-20,int(y[i]*resize_y)+window_size//2+1+20):
			for b in range (int(x[i]*resize_x)-window_size//2-20,int(x[i]*resize_x)+window_size//2+1+20):
				if 0<=a<resize_y and 0<=b<resize_x:
					img1[a][b] = [19,69,139]
	
	# The sliding window function is used to find the HOG feature vector inside the negative image with the window size
	# as mentioned above. stepSize makes the window jump steps rather than repeating the steps.
		for (p, q, window) in sliding_window(img1, stepSize=50, windowSize=(winW, winH)):
		# if the window does not meet our desired window size, ignore it
			if window.shape[0] != winH or window.shape[1] != winW:
				continue
			negative_hog.compute_hog_descriptor(window)
			check_for_phone = np.reshape(negative_hog.descriptor,(1,576))
			negative_train_sample[j] = check_for_phone[0]
			j+=1

# Negative feature vectors being added to the dataset
	for i in range(0,rows*100):
		X_data[i+rows] = negative_train_sample[i-rows]

# Train using the SVM
	svclf = svm_classifier.svm_clf(X_data,y_data)
	svclf.svm_train()


# Hard Negative Mining for removing the false positives
# Defining the necessary variables 
	hnm_data = np.zeros(())
	hnm_not_empty = False
	hnm_hog = find_hog.find_hog_of_image((window_size,window_size), window_size//5*2, window_size//5*2, (window_size//5,window_size//5))

# Running loop over all the lines from the text file
	for i in range(0,rows):
		img1 = cv2.imread(home+ '/' + file_name[i])
		img = cv2.resize(img1, (500,500))
	# Again adding a mask to cover the positive part
		for a in range (int(y[i]*resize_y)-window_size//2-20,int(y[i]*resize_y)+window_size//2+1+20):
			for b in range (int(x[i]*resize_x)-window_size//2-20,int(x[i]*resize_x)+window_size//2+1+20):
				if 0<=a<resize_y and 0<=b<resize_x:
					img[a][b] = [19,69,139]
		prediction = 0
		count = 0
		clone = img.copy()
	# window function with lesser step size to detect the false positives
		for (p, q, window) in sliding_window(img, stepSize=10, windowSize=(winW, winH)):
		# if the window does not meet our desired window size, ignore it
			if window.shape[0] != winH or window.shape[1] != winW:
				continue
			hnm_hog.compute_hog_descriptor(window)
			check_for_phone = np.reshape(hnm_hog.descriptor,(1,576))
			prediction = svclf.clf.predict(check_for_phone)
			if prediction:
				if (not hnm_not_empty):
					hnm_data = check_for_phone
					hnm_not_empty = True
				else:
					hnm_data = np.vstack((hnm_data,check_for_phone))

# Adding the false positives from the Hard Negative Mining to the previous dataset
	if hnm_not_empty:
		hnm_label = np.zeros((len(hnm_data)))
		X_data = np.vstack((X_data,hnm_data))
		y_data = np.hstack((y_data, hnm_label))

	#	Again training the classifier 
		svclf.svm_train(flag=True, train_data=X_data, train_label=y_data)

# Saving the classifier model to a file. This file would be used by the find_phone.py file
	dump(svclf.clf, 'classifier_model/HOG_SVM_training.joblib')

	print("TRAINING COMPLETED SUCCESSFULLY")

if __name__ == "__main__":
	main(sys.argv[1])
