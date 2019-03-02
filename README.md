##################################################  
################## PHONE_DETECTION ###############  
##################################################  

The aim is to detect phone using Computer Vision Algorithm and Machine Learning. HOG features are being computed for the images and being fed to SVM.  
  
Required packages:  
Python 3  
OpenCV 3 or above that is compatible with Python 3  
  
PIP packages required:  
numpy  
scikit learn  
joblib  
imutils  
  
  
Steps to run:  
	-> Clone this repository and go to the folder  
	-> run this command for training the model - python train_phone_finder.py path-to-the-dataset-directory  
	-> run this command for testing - python find_phone.py file-name-with-path-to-the-test-images-directory  
	
