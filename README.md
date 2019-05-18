## ***PHONE_DETECTION***

## The aim is to detect phone using Computer Vision Algorithm and Machine Learning. HOG features are being computed for the images and being fed to SVM.  
  
**Required packages:**  
Python3  
OpenCV3 or above that is compatible with Python3  
  
**Python3 packages required:**  
numpy  
scikit learn  
joblib  
imutils  
  
**Steps to run:**  
>git clone https://github.com/tyaga08/phone_detection.git  
cd phone_detection  
chmod +x install.sh  
./install.sh  
python3 train_phone_finder.py < path-to-the-dataset-directory >  
python find_phone.py < file-name-with-path-to-the-test-images-directory >  
  
  
