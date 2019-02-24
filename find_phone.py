# including standard library headers
import numpy as np
import cv2
import matplotlib as plt
from joblib import dump, load

# including headers containing various classes
from include import find_hog
from include import svm_classifier



def main():
    img1 = cv2.imread('find_phone/4.jpg')
    img = cv2.resize(img1, (500,300))

    # img = cv2.resize(img1, (450,450))
    test_hog = find_hog.find_hog_of_image((100,60),24,40,(20,12))
    # test_hog = find_hog.find_hog_of_image((45,45),18,18,(9,9))
    test_hog.compute_hog_descriptor(img)

    print(len(test_hog.descriptor))
    # print(test_hog.descriptor)
    check_for_phone = np.reshape(test_hog.descriptor,(len(test_hog.descriptor)//576,576))
    clf = load('HOG_SVM_training.joblib')
    print(len(check_for_phone[0]))
    print(len(check_for_phone))
    prediction = 0
    for i in range(0,len(check_for_phone)):
        prediction = clf.predict([check_for_phone[i]])
        print(prediction)
        if prediction == 1:
            print("Phone present")
            break
    
    if prediction == 0:
        print("No phone")

    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()