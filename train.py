# including standard library headers
import numpy as np
import cv2
import sklearn as sk
import pylint

# including headers containing various classes
from include import find_hog

def main():
	with open('find_phone/labels.txt') as file:
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

	# print(file_name)
	# print(xstr)
	# print(ystr)

	positive_train_sample = [find_hog.find_hog_of_image((50,50), 20, 20, (10,10))]*100
	negative_train_sample = [find_hog.find_hog_of_image((500,300), 120, 200, (100,60))]*100

	for i in range(0,100):
		print(i)
		img1 = cv2.imread('find_phone/'+file_name[i])#, cv2.IMREAD_GRAYSCALE)
		
		y_pix = int(y[i]*326)
		x_pix = int(x[i]*490)
		rows_img = len(img1)
		cols_img = len(img1[0])

		if y_pix < 25:
			ycrop_min = 0
			ycrop_max = 50
		elif y_pix >= rows_img:
			ycrop_min = rows_img-1-50
			ycrop_max = rows_img-1
		else:
			ycrop_min = y_pix - 25
			ycrop_max = y_pix + 25
		
		if x_pix < 25:
			xcrop_min = 0
			xcrop_max = 50
		elif x_pix >= cols_img:
			xcrop_min = cols_img-1-50
			xcrop_max = cols_img-1
		else:
			xcrop_min = x_pix - 25
			xcrop_max = x_pix + 25

		crop_img = img1[ycrop_min:ycrop_max, xcrop_min:xcrop_max].copy()
	
		for a in range (int(y[i]*326)-25,int(y[i]*326)+25+1):
			for b in range (int(x[i]*490)-25,int(x[i]*490)+25+1):
				img1[a][b] = [255,255,255]
		img = cv2.resize(img1, (500,300))
		
		positive_train_sample[i].compute_hog_descriptor(crop_img)
		negative_train_sample[i].compute_hog_descriptor(img)
		# print(len(positive_train_sample[0].descriptor))
	
	print(len(positive_train_sample[0].descriptor))
	print(len(negative_train_sample[0].descriptor))
	# print(len(positive_train_sample))
	cv2.imshow('image',img)
	cv2.imshow('crop_image',crop_img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
if __name__ == "__main__":
	main()
# width = 490
# height = 326