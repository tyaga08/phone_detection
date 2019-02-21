import numpy as np
import cv2
import sklearn as sk

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

print(file_name)
print(xstr)
print(ystr)



img = cv2.imread('find_phone/'+file_name[0])
crop_img = img[int(y[0]*326)-25:int(y[0]*326)+25, int(x[0]*490)-25:int(x[0]*490)+25]

cv2.imshow('image',img)
cv2.imshow('crop_image',crop_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Added this comment just to check

# x = 406.994
# y = 44.01