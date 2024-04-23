# This script takes the contents of 2 folders and writes it to
# one folder. Useful if 2 folders share the same file names
# Currently set up to combine NDVI outputs from separate runs

import os
import cv2
from pathlib import Path

def main():
	readformat = '.tiff'
	read_path1 = Path('/media/picam/Samsung256GB/NDVIOutput 4-11-24')
	# read_path2 = Path('/media/picam/Samsung256GB/NDVIOutput 3-10-24')
	save_path = Path('/media/picam/Samsung256GB/Training')
	i = 1
	j = 1
	extensionMax = 200

	while True:
		while Path((str(read_path1) + '/' + 'RGB' + str(i) + readformat)).is_file():
			img1 = cv2.imread(str(read_path1) + '/' + 'RGB' + str(i) + readformat)
			img2 = cv2.imread(str(read_path1) + '/' + 'NDVI' + str(i) + readformat)
			save(save_path, img1, 'RGB')
			save(save_path, img2, 'NDVI')
			i = i + 1
		
		if not(Path((str(read_path1) + '/' + 'RGB' + str(i) + readformat)).is_file()) and i < extensionMax:
			i = i + 1
			print('Image file with extension ' + str(i-1) + ' not found. Attempting extension ' + str(i) + '.')
			if i >= extensionMax:
				print('Maximum file extension number reached')
				break
			continue
			
	# while True:
		# while Path((str(read_path2) + '/' + 'RGB' + str(j) + readformat)).is_file():
			# img1 = cv2.imread(str(read_path2) + '/' + 'RGB' + str(j) + readformat)
			# img2 = cv2.imread(str(read_path2) + '/' + 'NDVI' + str(j) + readformat)
			# save(save_path, img1, 'RGB')
			# save(save_path, img2, 'NDVI')
			# j = j + 1
			
		# if not(Path((str(read_path2) + '/' + 'RGB' + str(j) + readformat)).is_file()) and j < extensionMax:
			# j = j + 1
			# print('Image file with extension ' + str(j-1) + ' not found. Attempting extension ' + str(j) + '.')
			# if j >= extensionMax:
				# print('Maximum file extension number reached')
				# break
			# continue
	
def save(image_path, image, imtype):
    i = 1
    while Path((str(image_path) + '/' + imtype + str(i) + '.tiff')).is_file():
        i = i + 1
    cv2.imwrite(str(image_path) + '/' + imtype + str(i) + '.tiff', image)
	
if __name__ == "__main__":
    main()
