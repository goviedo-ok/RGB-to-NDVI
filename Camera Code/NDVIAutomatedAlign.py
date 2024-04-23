import os
import cv2
import time
from PIL import Image
from pathlib import Path
import numpy as np
import math
import matplotlib.pyplot as plt

def main():
	# Check files
	image_path = Path('/media/picam/Samsung256GB/NDVIInput')
	save_path = Path('/media/picam/Samsung256GB/NDVIOutput') # Training data
	save_path2 = Path('/media/picam/Samsung256GB/Red-NIR') # NIR and RED for presentations
	
	# Define file extensions
	readformat = '.tiff'
	saveformat = '.tiff'
	
	# Define color scale 0 = "standard" | 1 = low-NDVI grayscale | NOTE - '1' is much easier to look at
	colorScale = 1
	
	# Define output image dimensions (Note, use 4:3 aspect ratio)
	width = 1600
	height = 1200
	
	# Define intermediate image dimensions
	# This typically improves output image quality
	# Note, increases computation time drastically!!
	# This value must be >= output image dimensions and equivalent aspect ratio
	intScaleFactor = 1 # Factor to scale each dimension by
	
	width1 = intScaleFactor * width
	height1 = intScaleFactor * height
	
	# Define blur type and amount. None = 0 | Gaussian = 1 | Median = 2
	# Median greatly reduces "salt and pepper" noise, Gaussian suppresses high-freq noise
	blurtype = 0
	blur = 3 # MUST BE POSITIVE AND ODD

	# Define bit depth
	bitdepth = 8
	
	maxbit = 2 ** bitdepth - 1
	# print(maxbit)

	# Alignment params
	max_features = 5000
	feature_retention = 0.1
	
	# Crop factor (Image alignment leaves weird edges) | 1 = no crop
	cropfactor = 0.9
	
	# Define beginning extension number (useful for restarts). Default = 1
	startnum = 1
	
	# Define maximum extension number to check
	# NOTE - Only applicable if files to be processed DON'T start at index 1
	# or if file numbering skips values (useful if certain images have been removed from the set)
	extensionMax = 200

	# Camera resolution, don't change
	xresolution = 4656
	yresolution = 3496
	
	
	k = startnum
	
	while True:
		while Path((str(image_path) + '/' + 'RGB' + str(k) + readformat)).is_file():
			print('Image file with extension ' + str(k) + ' found! Aligning images...')
			
			# Open images
			img1 = cv2.imread(str(image_path) + '/' + 'NIR' + str(k) + readformat) # NIR
			img2 = cv2.imread(str(image_path) + '/' + 'RED' + str(k) + readformat) # RED
			img3 = cv2.imread(str(image_path) + '/' + 'RGB' + str(k) + readformat) # RGB
	
			# Correct for cv2 BGR
			img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
			img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
			img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
			
			# print(np.mean(img1[:,:,0]))
			# print(np.mean(img1[:,:,1]))
			# print(np.mean(img1[:,:,2]))
			
			#plt.figure()
			#plt.imshow(img1, cmap='gray') # NIR
			#plt.figure()
			#plt.imshow(img2, cmap='gray') # RED
			# plt.figure()
			# plt.imshow(img1)
			# plt.figure()
			# plt.imshow(img2) # RGB
			# plt.show()
	
			# Align images, reference RGB
			try:

				#matrix = np.array([[9.89931366e-01, 1.59708027e-02, 1.20612198e+02], [2.59678626e-02, 1.03448759e+00, -1.31403663e+02], [-6.49244230e-06, 1.82158599e-05, 1.00000000e+00]])
				img2aligned, warp_matrix = featureAlign(img2, img3, max_features, feature_retention, 'RED')
				# print(warp_matrix)
			
				#matrix = np.array([[9.94595751e-01, -2.97333138e-02, 7.68656960e+01], [2.89631775e-02, 1.00450363e+00, -9.46807410e+01], [-5.48662557e-07, 2.92750337e-06, 1.00000000e+00]])
				img1aligned, warp_matrix = featureAlign(img1, img3, max_features, feature_retention, 'NIR')
				# print(warp_matrix)
			
				img1 = img1aligned
				img2 = img2aligned

			except:
				print('Error aligning images with CV2, possible image error. Please check input images with extention ' + str(k))
				k = k + 1
				continue
				# Define image shift between cameras
				# Find a feature, then put its XY value here
				# Note that this assumes negligible disortion
			
				# img1x = 2914
				# img1y = 2250
				# img2x = 2835
				# img2y = 2283
				# img3x = 2898
				# img3y = 2238
				
				# # Pick a reference value
				# referencex = min(img1x, img2x, img3x)
				# referencey = min(img1y, img2y, img3y)
			
				# img1xshift = referencex - img1x
				# img1yshift = referencey - img1y
				# img2xshift = referencex - img2x
				# img2yshift = referencey - img2y
				# img3xshift = referencex - img3x
				# img3yshift = referencey - img3y
				
				# T1 = np.float32([[1, 0, img1xshift], [0, 1, img1yshift]])
				# T2 = np.float32([[1, 0, img2xshift], [0, 1, img2yshift]])
				# T3 = np.float32([[1, 0, img3xshift], [0, 1, img3yshift]])
		
				# print(T1)
				# print(T2)
				# print(T3)
		
				# img1 = cv2.warpAffine(img1, T1, (xresolution, yresolution))
				# img2 = cv2.warpAffine(img2, T2, (xresolution, yresolution))
				# img3 = cv2.warpAffine(img3, T3, (xresolution, yresolution))
				
			# Crop images
			x = 0
			y = 0
			
			finalx = xresolution * cropfactor
			minx = int((xresolution - finalx) / 2)
			maxx = int(xresolution - minx)
			
			finaly = yresolution * cropfactor
			miny = int((yresolution - finaly) / 2)
			maxy = int(yresolution - miny)
			
			img1 = img1[miny:maxy, minx:maxx]
			img2 = img2[miny:maxy, minx:maxx]
			img3 = img3[miny:maxy, minx:maxx]
			
			# Resize
			img1 = cv2.resize(img1, (width1, height1))
			img2 = cv2.resize(img2, (width1, height1))
			img3 = cv2.resize(img3, (width1, height1))
			
			# Convert NIR and RED to greyscale
			# Note weights for NIR are intentional
			# img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
			# img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
			
			img1 = np.round((56/102) * img1[:,:,0] + (13/102) * img1[:,:,1] + (33/102) * img1[:,:,2])
			img2 = img2[:,:,0] # Only read red channel
	
			# plt.figure()
			# plt.imshow(img1, cmap='gray') # NIR
			# plt.figure()
			# plt.imshow(img2, cmap='gray') # RED
			# plt.figure()
			# plt.imshow(img3) # RGB
			# plt.show()
	
			# Assign to arrays
			img1 = np.array(img1) # NIR
			img2 = np.array(img2) # RED
			img3 = np.array(img3) # RGB
	
			# Remove NaN
			img1 = np.nan_to_num(img1)
			img2 = np.nan_to_num(img2)
	
			# Create new blank image
			NDVI = np.zeros((img1.shape[0], img1.shape[1], 3), dtype=np.uint8)
	
			# Apply contrast stretch to both images		
			print('Images aligned successfully, stretching images...')
			img1 = contrast_stretch(img1)
			img2 = contrast_stretch(img2)
			print('Image stretch complete, performing NDVI operation...')
			
			l = k
			# Save NIR and RED if needed
			while Path(str(save_path2) + '/' + 'NIR' + str(l) + saveformat).is_file():
				l = l + 1
			
			cv2.imwrite(str(save_path2) + '/' + 'NIR' + str(l) + saveformat, img1)
			cv2.imwrite(str(save_path2) + '/' + 'RED' + str(l) + saveformat, img2)
			
			# plt.figure()
			# plt.imshow(img1, cmap='gray') # NIR
			# plt.figure()
			# plt.imshow(img2, cmap='gray') # RED
			# plt.figure()
			# plt.imshow(img1s, cmap='gray') # NIR
			# plt.figure()
			# plt.imshow(img2s, cmap='gray') # RED
			# plt.figure()
			# plt.imshow(img3) # RGB
			# plt.show()
	
			# Work through image to apply NDVI Formula (NDVI = (NIR - RED) / (NIR + RED))
			for i in range(img1.shape[0]):
				for j in range(img1.shape[1]):
					#r1, g1, b1 = img1[i, j]
					#r2, g2, b2 = img2[i, j]
					
					#NIR = (r1 + g1 + b1) / 3
					#RED = (r2 + g2 + b2) / 3
					
					NIR = float(img1[i, j])
					RED = float(img2[i, j])
					
					# if NIR < 0:
						# NIR = 0
					# if RED < 0:
						# RED = 0
					# print(RED)
					# print(NIR)
					pixelVal = (NIR - RED) / (NIR + RED)
					
					if math.isnan(pixelVal):
						pixelVal = 0
					
					# print(pixelVal)
					
					r = 0
					g = 0
					b = 0
					
					# Standard NDVI color mapping
					if colorScale == 0:
						if pixelVal > -0.7 and pixelVal <= 0:
							b = (maxbit * (pixelVal / -0.7))
							r = maxbit - b
							
						if pixelVal > 0:
							g = maxbit * pixelVal
							r = maxbit - g
							
						if pixelVal <= -0.7:
							b = maxbit - (maxbit * ((pixelVal + 0.7) / (-0.3)))
						
					# Color mapping with grayscale for NDVI<0
					threshold2 = 1 # Healthy threshold. Default = 1
					threshold1 = 0.3 # Poor health threshold. Default = 0.2
					threshold = 0.2 # Non-living threshold. Defualt = 0
					healthsens = 1 # Makes pixelval nonlinear. Default = 1
					if colorScale == 1:
						
						pixelVal1 = pixelVal ** healthsens
						pixelVal1 = np.real(pixelVal1)
						if pixelVal1 <= threshold:
							r = maxbit * ((pixelVal1 - threshold) / (-1 - threshold))
							g = maxbit * ((pixelVal1 - threshold) / (-1 - threshold))
							b = maxbit * ((pixelVal1 - threshold) / (-1 - threshold))
							
						if pixelVal1 > threshold1 and pixelVal1 <= threshold2:
							g = maxbit * (((pixelVal1) - threshold1) / (threshold2 - threshold1))
							r = maxbit - g
							
						if pixelVal1 > threshold2:
							g = maxbit
							
						if pixelVal1 > threshold and pixelVal1 <= threshold1:
							r = maxbit * ((pixelVal1 - threshold) / (threshold1 - threshold))
							
						# if pixelVal <= 0:
							# r = (maxbit * pixelVal * -1)
							# g = (maxbit * pixelVal * -1)
							# b = (maxbit * pixelVal * -1)
		
					
					NDVI[i, j] = [r, g, b]
	
			# Apply blur
			if blurtype == 1:
				NDVI = cv2.GaussianBlur(NDVI, (blur, blur), 0)
				
			if blurtype == 2:
				NDVI = cv2.medianBlur(NDVI, blur)
				
			# Resize to final dimensions
			if height1 > height:
				NDVI = cv2.resize(NDVI, (width, height))
				img3 = cv2.resize(img3, (width, height))
				
			NDVI = Image.fromarray(NDVI)
			img3 = Image.fromarray(img3)
			
			h = k
			while Path(str(save_path) + '/' + 'NDVI' + str(h) + saveformat).is_file():
				h = h + 1
			
			NDVI.save(str(save_path) + '/' + 'NDVI' + str(h) + saveformat)
			img3.save(str(save_path) + '/' + 'RGB' + str(h) + saveformat)
			
			k = k + 1
					
			#plt.imshow(NDVI)
			#plt.show()
		
		# Iterate k for instance of image files not starting at 1
		if not(Path((str(image_path) + '/' + 'RGB' + str(k) + readformat)).is_file()) and k < extensionMax:
			k = k + 1
			print('Image file with extension ' + str(k-1) + ' not found. Attempting extension ' + str(k) + '.')
			if k >= extensionMax:
				print('Maximum file extension number reached')
			continue
			
		print('Successfully processed all present images')
		break
	
# (ORB) feature based alignment      
def featureAlign(im1, im2, max_features, feature_retention, graytype):
  
  # Convert images to grayscale
  #im1Gray = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
  #im2Gray = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
  if graytype == 'NIR':
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)   # queryImage
    im1Gray = np.round((56/102) * im1[:,:,0] + (13/102) * im1[:,:,1] + (33/102) * im1[:,:,2])   # trainImage
    
  if graytype == 'RED':
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    im1Gray = im1[:,:,0] # Only read red pixels
    
  im1Gray = contrast_stretch(im1Gray)
  im2Gray = contrast_stretch(im2Gray)
    
  # Detect ORB features and compute descriptors.
  orb = cv2.ORB_create(max_features)
  keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
  keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
  
  # Match features.
  matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
  matches = matcher.match(descriptors1, descriptors2, None)
  
  # Sort matches by score
  matches.sort(key=lambda x: x.distance, reverse=False)

  # Remove not so good matches
  numGoodMatches = int(len(matches) * feature_retention)
  matches = matches[:numGoodMatches]

  # Draw top matches
  imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
  #cv2.imwrite("matches.jpg", imMatches)
  
  # Extract location of good matches
  points1 = np.zeros((len(matches), 2), dtype=np.float32)
  points2 = np.zeros((len(matches), 2), dtype=np.float32)

  for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt
  
  # Find homography
  h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
  
  print("Maximum shift value = " + str(np.max(h)))
  if np.max(h) > 250:
    raise Exception('Warp matrix too aggressive!')
    
  #h = alignmentMatrix

  # Use homography
  height, width, channels = im2.shape
  im1Reg = cv2.warpPerspective(im1, h, (width, height))
  
  return im1Reg, h 
  
# def align(img1, img2, graytype):
	# MIN_MATCH_COUNT = 10
	
	# if graytype == 'NIR':
		# img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)   # queryImage
		# img2 = np.round((56/102) * img2[:,:,0] + (13/102) * img2[:,:,1] + (33/102) * img2[:,:,2])   # trainImage
	 
	# if graytype == 'RED':
		# img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
		# img2 = img2[:,:,0] # Only read red pixels
		
	# # Initiate SIFT detector
	# sift = cv.SIFT_create()
	 
	# # find the keypoints and descriptors with SIFT
	# kp1, des1 = sift.detectAndCompute(img1,None)
	# kp2, des2 = sift.detectAndCompute(img2,None)
	 
	# FLANN_INDEX_KDTREE = 1
	# index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	# search_params = dict(checks = 50)
	 
	# flann = cv.FlannBasedMatcher(index_params, search_params)
	 
	# matches = flann.knnMatch(des1,des2,k=2)
	 
	# # store all the good matches as per Lowe's ratio test.
	# good = []
	# for m,n in matches:
	    # if m.distance < 0.7*n.distance:
	        # good.append(m)
	        
	# if len(good)>MIN_MATCH_COUNT:
    # src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    # dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
 
    # M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
    # matchesMask = mask.ravel().tolist()
 
    # h,w = img1.shape
    # pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    # dst = cv.perspectiveTransform(pts,M)
 
    # img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)

# else:
    # print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    # matchesMask = None
    
    # return img2
  
def stretch(img, sigma = 3, plot_hist=False):
    stretched = np.zeros(img.shape) 
    for i in range(1):  #looping through the bands
        band = img[:,:] # copying each band into the variable `band`
        if np.min(band)<0: # if the min is less that zero, first we add min to all pixels so min becomes 0
            band = band + np.abs(np.min(band)) 
        band = band / np.max(band)
        band = band * 255 # convertaning values to 0-255 range
        if plot_hist:
            plt.hist(band.ravel(), bins=256) #calculating histogram
            plt.show()
        # plt.imshow(band)
        # plt.show()
        std = np.std(band)
        mean = np.mean(band)
        max = mean+(sigma*std)
        min = mean-(sigma*std)
        band = (band-min)/(max-min)
        band = band * 255
        # this streching cuases the values less than `mean-simga*std` to become negative
        # and values greater than `mean+simga*std` to become more than 255
        # so we clip the values ls 0 and gt 255
        band[band>255]=255  
        band[band<0]=0
        print('band',1,np.min(band),np.mean(band),np.std(band),np.max(band))
        if plot_hist:
            plt.hist(band.ravel(), bins=256) #calculating histogram
            plt.show()
        stretched[:,:] = band
    
    
    stretched = stretched.astype('int')
    return stretched
    
def stretch2(img, maxbit):
	hilim = maxbit
	lowlim = 0
	stretched = np.zeros(img.shape)
	stretched = cv2.normalize(img, stretched, 0, maxbit, cv2.NORM_MINMAX)
	# stretched = np.zeros(img.shape)
	# stretched = ((maxbit / (hilim - lowlim)) * (img - lowlim))
	# for i in range(img.shape[0]):
		# for j in range(img.shape[1]):
			# stretched[i, j] = int(stretched[i, j])
			# if stretched[i, j] > maxbit:
				# stretched[i, j] = maxbit
			# if stretched[i, j] < 0:
				# stretched[i, j] = 0
	return stretched
	
def contrast_stretch(img):
		min_percent = 2
		max_percent = 98
		
		min_val = 5
		max_val = 250
		
		lo, hi = np.percentile(img, (min_percent, max_percent))
		
		res_img = (img.astype(float) - lo) / (hi - lo)
		
		res_img = np.maximum(np.minimum(res_img*255, max_val), min_val).astype(np.uint8)
		
		return res_img
		
		return out

  
if __name__ == "__main__":
    main()
