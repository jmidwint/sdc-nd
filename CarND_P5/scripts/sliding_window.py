# This code is taken from the Udacity Classoom SDC "Project: Vehicle Detection & Tracking"
#  - from Lesson 32) Sliding Window Implementation
#
#  Modifications:
#  - added code to create and display multiple windows of varying size, forprottyping purposes
#  - added search_windows, copied from Lesson 34- Search & Classify 
#  - added singleimage_features - copied from Lesson 34- Search & Classify  

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



# from hog_classify import get_hog_features
from feature_explore import get_hog_features, bin_spatial, color_hist, convert_color

#
#
# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):    
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    feature_image = convert_color(img, color_space=color_space)
    #
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))      
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)
    # 
    #9) Return concatenated array of features
    return np.concatenate(img_features)

# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
JKM_DEBUG = False
def search_windows(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):
    #
    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
            if JKM_DEBUG: print("Found a match at: ", window)
    #8) Return windows for positive detections
    return on_windows
    
    



# ============================================
#image = mpimg.imread('bbox-example-image.jpg')
image = mpimg.imread('../examples/bbox-example-image.jpg')

# Here is your draw_boxes function from the previous exercise
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy
    
    

# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

# Used for testing - comment for now
#windows = slide_window(image, x_start_stop=[None, None], y_start_stop=[y_start, None], 
#                    xy_window=(128, 128), xy_overlap=(0.5, 0.5))                       
#window_img = draw_boxes(image, windows, color=(0, 0, 255), thick=6)                    
#plt.imshow(window_img)

# JKM 
# Experimenting with window sizes & ranges
# Might be more straight forward to just hard code this 

Y_START_P = 0.6  # Eliminate 60% of the search space in the y direction
y_start = int( Y_START_P * image.shape[0])  # actual y value for starting
y_remain = image.shape[0] - y_start # Size of remaining search space in y direction
 

W_MIN = 96    # smallest search window size
W_MAX = 200   # largest window size
W_STEP = 32   # amount to increase window size per search


Y_REM_MAX_P = 1.0
Y_REM_MIN_P = 0.5
Y_REM_STEP_P = 0.15

y_rem_max = int(Y_REM_MAX_P * y_remain)
y_rem_min = int(Y_REM_MIN_P * y_remain)
y_rem_step = int(Y_REM_STEP_P * y_remain) 
y_rem_list = list(range(y_rem_min, y_rem_max, y_rem_step))

w_size_list = list(range(W_MIN, W_MAX, W_STEP))  #75 A Range of window sizes to try 
# w_stop_list = [i* W_NUM for i in w_size_list]    # the window stop distance 
y_stop_list = [y_start + W_STEP + i for i in y_rem_list] # the y stop number based on numwindows and y_start  

# Get a bunch of different sized windows
def get_windows(image):
    # loop through the various window sizes
    w_list = []
    for w_size, y_stop in zip (w_size_list, y_stop_list):  
        windows = slide_window(image, x_start_stop=[None, None], y_start_stop=[y_start, y_stop], 
                            xy_window=(w_size, w_size), xy_overlap=(0.50, 0.50))
        w_list.append(windows)
    # Flatten my list - JKM - find faster method?
    flat_list_of_windows = [item for sublist in w_list for item in sublist] 
    return flat_list_of_windows

# Added this for testing
# get a list of the different sized boxes and display them on a image
#w_list = get_windows(image)  # Move this to main eventually ?
# boxes_image = np.copy(image)

# Draws boxes from a list of list of boxes
# should be necessary anymore as I will flatten the list above 
def draw_all_boxes(boxes_image, w_list):
    for w in w_list:                       
        boxes_image = draw_boxes(boxes_image, w, color=(0, 0, 255), thick=6) 
    return boxes_image
# window_img = draw_all_boxes(boxes_image, w_list) 

#window_img =                   
# plt.imshow(window_img)




