# This code is taken from the Udacity Classoom SDC "Project: Vehicle Detection & Tracking"
#  - from Lesson 35) Hog Sub-sampling Window Search
# 
#     Modifications:
#     - not using their pickled file 
#     - not using their lesson_function.py for now
#         get_hog_features from my hog_classify.py
#     - modified their colour conversion proc

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
# import pickle
import cv2
# from lesson_functions import *
# from hog_classify import get_hog_features
# from lessons2 import convert_color, get_hog_features

from feature_explore import convert_color, get_hog_features, bin_spatial, color_hist

#dist_pickle = pickle.load( open("svc_pickle.p", "rb" ) )
#svc = dist_pickle["svc"]
#X_scaler = dist_pickle["scaler"]
#orient = dist_pickle["orient"]
#pix_per_cell = dist_pickle["pix_per_cell"]
#cell_per_block = dist_pickle["cell_per_block"]
#spatial_size = dist_pickle["spatial_size"]
#hist_bins = dist_pickle["hist_bins"]

# use my otherimage for now
# img = mpimg.imread('test_image.jpg')

JKM_DEBUG = False
# Define a single function that can extract features using hog sub-sampling and make predictions
#   - Returns : the image with the matches drawn
#         + tracking of all the found matches (boxes where cars are found)
#         + confidence factor per match
#         + all the places that were searched
#    
def find_cars(img, ystart, ystop, scale, svc, X_scaler, 
               orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, colorspace):
    #
    if JKM_DEBUG: print("JKM: passed in :" , ystart, ystop, scale, svc, X_scaler, 
               orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, colorspace) 
    # 
    draw_img = np.copy(img)
    img = img.astype(np.float32)/255  # JKM - need to scale if classifier was trained earlier to do this
    #
    img_tosearch = img[ystart:ystop,:,:]
    # ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb') # I was training using YUV
    ctrans_tosearch = convert_color(img_tosearch, color_space=colorspace) # JKM 
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
    #         
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]
    #
    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    # 
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    #  JKM - I do not know how this 64 is obtained ? 
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step   # JKM  + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    #
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    #
    # JKM - add some set up for tracking
    match_list = []       # to track the windows where there is a match for a car detected
    conf_list = []        # to track how confident the prediction was
    search_list = []      # to track where the search took place
    #
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            #
            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell
            #
            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
            #         
            # Get spatial & color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)
            #
            # Scale features and make a prediction
            test_features = X_scaler.transform(
                                 np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            #
            # JKM test_features = X_scaler.transform(np.hstack((hog_features)).reshape(1, -1))  
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            conf_score = svc.decision_function(test_features)  # also capture the confidence in the prediction
            #
            #
            # 
            xbox_left = np.int(xleft*scale)
            ytop_draw = np.int(ytop*scale)
            win_draw = np.int(window*scale)
            win_loc = ((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart))
            # if (test_prediction == 1 and conf_score > 0.3):  # JKM - just try this value 
            if (test_prediction == 1):           
                # cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6)
                cv2.rectangle(draw_img,win_loc[0], win_loc[1],(0,0,255),6) 
                #
                # Add some tracking here                
                match_list.append(win_loc)
                conf_list.append(conf_score)
                if JKM_DEBUG: print("JKM found a match. Confidence: ", conf_score) 
            #
            #
            # Keep track of where we searched
            search_list.append(win_loc) 
    #             
    #
    #        
    return draw_img, match_list, conf_list, search_list


''' JKM    
ystart = 400
ystop = 656
scale = 1.5
    
out_img = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

plt.imshow(out_img)
'''
