# This code is taken from the Udacity Classoom SDC "Project: Vehicle Detection & Tracking"
#  - from Lesson 29) Hog Classify
# 
#     Modifications:
#        - read in cars & non cars in main.py
#        - use entire small set instead of a smaller sample size
#        - modified the HOG classifier: 
#              - Set a random seed so when we split our test & training set each time, 
#                it will be the same each time to allow for fair comparisons
#              - removed most of extra print statements
#              - added print out score & parms
#        - added routine for searching best hog features
#        - add spatial & color features
#     TODO: Add search for good svc parms

JKM_DEBUG = False # Debug printing data flag

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
import pickle
from sklearn.svm import LinearSVC, SVC
# from sklearn.grid_search import GridSearchCV # deprecated as of v18 
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# from sklearn.cross_validation import train_test_split
# for scikit-learn >= 0.18 use:
from sklearn.model_selection import train_test_split



# Define a function to convert color spaces
def convert_color(img, color_space='RGB'):
    # Here we assume we are converting from RGB
    #  Add more if statements, otherwise
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)      
    return feature_image

# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

# Define a function to compute color histogram features 
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32 ,  bins_range=(0, 256)):
# def color_hist(img, nbins=32, bins_range=(0, 1)): # for .png files
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features



# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        feature_image = convert_color(image, color_space=color_space)
        #
        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features
    


# Reduce the sample size because HOG features are slow to compute
# The quiz evaluator times out after 13s of CPU time
# sample_size = 500
# cars = cars[0:sample_size]
# notcars = notcars[0:sample_size]


# JKM - Added an exploration of best approach
# Add parms to try here
colorspace_parms = ['YUV', 'YCrCb'] # ['RGB', 'HSV' , 'LUV', 'HLS', 'YUV',  'YCrCb']
hog_channel_parms =  [2, 'ALL'] # [0, 1, 2, "ALL"] # Can be 0, 1, 2, or "ALL"
orient_parms = [9, 10] # [9, 8, 7, 10, 11, 12]
pix_per_cell_parms = [8, 16] # [8, 16, 32, 4]
cell_per_block_parms = [2]   # [2, 4, 8, 1]
spatial_size_parms=[(32,32), (16,16)]
hist_bins_parms=[32, 16]

# JKM - Hard code this here but should be in dictionary to pickle
# TODO: Add this to pickle
spatial_feat=True 
hist_feat=True 
hog_feat=True

# a routine to sort & print the results of parm exploration 
# JKM - added 
def print_sort_explore_results(result):
    # This turns into all entries to strings np.str type
    res = np.array(result)
    res_sorted = res[res[:,0].argsort()]
    print("\n Sorted Summary Results for Parm Exploration")
    for entry in res_sorted:
        best = entry # to lazy to re-type this 
        entry = [(best[0]), best[1], int(best[2]), int(best[3]), int(best[4]),  
                 best[5], (int(best[6]), int(best[7])), int(best[8])] 
        print("  ", entry)
    # TODO? covert back to int? here ?best = res_sorted[-1]
    best = res_sorted[-1]
    best = [(best[0]), best[1], int(best[2]), int(best[3]), int(best[4]),  
           best[5], int(best[6]), int(best[7]), int(best[8])]  
    return best

# JKM - Added 
def explore_feature_extraction_parms(cars, notcars, seed=None):
    # This does an search to determine which HOG parms yield the best results.
    #  - using the small data set, 
    #  - to reduce the exploration set, the exploration is done in stages with
    #     only 1-2 parms being modified at a time , while the others are held 
    #     constant
    result=[]  # This will be tuples of the parms and the score
    #
    # Holding other parms constant, explore colorspace & hog channel 
    orient = orient_parms[0]
    pix_per_cell = pix_per_cell_parms[0]
    cell_per_block = cell_per_block_parms[0]
    #
    #spatial_size = spatial_size_parms[0]
    #hist_bins = hist_bins_parms[0]
    #
    print("\n Exploring parms: 'colorspace', 'hog_channel', 'spatial_size', 'hist_bins' ")   
    for colorspace in colorspace_parms:
        for hog_channel in hog_channel_parms:
            for spatial_size in spatial_size_parms:
                for hist_bins in hist_bins_parms: 
                    score, _, _ = run_classifier(cars, notcars, cspace=colorspace, orient=orient, 
                                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                                        hog_channel=hog_channel, seed=seed,
                                        spatial_size=spatial_size,
                                        hist_bins=hist_bins,
                                        spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
                    result.append((score, colorspace, orient, pix_per_cell, cell_per_block, hog_channel, 
                                   spatial_size[0], spatial_size[1], hist_bins))     
    # Select the colorspace & hog_channel from the best
    best = print_sort_explore_results(result)
    colorspace, hog_channel, spatial_size, hist_bins = best[1], best[5], (best[6], best[7]), best [8]
    print("\n Continuing with parms 'colorspace' & 'hog_channel' set to: ", colorspace, hog_channel)
    #
    # Explore orient parm
    result = []
    print("\n Exploring parms: 'orient' . ")
    for orient in orient_parms:
        score, _, _ = run_classifier(cars, notcars, cspace=colorspace, orient=orient, 
                                pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                                hog_channel=hog_channel, seed=seed,
                                spatial_size=spatial_size,
                                hist_bins=hist_bins,
                                spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
        result.append((score, colorspace, orient, pix_per_cell, cell_per_block, hog_channel,
                       spatial_size[0], spatial_size[1], hist_bins))
    # Select orient from the best 
    best = print_sort_explore_results(result)
    orient = best[2]
    print("\n Continuing with parm 'orient' set to: ", orient)
    #
    # Explore pix_per_cell & cell_per_block parms. Note - there are some combos that do not work,
    #  so we skip those if it causes an error. 
    print("\n Exploring parms: 'pix_per_cell' & 'cell_per_block'. ")
    result = []
    for pix_per_cell in pix_per_cell_parms:
        for cell_per_block in cell_per_block_parms:
            try:
                score, _, _ = run_classifier(cars, notcars, cspace=colorspace, orient=orient, 
                                pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                                hog_channel=hog_channel, seed=seed,
                                spatial_size=spatial_size,
                                hist_bins=hist_bins,
                                spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
                result.append((score, colorspace, orient, pix_per_cell, cell_per_block, hog_channel, 
                               spatial_size[0], spatial_size[1], hist_bins))
            except ValueError: 
               print("ERROR: Skipping", pix_per_cell, cell_per_block)
    best = print_sort_explore_results(result)
    final_score, pix_per_cell, cell_per_block = best[0], best[3], best[4]
    print("\n Final Exploration HOG Parms Obtained")  
    return [colorspace, orient, pix_per_cell, cell_per_block, hog_channel, 
            (spatial_size[0], spatial_size[1]), hist_bins]




# JKM - modified to reduce number of print statements, set a random SEED & print final results
def run_classifier(cars, notcars, cspace='RGB', orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0, 
                   seed=None, linear=True,
                   spatial_size=(32, 32), 
                   hist_bins=32, 
                   spatial_feat=True, hist_feat=True, hog_feat=True,
                   pickle=False):
    # Runs the classifier code 
    # Returns score, svc, and X_scaler 
    # 
    print("\n === Extracting Features & Running Classifier === ")
    t=time.time()
    car_features = extract_features(cars, color_space=cspace, spatial_size=spatial_size,
                        hist_bins=hist_bins, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
                        spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
    #
    notcar_features = extract_features(notcars, color_space=cspace, spatial_size=spatial_size,
                        hist_bins=hist_bins, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
                        spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
    #
    t2 = time.time()
    # print(round(t2-t, 2), 'Seconds to extract HOG features...')
    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    #
    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
    #
    # Split up data into randomized training and test sets
    if seed != None: 
        if JKM_DEBUG: print("\n  ** Using seed to get predictable random train & test split")
        np.random.seed(seed)
    rand_state = np.random.randint(0, 100)
    # print("Random State = ", rand_state)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)
    #
    # print('Using:',orient,'orientations',pix_per_cell,
    #    'pixels per cell and', cell_per_block,'cells per block')
    # print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC 
    if linear:
        svc = LinearSVC() 
    else:
        svc = SVC(C=10.0, kernel='rbf', verbose=True)
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    # print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    score = np.round(svc.score(X_test, y_test), 4) # calculate to 4 decimal points
    # print('Test Accuracy of SVC = ', score)
    # Check the prediction time for r_all = explore_feature_extraction_parms()a single sample
    #t=time.time()
    #n_predict = 10
    #print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    #print('For these',n_predict, 'labels: ', y_test[0:n_predict])
    #t2 = time.time()
    #print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')
    # print(" Results: score cspace orient pix_per_cell cell_per_block hog_channel spatial_size hist_bins ")
    print(" \n Results: ", score, cspace, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins) 
    # print("===========================================\n")
    #
    return score, svc, X_scaler


PICKLE_FILE_NAME = 'svc_car.p'
def get_pickle_data():
    # Check if have already pickled the results
    car_dict = {}
    try:
        f = open(PICKLE_FILE_NAME, 'rb')
        car_dict = pickle.load(f)
        f.close()
        if JKM_DEBUG: print("Data obtained from pickle file: ", PICKLE_FILE_NAME)    
    except FileNotFoundError:
        print("\n       ERROR: Pickle file not found. ")
    #
    return car_dict


def run_classifier_and_pickle(cars, notcars, best, seed):
    # Runs the classifier with the selected parms and pickles the results.
    # returns the dictionary
    colorspace, hog_channel, spatial_size, hist_bins = best[0], best[4], best[5], best [6] #NOTE spatial_size already combined
    orient, pix_per_cell, cell_per_block = best[1], best[2], best[3]
    score, svc, X_scaler = run_classifier(cars, notcars, cspace=colorspace, orient=orient, 
                                pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                                hog_channel=hog_channel, seed=seed,
                                spatial_size=spatial_size,
                                hist_bins=hist_bins,
                                spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat,
                                pickle=True)
    #
    print("\n  Pickling feature extraction parms and classifier to: ", PICKLE_FILE_NAME)
    car_dict = {'svc': svc, 'scaler': X_scaler, 
                'colorspace': colorspace, 'hog_channel': hog_channel, 
                'spatial_size':spatial_size, 'hist_bins': hist_bins, 
                'orient':orient, 'pix_per_cell': pix_per_cell, 'cell_per_block':cell_per_block 
                }
    f = open(PICKLE_FILE_NAME, 'wb')   # 'wb' instead 'w' for binary file
    pickle.dump(car_dict, f, -1)  # -1 specifies highest binary protocol
    f.close() 
    # 
    return car_dict




# This takes a very long time to run.
# Do not invoke
#  Conclusion - results are that there is not much advantage at this point to 
#   running a non linear kernel due to huge increase in classification time. 
#   Such a small increas in classification accuracy is not worth the substantially
#   longer time.  
def explore_classifier_parms(cars, notcars, 
                             cspace='YUV', orient=10, pix_per_cell=4, cell_per_block=4, hog_channel='ALL',
                             svm_parms = {'kernel':('linear', 'rbf'), 'C':[1, 10]} ): 
    # 
    # Explores to find best SVM parms 
    #   - using the parameters passed in to invoke grid search
    #   - first extract features and build test data & target
    # 
    print("\n Extracting Features")
    # t=time.time()
    #
    car_features = extract_features(cars, cspace=cspace, orient=orient, 
                            pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                            hog_channel=hog_channel)
    notcar_features = extract_features(notcars, cspace=cspace, orient=orient, 
                            pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                            hog_channel=hog_channel)
    #t2 = time.time()
    # print(round(t2-t, 2), 'Seconds to extract HOG features...')
    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    #
    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
    #
    print(" \n Exploring SVM parms via gridsearch")
    svr = SVC() 
    clf = GridSearchCV(svr, svm_parms)
    clf.fit(scaled_X, y) 
    #print(" Results: ", score, cspace, orient, pix_per_cell, cell_per_block, hog_channel) 
    # print("=================================\n")
    #
    return clf.best_params_


# To test explore SVC parms , uncomment the following
# Hardcode these for now based on hog parm exploration 
# cspace, orient, pix_per_cell, cell_per_block, hog_channel = 'YUV', 10, 4, 4, 'ALL'
# To test SVC
# run_classifier(cars, notcars, cspace='YUV', orient=10, pix_per_cell=4, cell_per_block=4, hog_channel='ALL', seed=42)

