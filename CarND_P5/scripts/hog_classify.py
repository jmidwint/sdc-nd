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
#     TODO: Add search for good svc parms

JKM_DEBUG = True # Debug printing data flag

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC, SVC
# from sklearn.grid_search import GridSearchCV # deprecated as of v18 
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# from sklearn.cross_validation import train_test_split
# for scikit-learn >= 0.18 use:
from sklearn.model_selection import train_test_split


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
def extract_features(imgs, cspace='RGB', orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    if JKM_DEBUG: print("Extracting Features Using: ",  cspace, orient, pix_per_cell, cell_per_block, hog_channel)
    for file in imgs:
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)      
        #
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
        features.append(hog_features)
    # Return list of feature vectors
    return features


# Reduce the sample size because HOG features are slow to compute
# The quiz evaluator times out after 13s of CPU time
# sample_size = 500
# cars = cars[0:sample_size]
# notcars = notcars[0:sample_size]


# JKM - Added an exploration of best approach
# Add parms to try here
colorspace_parms = ['RGB', 'HSV' , 'LUV', 'HLS', 'YUV',  'YCrCb']
orient_parms = [9, 8, 7, 10, 11, 12]
pix_per_cell_parms = [8, 16, 32, 4]
cell_per_block_parms = [2, 4, 8, 1]
hog_channel_parms = [0, 1, 2, "ALL"] # Can be 0, 1, 2, or "ALL"


# a routine to sort & print the results of parm exploration 
# JKM - added 
def print_sort_explore_results(result):
    res = np.array(result) 
    res_sorted = res[res[:,0].argsort()]
    print("\n Sorted Summary Results for Parm Exploration")
    for entry in res_sorted: print("  ", entry)
    return res_sorted[-1]

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
    print("\n Exploring parms: 'colorspace' & 'hog_channel' . ")   
    for colorspace in colorspace_parms:
        for hog_channel in hog_channel_parms:
            score = run_classifier(cars, notcars, cspace=colorspace, orient=orient, 
                            pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, seed=seed)
            result.append((score, colorspace, orient, pix_per_cell, cell_per_block, hog_channel))       
    # Select the colorspace & hog_channel from the best
    best = print_sort_explore_results(result)
    colorspace, hog_channel = best[1], best[-1]
    print("\n Continuing with parms 'colorspace' & 'hog_channel' set to: ", colorspace, hog_channel)
    #
    # Explore orient parm
    result = []
    print("\n Exploring parms: 'orient' . ")
    for orient in orient_parms:
        score = run_classifier(cars, notcars, cspace=colorspace, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, seed=seed)
        result.append((score, colorspace, orient, pix_per_cell, cell_per_block, hog_channel))
    # Select orient from the best 
    best = print_sort_explore_results(result)
    orient = int(best[2])
    print("\n Continuing with parm 'orient' set to: ", orient)
    #
    # Explore pix_per_cell & cell_per_block parms. Note - there are some combos that do not work,
    #  so we skip those if it causes an error. 
    print("\n Exploring parms: 'pix_per_cell' & 'cell_per_block'. ")
    result = []
    for pix_per_cell in pix_per_cell_parms:
        for cell_per_block in cell_per_block_parms:
            try:
                score = run_classifier(cars, notcars, cspace=colorspace, orient=orient, 
                                pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                                hog_channel=hog_channel, seed=seed)
                result.append((score, colorspace, orient, pix_per_cell, cell_per_block, hog_channel))
            except ValueError: 
               print("ERROR: Skipping", pix_per_cell, cell_per_block)
    best = print_sort_explore_results(result)
    print("\n Final Exploration HOG Parms Obtained") 
    # pix_per_cell, cell_per_block = int(best[3]), int(best[4]) 
    return best

# JKM - modified to reduce number of print statements, set a random SEED & print final results
def run_classifier(cars, notcars, cspace='RGB', orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0, 
                   seed=None, linear=True):
    # Runs the classifier code 
    # return score (for now) 
    # 
    print("\n === Extracting Features & Running Classifier === ")
    t=time.time()
    car_features = extract_features(cars, cspace=cspace, orient=orient, 
                            pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                            hog_channel=hog_channel)
    notcar_features = extract_features(notcars, cspace=cspace, orient=orient, 
                            pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                            hog_channel=hog_channel)
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
        if JKM_DEBUG: print("Using seed to get predictable random train & test split")
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
    print(" Results: ", score, cspace, orient, pix_per_cell, cell_per_block, hog_channel) 
    print("=================================\n")
    #
    return score


# This takes a very long time to run.
# Do not invoke
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

