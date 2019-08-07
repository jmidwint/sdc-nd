# This is the main script for the Udacity SDC Vehicle Detection Project.
#
#    This is to be run to create the final video. 
#    Vehicle training detection is already run previously? 
#       -> But for now I will do it from here until I can pickle the model & training results.
# 

import numpy as np
import cv2
import matplotlib.image as mpimg
from matplotlib import cm
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip

# System Needs
import code
import sys
import os
import glob   # to read in global names

# not needed here  
import pickle 

# Locals
from helpers import *
from draw_bboxes import draw_boxes, add_heat, apply_threshold, draw_labeled_bboxes, apply_threshold_of_confidence
# from hog_classify import *
from feature_explore import explore_feature_extraction_parms, run_classifier_and_pickle, get_pickle_data
from feature_explore import convert_color, get_hog_features, bin_spatial, color_hist
from hog_subsample import find_cars



# Set Debug flag
JKM_DEBUG=False

# make a list of test images
test_images = glob.glob("../test_images/test*.jpg")

extension='.jpg' # 
# extension='.png'   # use this to capture binary images

# List of car & non-car images for training
# JKM: TODO Move this to common area where we will run training?
# Location of images
dir_images = '../car_training_examples/'
dir_small = 'small_data_set/'
dir_all = 'all/'
dir_car = 'vehicles/'
dir_non_car = 'non-vehicles/'
dir_large = 'large_data_set/'

# Get the image names
#  This is the small data set. This was used originally to get best parms. 
#  These are jpg format so, they will be read in 0-255
# cars = glob.glob(dir_images + dir_small  + dir_car + dir_all + '*.jpeg')
# notcars = glob.glob(dir_images + dir_small + dir_non_car + dir_all +  '*.jpeg')

# This is the large data set.
#  These are png files which are read in with imread() as 0-1 scale
print("\n Large Data set in Use")
cars = glob.glob('../car_training_examples/large_data_set/vehicles/*/*.png')
notcars = glob.glob('../car_training_examples/large_data_set/non-vehicles/*/*.png')


SEED = 42 # Use this during testing & debug to get predictable results.
EXPLORE=False  # Set this to False to used pickled fit classifier. 
OVERRIDE = False  # Set this to False to override use of parms from exploring & to use a hardcoded set.  
# These were the top 3 during exploration with small dataset
# BEST_PARMS=['YUV', 9, 8, 2, 'ALL', (16, 16), 32]
BEST_PARMS=['YCrCb', 9, 8, 2, 'ALL', (16, 16), 32]
# BEST_PARMS=['YCrCb', 9, 16, 2, 'ALL', (16, 16), 32]

# Move this to bboxes file? or helpers file? 
def capture_interim_results(image, stage, out_fn, heat=False):
    #
    dir_out = os.path.dirname(out_fn)
    file_name_out = os.path.basename(out_fn)
    filename, file_extension = os.path.splitext(file_name_out)
    new_out_fn = dir_out + '/' + stage + '_' + filename + file_extension
    print(" \n Capturing interim image: ", new_out_fn)
    if heat:
        new_out_fn = dir_out + '/' + stage + '_' + filename + '.png' 
        mpimg.imsave(new_out_fn, image, cmap='hot')
    else:
        # Convert to BGR & use cv2
        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(new_out_fn, result) 

# To run the car detection pipeline on sample images
def test_carDetectPipeline():
    # get input test fn & output dir
    results_test_output_dir = '../output_images/output_test_images/'
    test_img_fn_list = glob.glob('../test_images/' + '*.jpg')
    print(" \n List of test images: ", test_img_fn_list )     
    #
    #
    #
    for fn in test_img_fn_list:
        #
        # Need to initialize the pipeline every time since there is just one frame
        init_carDetectPipeline()
        # 
        if JKM_DEBUG: print(" \n Starting pipeline on image: ", fn)
        image = mpimg.imread(fn)
        #
        # Call to pipeline goes here        
        # Write the result image
        out_fn =  results_test_output_dir + 'result_' + os.path.basename(fn)
        # 
        # Invoke the pipeline
        result_image = carDetectPipeline(image, out_fn=out_fn, interim=True, debug=True, video=False)
        # for now just copy the image
        # result_image = np.copy(image)
        print(" \n Storing result of pipeline to: ", out_fn)
        #mpimg.imsave(out_fn, result_image)  # TODO : find out why this produces unviewable files?
        # Convert to BGR & use cv2
        result = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_fn, result)

# Initialize these later
# Let's keep track of 6 frames
NUM_FRAMES_TO_TRACK=6
frame_info = {'frame': 0, 'match_data': [], 'conf_data': [] }

# Initialize the pipeline
def init_carDetectPipeline():
    # Initialize the data that we want to track
    init_carDetectData()


def init_carDetectData():
    global frame_info
    frame_info['frame'] = 0
    match, conf = [], [] 
    for i in range(NUM_FRAMES_TO_TRACK):
        match.append([])
        conf.append([])
    frame_info['match_data'] = match
    frame_info['conf_data'] = conf


# Use this to update the pipeline info after every frame
def update_carDetect_data(match, conf):
    global frame_info
    frame = frame_info['frame']
    frame_info['match_data'][frame] = match
    frame_info['conf_data'][frame] = conf    
    # update the ptr for next time
    frame += 1
    if (frame >= NUM_FRAMES_TO_TRACK): frame = 0
    frame_info['frame'] = frame


# use this to get the information from frame info
# NOT USED
def get_carDetect_data():
    # we want all the matches in one list & the confs in another list
    # so convert the list of lists to one list
    return frame_info['match_data'], frame_info['conf_data']    


# TODO: define a global initial & make car_dict a global
# for now put my pipeline in main
def carDetectPipeline(image,  out_fn="", interim=False, debug=False, video=True):
    #
    # 
    # Step 1
    # 
    # Get the stored parms & classifier
    #  TODO: set the data as global & get only once on an initialize?
    if debug: print("\n Car classifier parms being loaded")
    car_dict = get_pickle_data()  # JKM - make this a global 
    svc = car_dict['svc']
    X_scaler = car_dict['scaler']
    colorspace = car_dict['colorspace']
    hog_channel = car_dict['hog_channel']
    spatial_size = car_dict['spatial_size']
    hist_bins = car_dict['hist_bins']
    orient = car_dict['orient']
    pix_per_cell = car_dict['pix_per_cell']
    cell_per_block = car_dict['cell_per_block']
    spatial_feat=True  # should be in the car_dict 
    hist_feat=True     # ditto 
    hog_feat=True      # ditto 
    #
    if debug: print("\n Car classifier parms loaded ")
    #
    #  Step 2: Look for potential cars in the image.
    # 
    # try a loop to vary the search area - JKM - Is this too many? 
    #   TODO: later try to see which of these yields the most hits
    if debug: print("\n Car detection started. ")
    # 
    # Search area is hard coded for now.
    find_cars_search_area = [[ 380, 660, 2.5],  # 0
                             [ 380, 656, 2.0],  # 1
                             [ 380, 656, 1.5],  # 2
                             [ 380, 550, 1.75], # 3
                             [ 380, 530, 1.25]] # 4
    #
    # 
    out_images_list = []
    match_list, conf_list, search_list = [], [], [] 
    for ystart, ystop, scale in find_cars_search_area:
        if JKM_DEBUG: print(" \n Search for cars: ", ystart, ystop, scale)
        out_img, match_l, conf_l, search_l = find_cars(image, ystart, ystop, scale, 
                                                       svc, X_scaler, orient, pix_per_cell, 
                                                       cell_per_block, spatial_size, hist_bins, 
                                                       colorspace)
        out_images_list.append(out_img)
        match_list = match_list + match_l
        conf_list = conf_list + conf_l
        search_list = search_list + search_l
    #
    #
    all_matches_image = draw_boxes(image, match_list)
    if debug: print(" \n Potential car matches found: ", len(match_list))
    if interim:
        stage = 'STEP_2_all_matches'
        capture_interim_results(all_matches_image, stage, out_fn)
    # 
    #
    #
    #  Step 3: Threshold on the confidence value for each of the possible matches
    CONF_THRESH = 0.1 # 0.3 Hardcoded for now, should find a better way to do this
    match_list, conf_list = apply_threshold_of_confidence(match_list, conf_list, CONF_THRESH)
    all_matches_image = draw_boxes(image, match_list)
    if debug: print(" \n Potential car matches found after applying threshold of confidence factor: ", len(match_list))
    if interim:
        stage = 'STEP_3_all_conf_matches'
        capture_interim_results(all_matches_image, stage, out_fn)
    #
    #  Step 4 Store Results in Rolling Frame Structure 
    #
    update_carDetect_data(match_list, conf_list)
    #
    #
    #  Step 5: Remove duplicates and false positives
    # 
    # Heat is for one single picture
    if debug: print("\n Removing Duplicates & false positives. ")    
    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    #
    # Add heat to each box in our matches
    heat = add_heat(heat, match_list)
    #
    # Apply threshold to help remove false positives
    FP_THRESH =  2 # 1
    heat = apply_threshold(heat, FP_THRESH)
    #
    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)
    #
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    num_cars=len(labels)
    removed_dups_image = draw_labeled_bboxes(np.copy(image), labels)
    if debug: print(" \n Cars found (after removing dups & fp's) : ", num_cars)
    if interim:
        stage = 'STEP_5_removed_dups_fps'
        capture_interim_results(removed_dups_image, stage, out_fn)
        #
        stage= 'STEP_5_heatmap'
        #
        capture_interim_results(heatmap, stage, out_fn, heat=True)
    #
    #
    # Step 6: Calculate Heat_sum is for several frames
    #
    #
    if debug: print("\n Removing Duplicates & false positives, over set of frames. ")    
    heat_sum = np.zeros_like(image[:,:,0]).astype(np.float)
    #
    # Add heat to each box in our matches
    for m_l in frame_info['match_data']:
        if m_l != []: heat_sum = add_heat(heat_sum, m_l)
    #
    # Apply threshold to help remove false positives
    if video: 
        FP_FRAME_THRESH =  FP_THRESH * NUM_FRAMES_TO_TRACK 
    else: 
        FP_FRAME_THRESH = FP_THRESH
    heat_sum = apply_threshold(heat_sum, FP_FRAME_THRESH )
    # Visualize the heatmap when displaying    
    heatmap_sum = np.clip(heat_sum, 0, 255)
    #
    # Find final boxes from heatmap using label function
    labels_sum = label(heatmap_sum)
    num_cars_sum=len(labels_sum)
    removed_dups_image_sum = draw_labeled_bboxes(np.copy(image), labels_sum)
    if debug: print(" \n Cars found (after removing dups & fp's) over set of frames: ", num_cars_sum)
    if interim:
        stage = 'STEP_6_sum_removed_dups_fps'
        capture_interim_results(removed_dups_image_sum, stage, out_fn)
        #
        stage= 'STEP_6_sum_heatmap'
        capture_interim_results(heatmap_sum, stage, out_fn, heat=True)
    #
    #
    #  Add additional steps here
    #
    print(" \n Done ")
    #
    return removed_dups_image_sum






# MAIN
# ====
def main(argv=None):  # pylint: disable=unused-argument

    # JKM For now set seed
    print("\n NOTE: Setting seed to get predictable results during development phase")
    np.random.seed(SEED)


    # Run everything inside of main


    ### 
    ### This section for exploring & training a classifier. 
    ##    Fitted classifier & parms used for that get pickled.
    ###                                                                      
    if EXPLORE:
        print("Starting - Step 1 - Explore & Find Best Feature Extraction Parms")
        
        # Want to override so as to reduce parms later          
        if OVERRIDE == True:
            best_parms = BEST_PARMS
            print(" \n Overridden - using these parms: ", best_parms)
        else:
            # This explores best parms
            best_parms = explore_feature_extraction_parms(cars, notcars, seed=SEED)
            print("\n Best parms found: ", best_parms)

        car_dict = run_classifier_and_pickle(cars, notcars, best_parms, seed=SEED)
        # Run the classifier again with the selected parms & pickle the results
    else:
        print("Starting - Step 1 - Get Pickled Data")
        car_dict = get_pickle_data()           

    # Get the stored parms & classifier
    svc = car_dict['svc']
    X_scaler = car_dict['scaler']
    colorspace = car_dict['colorspace']
    hog_channel = car_dict['hog_channel']
    spatial_size = car_dict['spatial_size']
    hist_bins = car_dict['hist_bins']
    orient = car_dict['orient']
    pix_per_cell = car_dict['pix_per_cell']
    cell_per_block = car_dict['cell_per_block']

    spatial_feat=True 
    hist_feat=True 
    hog_feat=True

    print("\n Step_1__Program paused. Press Ctrl-D to continue.\n")
    code.interact(local=dict(globals(), **locals()))
    print(" ... continuing\n ")
    
 
    ### 
    ### This section for finding cars in an image using sliding window.       
    ###                                                                     

    print("\n Step 2 : Explore create and search windows for cars using sliding window algorithm. \n ")
    
    # For now use a sample image
    # IMAGE 1
    # -------
    image = mpimg.imread('../examples/bbox-example-image.jpg')
    draw_image = np.copy(image)
    
    # jkm - move this up to other imports
    from sliding_window import get_windows, search_windows, slide_window
    # w_list = get_windows(image)
    # flat_list_of_windows = [item for sublist in w_list for item in sublist] - already flattened

    # jkm - just work with one set of windows at this time
    #Y_START_P = 0.6  # Eliminate 60% of the search space in the y direction
    #y_start = int( Y_START_P * image.shape[0])
    #y_start_stop = [y_start, None]
    #w96_list = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop, 
    #                xy_window=(96, 96), xy_overlap=(0.5, 0.5))
    #w192_list = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop, 
    #                xy_window=(192, 192), xy_overlap=(0.5, 0.5))

    # Debugging code - Large & medium windows - HARD CODE 
    w_large = slide_window(image, x_start_stop=[None, None], y_start_stop=[500, None], 
                    xy_window=(250, 200), xy_overlap=(0.5, 0.5))

    w_med = slide_window(image, x_start_stop=[100, 1200], y_start_stop=[490, 600], 
                    xy_window=(110, 80), xy_overlap=(0.5, 0.5))

    # combine lists
    w_all = w_large + w_med

    # Find the  set of windows that match
    hot_windows = search_windows(image, w_all, svc, X_scaler, color_space=colorspace, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)  
    print("\n Windows Found: ", len(hot_windows))
    # temp - for testing - draw the hot windows
    window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)
    plt.imshow(window_img)

    
    # IMAGE 2
    # --------
    # try with another image. the one they use for hog sub-sampling
    #   Need a whooe new set of windows to search 
    image2 = mpimg.imread('../test_images/test4.jpg')
    plt.imshow(image2) # look at it first
    # 
    draw_image2 = np.copy(image2)
    # Get new set of windows
    w_large2 = slide_window(image2, x_start_stop=[None, None], y_start_stop=[350, None], 
                    xy_window=(250, 200), xy_overlap=(0.5, 0.5))

    w_med2 = slide_window(image2, x_start_stop=[200, None], y_start_stop=[350, 550], 
                    xy_window=(110, 80), xy_overlap=(0.5, 0.5))
    w_all2 = w_large2 + w_med2
    #
    image2_bb = draw_boxes(draw_image2, w_all2, color=(0, 0, 255), thick=6)
    plt.imshow(image2_bb)
    # Find the  set of windows that match
    #   - this doesn't find any
    hot_windows2 = search_windows(image2, w_all2, svc, X_scaler, color_space=colorspace, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)  
    print("\n Windows Found: ", len(hot_windows2))

    # temp - for testing - draw the hot windows
    window_img2 = draw_boxes(draw_image2, hot_windows2, color=(0, 0, 255), thick=6)
    plt.imshow(window_img2)



    print("\n Step_2__Program paused. Press Ctrl-D to continue.\n")
    code.interact(local=dict(globals(), **locals()))
    print(" ... continuing\n ")

    ### 
    ### This section for finding cars in an image using hog subsampling.       
    ###                                                                

    print("\n Step 3 : Exploring Use of hog_subsample to locate cars.")
    
        
    # using their image for hog-subsampling
    image2 = mpimg.imread('../test_images/test4.jpg')

    ystart = 400
    ystop = 656
    scale = 1.5

    from hog_subsample import find_cars

    # use the same image as in the lesson
    # from feature_explore import convert_color, bin_spatial, color_hist, get_hog_features
    out_img2, match_l, conf_l, search_l = find_cars(image2, ystart, ystop, scale, svc, X_scaler, 
                        orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, colorspace)
    plt.imshow(out_img2)

    # Draw the same thing again but with the box list
    match_img2 = draw_boxes(image2, match_l)
    plt.imshow(match_img2)

    # Draw all the  search areas
    search_img2 = draw_boxes(image2, search_l)
    plt.imshow(search_img2)

    # Need different start stops for this one - these ones don't work that  well
    out_img1, match_l1, conf_l1, search_l1 = find_cars(image, ystart, ystop, scale, svc, X_scaler, 
                       orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, colorspace)
    plt.imshow(out_img1)

    match_img1 = draw_boxes(image, match_l1)
    plt.imshow(match_img1)

    # Draw all the  search areas
    search_img1 = draw_boxes(image, search_l1)
    plt.imshow(search_img1)

    # try a loop to vary the search area - JKM - Is this too many? 
    #   TODO: later try to see which of these yields the most hits
    find_cars_search_area = [[ 380, 660, 2.5],  # 0
                             [ 380, 656, 2.0],  # 1
                             [ 380, 656, 1.5],  # 2
                             [ 380, 550, 1.75], # 3
                             [ 380, 530, 1.25]] # 4

    # Image2
    out_images2_list = []
    match_list, conf_list, search_list = [], [], []
    for ystart, ystop, scale in find_cars_search_area:
        if JKM_DEBUG: print(" \n Search for cars: ", ystart, ystop, scale)
        out2, match_l, conf_l, search_l = find_cars(image2, ystart, ystop, scale, 
                                                       svc, X_scaler, orient, pix_per_cell, 
                                                       cell_per_block, spatial_size, hist_bins, 
                                                       colorspace)
        out_images2_list.append(out2)
        match_list = match_list + match_l
        conf_list = conf_list + conf_l
        search_list = search_list + search_l

    # Draw the same thing again but with the box list
    match_list_img2 = draw_boxes(image2, match_list)
    plt.imshow(match_list_img2)

    # Draw all the  search areas
    search_list_img2 = draw_boxes(image2, search_list)
    plt.imshow(search_list_img2)

    # use this to print out the images
    #plt.imshow(out_images2[0])


    # Image1 aka Image
    # Image
    out_images1_list = []
    match_list_1, conf_list_1, search_list_1 = [], [], []
    out_images1 = []
    for ystart, ystop, scale in find_cars_search_area:
        print(" \n Search for cars: ", ystart, ystop, scale)
        out1, match_l, conf_l, search_l = find_cars(image, ystart, ystop, scale, svc, X_scaler, 
                            orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, colorspace)
        out_images1.append(out1)
        #
        match_list_1 = match_list_1 + match_l
        conf_list_1 = conf_list_1 + conf_l
        search_list_1 = search_list_1 + search_l
    #plt.imshow(out_images1[0])

    # Draw the same thing again but with the box list
    match_list_img1 = draw_boxes(image, match_list_1)
    plt.imshow(match_list_img1)

    # Draw all the  search areas
    search_list_img1 = draw_boxes(image, search_list_1)
    plt.imshow(search_list_img1)

    print("\n Step_3__Program paused. Press Ctrl-D to continue.\n")
    code.interact(local=dict(globals(), **locals()))
    print(" ... continuing\n ")

    ### 
    ### This section for exploring heat maps.       
    ###         

    print("\n Step 4 : Exploring Use of heat maps to remove duplicates & false positives.")

    # Working with the same image - image2 from before
    # 
    box_list = match_list
    heat = np.zeros_like(image2[:,:,0]).astype(np.float)

    # Add heat to each box in box list
    heat = add_heat(heat,box_list)

    # Apply threshold to help remove false positives
    FP_THRESH =  2 # 1
    heat = apply_threshold(heat,FP_THRESH)

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image2), labels)

    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(draw_img)
    plt.title('Car Positions')
    plt.subplot(122)
    plt.imshow(heatmap, cmap='hot')
    plt.title('Heat Map')
    fig.tight_layout()


    #  2nd image aka bbox image
    # Try the other image aka image
    #  NOTE - issue here is that these are clumped together
    box_list = match_list_1
    heat = np.zeros_like(image[:,:,0]).astype(np.float)

    # Add heat to each box in box list
    heat = add_heat(heat,box_list)

    # Apply threshold to help remove false positives
    FP_THRESH = 2 # 1
    heat = apply_threshold(heat,FP_THRESH)

    # Visualize the heatmap when displaying    
    heatmap= np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    # print(labels[1], 'cars found')
    draw_img = draw_labeled_bboxes(np.copy(image), labels)

    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(draw_img)
    plt.title('Car Positions')
    plt.subplot(122)
    plt.imshow(heatmap, cmap='hot')
    plt.title('Heat Map')
    fig.tight_layout()

    print("\n Step_4__Program paused. Press Ctrl-D to continue.\n")
    code.interact(local=dict(globals(), **locals()))
    print(" ... continuing\n ")

    ### 
    ### This section for exploring pipelines.       
    ###         

    print("\n Step 5 : Exploring Pipeline")

    # To test single invokation of pipeline
    fn = '../test_images/test4.jpg'
    results_test_output_dir = '../output_images/output_test_images/'
    out_fn =  results_test_output_dir + 'result_' + os.path.basename(fn)
    init_carDetectPipeline()
    result_image = carDetectPipeline(image, out_fn=out_fn, interim=True, debug=True, video=False)
    plt.imshow(result_image)


    print("\n Step_5__Program paused. Press Ctrl-D to continue.\n")
    code.interact(local=dict(globals(), **locals()))
    print(" ... continuing\n ")


    print("\n Step 6 : Exploring Video")

    # To test pipeline on video

    # uncomment this to test with a short video
    # output_path = '../output_videos/test_video.mp4'
    # input_path = '../test_video.mp4'

    # Use this to do the assignment video
    output_path = '../output_videos/project_video.mp4'
    input_path = '../project_video.mp4'

    # Start here
    init_carDetectPipeline()
    input_clip = VideoFileClip(input_path) # 
    #input_clip = VideoFileClip(input_path).subclip(40,43) # problematic part
    output_clip = input_clip.fl_image(carDetectPipeline)
    output_clip.write_videofile(output_path, audio=False)    


    print("\n Step_6__Program paused. Press Ctrl-D to continue.\n")
    code.interact(local=dict(globals(), **locals()))
    print(" ... continuing\n ")


    print("\n Step 7 : Exploring Combined Video")
   
    # This is to try to merge the two results
    #  Note: This doesn't work as well as I would like
    from advancedLaneLines import pipeline 
    def combinedPipeline(image):
        image = pipeline(image)             # advanced Lane Finding pipeline
        image = carDetectPipeline(image)    # car detection Pipeline
        return image

    # uncomment this to test with a short video
    #output_path = '../output_videos/test_video.mp4'
    #input_path = '../test_video.mp4'

    # Use this to do the assignment video
    output_path = '../output_videos/combined_project_video.mp4'
    input_path = '../project_video.mp4'

    # Start here
    init_carDetectPipeline()
    input_clip = VideoFileClip(input_path) # 
    #input_clip = VideoFileClip(input_path).subclip(40,43) # problematic part
    output_clip = input_clip.fl_image(combinedPipeline)
    output_clip.write_videofile(output_path, audio=False)    



    print("\n Step_7__Program paused. Press Ctrl-D to continue.\n")
    code.interact(local=dict(globals(), **locals()))
    print(" ... continuing\n ")


   
if __name__ == '__main__':
    #tf.app.run()
    main()


# This is used to test the pipeline with the test images as we build it.
os.path.basename = basename(p)
image = mpimg.imread('../test_images/test4.jpg')
os.path.basename('../test_images/test4.jpg')
filename, file_extension = os.path.splitext('/path/to/somefile.ext')
fn = '../test_images/test4.jpg'


