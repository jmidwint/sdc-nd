# Advanced Lane Finding - This is the main script for the advanced lane finding project.
#   This script should be run after the camera calibration script "camera.py" has been run.
#   
#   This script will first run a pipeline on a series of test images or on a video stream, then
#    write the results back to disk.
# 
#   This pipeline does the following:
#      1) Undistorts the image.
#      2) Transforms or warps the image perspective to a "top down view".
#      3) Performs colour thresholding to pick out white & yellow based sections of the "top down" view.
#      4) Uses a histogram & a sliding window algorithms to detect the two lane lanes, left & right, in 
#          which the car taking the images is traveling.
#      5) Determines the radius of curvature of the lane line & the relative distance of the car to the
#          the centre of the lane line (& if right or left of that centre line).
#      6) Finally, visually updates the image,  in the regular head-on perspective, with a pattern of solid 
#          green that identifies the lane to the user, as well as displaying the instantaneous value of
#          the radius of curvature & the location of the car w.r.t the centre of the lane.  
#          
#   

import numpy as np
import cv2

# Added by my code
import code
import sys
import os
import glob   # to read in global names
import pickle 

# Locals
from advancedLaneLine_helpers import *
#from camera import Camera
from advancedLaneLine_threshold import *
from advancedLaneLine_lane import * 
import matplotlib.image as mpimg
from matplotlib import cm


# Set Debug flag
JKM_DEBUG=True


# Get the camera calibration information
dist_pickle = pickle.load(open("../../CarND-Advanced-Lane-Lines/camera_cal/camera_calibration.p", "rb") )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]


#  




# Edit this function to create your own pipeline.
def pipeline(img):
    #
    # Step 1) Undistort the image 
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    #
    # Step 2) Find Src & Dst points, & create M & Minv 
    #  JKM - perhaps this can be moved to outside of the pipeline since this is a constant
    #  Note: These "hard-coded" src/dst points were manually obtained/tested to one of the 
    #         test images provided with the project ... straight_lines1.jpg 
    img_size = (img.shape[1], img.shape[0])
    src = np.float32(
             [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
             [((img_size[0] / 6) - 10), img_size[1]],
             [(img_size[0] * 5 / 6) + 60, img_size[1]],
             [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
    dst = np.float32(
             [[(img_size[0] / 4), 0],
             [(img_size[0] / 4), img_size[1]],
             [(img_size[0] * 3 / 4), img_size[1]],
             [(img_size[0] * 3 / 4), 0]])
    M= cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    #
    # Step 3 - Perform the perpective transform to top down view
    warped = cv2.warpPerspective(undist, M, img_size, flags=cv2.INTER_LINEAR)
    #
    # Step 4 - Perform Colour thresholding to find all yellow & white lines 
    colour_thresh=colour_select(warped)
    #
    # Step 5 - Find Lane Lines 
    # old - ploty, left_fitx, right_fitx = find_lane_lines_new(colour_thresh, visualize=True) 
    ploty, left_fitx, right_fitx, left_fit, right_fit, car_pix = find_lane_lines_new(colour_thresh)
    #
    # Step 5b - want to test the continue with lane line finding 
    ploty, left_fitx, right_fitx, left_fit, right_fit = find_lane_lines_continue(colour_thresh,
                      ploty, left_fitx, right_fitx, left_fit, right_fit) 
    #
    # Step 6 - Draw/fill onto our undist image
    final = get_drawn_lanes(colour_thresh, ploty, left_fitx, right_fitx, Minv, undist, img_size)
    #
    # Step 7 
    final_with_info = add_curve_and_car_info(final, ploty, left_fitx, right_fitx, car_pix)
    #
    return final_with_info

# make a list of test images
test_images = glob.glob("../test_images/test*.jpg")

extension='.jpg' # 
# extension='.png'   # use this to capture binary images

# MAIN
# ====
def main(argv=None):  # pylint: disable=unused-argument


    # Run everything inside of main
    print("Starting - Step 1 - Testing with Test Images")

    # testing the pipeline as we build it
    for idx, fname in enumerate(test_images):
        # 
        print("Reading Image: ", fname)
        img = mpimg.imread(fname)  # RGB
        #
        # Run the pipeline on colour RGB images
        srcRGB = pipeline(img)
        #
        # Convert to BGR       
        desBGR = cv2.cvtColor(srcRGB, cv2.COLOR_RGB2BGR)
        result= desBGR
        #result=srcRGB
        write_name = '../test_images/tracked_' + str(idx) + extension
        cv2.imwrite(write_name, result)
        #mpimg.imsave(write_name, result, cmap='gray') # use this to capture binary images
        print("Saved image: " , write_name)


    print("\n Step_1__Program paused. Press Ctrl-D to continue.\n")
    code.interact(local=dict(globals(), **locals()))
    print(" ... continuing\n ")

    print("/n Step 2 Processing Video")
    # 
    from moviepy.editor import VideoFileClip 
    output_path = '../output_videos/project_video.mp4'
    input_path = '../project_video.mp4'
    input_clip = VideoFileClip(input_path) # The full thing
    #input_clip = VideoFileClip(input_path).subclip(40,43) # problematic part
    output_clip = input_clip.fl_image(pipeline)
    output_clip.write_videofile(output_path, audio=False)


    print("\n Step_2__Program paused. Press Ctrl-D to continue.\n")
    code.interact(local=dict(globals(), **locals()))
    print(" ... continuing\n ")

# To run this as a script. 
if __name__ == '__main__':
    #tf.app.run()
    main()







