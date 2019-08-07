# Advanced Lane Finding Thresholding Procedures 
#    There are numerous thresholding procedures in this script.
#    All of these were used at one time during the prototyping phase to explore
#    which thresholding functions worked best.
# 
import numpy as np
import cv2


# Added by my code
import code
import sys
import os
# Set Debug flag
JKM_DEBUG=True

# Local
import advancedLaneLine_helpers

# Threshold Procedures
# =====================

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # A function that applies Sobel x or y, then takes an absolute value & 
    #  applies a threshold. 
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # read with mpimg
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray = cv2.cvtColor(img, cv2color)
    # 
    #
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    #parm1, parm2 = 1, 0 # assume 'x'
    if orient=='x':
        parm1, parm2 = 1, 0
    elif orient=='y':
        parm1, parm2 = 0, 1
    else:
        print ("ERROR on orient. should be x or y")
        return None
    sobel = cv2.Sobel(gray, cv2.CV_64F, parm1, parm2)
    #  
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    #
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    #
    # 5) Create a mask of 1's where the scaled gradient magnitude
    #         is > thresh_min and < thresh_max
    thresh_min, thresh_max = thresh
    if JKM_DEBUG: 
        print ("abs_sobel_thresh: orient, thresh min max", (orient, thresh_min, thresh_max))
    if thresh_min > thresh_max:
        print("ERROR thresh min needs to be less than thresh max", thresh)
        return img 
    grad_binary= np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    #
    # 6) Return this mask as your binary_output image
    return grad_binary

# Not used
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Calculate gradient magnitude
    # Apply threshold
    #
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # 3) Calculate the magnitude
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    abs_sobelxy = np.sqrt(np.square(sobelx) + np.square(sobely))
    
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobelxy/np.max(abs_sobelxy))
    
    # 5) Create a binary mask where mag thresholds are met
    mag_binary = np.zeros_like(scaled_sobel)
    thresh_min, thresh_max = mag_thresh
    mag_binary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    
    # 6) Return this mask as your binary_output image
    return mag_binary

# Not used
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    # Apply threshold
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # 3) Calculate the magnitude
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    abs_sobelxy = np.sqrt(np.square(sobelx) + np.square(sobely))
    
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    dirgrad = np.arctan2(abs_sobely, abs_sobelx)

    # 5) Create a binary mask where direction thresholds are met
    dir_binary = np.zeros_like(dirgrad)
    thresh_min, thresh_max = thresh
    dir_binary[(dirgrad >= thresh_min) & (dirgrad <= thresh_max)] = 1
    
    # 6) Return this mask as your binary_output image
    return dir_binary

# Define the channels
HLS_H, HLS_L, HLS_S = 0, 1, 2
HSV_H, HSV_S, HSV_V = 0, 1, 2
RGB_R, RGB_G, RGB_B = 0, 1 ,2
ALL=(0, 255)
def colour_select(img):
    # This is where I chose the colours to select via
    #  various thresholding of RGB, HLS, HSV colour maps
    #  in order to find yellows and whites that could be lane lines.
    #
    RGB_white = cv2.inRange(img, (100, 100, 200), (255, 255, 255))
    RGB_yellow = cv2.inRange(img, (225, 180, 0), (255, 255, 170))
    #
    HSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    HSV_yellow = cv2.inRange(HSV, (20, 100, 100), (50, 255, 255))
    HSV_white = cv2.inRange(HSV, (0,0,187), (255,20,255))
    # 
    HLS = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    HLS_white = cv2.inRange(HLS, (0,195,0), (255,255,60))
    HLS_yellow = cv2.inRange(HLS, (20, 120, 80), (45, 200, 255))
    #
    combined =  RGB_white | RGB_yellow | HLS_white | HLS_yellow | HSV_white | HSV_yellow
    return combined

def colour_select_orig(img):
    # This is where I chose the colours to select via
    #  various thresholding of RGB, HLS, HSV colour maps
    #  in order to find yellows and whites that could be lane lines.
    #
    # RGB: R (200, 255), G (200, 255), B (200 255)
    # HSV: All H, S (0, 20) & (100, 255), V (100, 255)
    # HLS: All H, L (195, 255), S (0, 60)
    R = rgb_select(img, thresh=(200, 255), channel=RGB_R)
    G = rgb_select(img, thresh=(200, 255), channel=RGB_G)
    B = rgb_select(img, thresh=(200, 255), channel=RGB_B)
    #RGB_white_yellow = R | G | B
    #
    # yellow: HSV H (20, 50), S(100, 255) , V (200, 255)
    #H = hsv_select(img, thresh=(20,50), channel=HSV_H)
    #S = hsv_select(img, thresh=(100, 255), channel=HSV_S)
    # V = hsv_select(img, thresh=(220, 255), channel=HSV_V)
    # issues wit this above
    HSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    HSV_yellow = cv2.inRange(HSV, (20, 100, 100), (50, 255, 255))
    HSV_white = cv2.inRange(HSV, (0,0,187), (255,20,255))
    # 
    #HSV_2 = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    #HSV_yellow2 = cv2.inRange(HSV_2, (20, 100, 100), (50, 255, 255))
    #
    # white: HLS:  H (0, 255) L (195, 255) S (0, 60)
    # This doesn't give the same things as HLS
    #HHH = hls_select(img, thresh=ALL, channel=HLS_H)
    #LLL = hls_select(img, thresh=(195, 255), channel=HLS_L)
    #SSS = hls_select(img, thresh=(0, 60), channel=HLS_S)
    # 
    HLS = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    HLS_white = cv2.inRange(HLS, (0,195,0), (255,255,60))
    #
    #combined = np.zeros_like(R)
    #combined[((R>0) | (G>0) | (B>0) | (HLS_white>0) | (HSV_white>0) | (HSV_yellow>0)) ] = 255
    #
    combined =  R | G | B | HLS_white | HSV_white | HSV_yellow
    return combined   

def rgb_select(img, thresh=(0, 255), channel=0):
    # Apply a threshold an rgb channe
    #  Default is R channel
    # 1) Already in RGB
    #
    # 2) Apply a threshold to the chosen channel
    C = img[:,:,channel]
    binary_output = np.zeros_like(C)
    binary_output[(C > thresh[0]) & (C <= thresh[1])] = 1    
    # 3) Return a binary image of threshold result
    return binary_output

# 
def hls_select(img, thresh=(0, 255), channel=2):
    # Apply a threshold an hls channel
    #  Default is S channel
    # 1) Convert to HLS color space
    print("hls")
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    #
    # 2) Apply a threshold to the chosen channel
    C = hls[:,:,channel]
    binary_output = np.zeros_like(C)
    binary_output[(C > thresh[0]) & (C <= thresh[1])] = 1    
    # 3) Return a binary image of threshold result
    return binary_output

# 
def hsv_select(img, thresh=(0, 255), channel=2):
    # Apply a threshold an hls channel
    #  Default is V channel
    # 1) Convert to HSV color space
    print("hsv")
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)    
    # 2) Apply a threshold to the chosen channel
    C = hsv[:,:,channel]
    binary_output = np.zeros_like(C)
    binary_output[(C > thresh[0]) & (C <= thresh[1])] = 1    
    # 3) Return a binary image of threshold result
    return binary_output


# Not used - jkm - mine 
def colour_threshold(img, hls_thresh=(0, 255), hsv_thresh=(0, 255)):    
    colour_hls = hls_select(img, thresh=hls_thresh)
    colour_hsv = hsv_select(img, thresh = hsv_thresh)
    output= np.zeros_like(colour_hls)
    output[(colour_hls == 1) | (colour_hsv == 1)] = 1
    return output

# From youtube video
def color_threshold(image, sthresh=(0,255), vthresh=(0,255)):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    # hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    s_channel = hls[:, :, 2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= sthresh[0])  & (s_channel <= sthresh[1])] = 1
    #	
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v_channel=hsv[:, :, 2]
    v_binary = np.zeros_like(v_channel)
    v_binary[ (v_channel >= vthresh[0]) & (v_channel <= vthresh[1]) ] = 1
    # 	
    output=np.zeros_like(s_channel)
    output[ (s_binary == 1) & (v_binary == 1) ] = 1
    return output


# testImageFn = 'signs_vehicles_xygrad.png' # when running from my project4 directory
testImageDir =  'test_images/'
testImageFn = 'test5.jpg'   # changed from test1.jpg


# Not used anymore
def test_thresholding_functions(image):
    ''' Testing the helper functions '''

    # testing the helpers here
    # Choose a Sobel kernel size
    ksize = 3 # Choose a larger odd number to smooth gradient measurements

    # Apply each of the thresholding functions
    print("\n Testing - find Gradient in x direction")
    # gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(0, 255))
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    jkm_plot_results(image, 'Original Image', gradx,  'Thresholded Gradient x')

    print("\n Step_1__Program paused. Press Ctrl-D to continue.\n")
    code.interact(local=dict(globals(), **locals()))
    print(" ... continuing\n ")

    print("\n Testing - find Gradient in y direction")
    # grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(0, 255))
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(20, 100))
    jkm_plot_results(image, 'Original Image', grady,  'Thresholded Gradient y')

    print("\n Step_1__Program paused. Press Ctrl-D to continue.\n")
    code.interact(local=dict(globals(), **locals()))
    print(" ... continuing\n ")

    print("\n Testing - find magnitude of the Gradient in x & y direction")
    # mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(0, 255))
    mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(30, 100))
    jkm_plot_results(image, 'Original Image', mag_binary,  'Thresholded Magnitude')
    
    print("\n Step_1__Program paused. Press Ctrl-D to continue.\n")
    code.interact(local=dict(globals(), **locals()))
    print(" ... continuing\n ")

    print("\n Testing - find direction of the gradient")
    #dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0, np.pi/2)) 
    dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0.7, 1.3))  
    jkm_plot_results(image, 'Original Image', dir_binary,  'Thresholded Direction')
    
    print("\n Step_1__Program paused. Press Ctrl-D to continue.\n")
    code.interact(local=dict(globals(), **locals()))
    print(" ... continuing\n ")

    # For example, here is a selection for pixels where both the x and y gradients 
    # meet the threshold criteria, or the gradient magnitude and direction are both
    #  within their threshold values.

    print("\n Testing - Looking at Combined")

    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    jkm_plot_results(image, 'Original Image', combined,  'Combined')



