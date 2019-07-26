# Advanced Lane Finding - Camera Calibration Script
#   This is run as a stand-alone script to calibrate the camera based on the chessboard
#   images. This only needs to be run once and it saves the camera calibration information
#   to a pickle file so that it can be used later by other programs.
#   
#   A camera calibration test function is also providedd and is run as part of the script
#    so that the user may see the results of the calibration on test chechboard images.
# 

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import code


# Added by my code
import glob   # to read in global names 
# Set Debug flag
JKM_DEBUG=True

# Local
import helpers



## Camera Calibration
## ===================
# we could have different cameras and different chessboard images
#  for now, we just assume only one

class Camera():
    def __init__(self,cameraCalDir='../camera_cal/', cameraCalGlob='calibration*.jpg',nx=9,ny=6):
         self.name="Camera"
    
         # Chessboard images information (note - make chessboard class?)
         self.nx = nx # number of inside corners in x
         self.ny = ny # number of inside corners in y

         # Information about where the camera calibration chessboard images
         self.dir = cameraCalDir
         self.globFn = cameraCalGlob

         # Arrays to store object points and image points from all the images
         self.objpoints=[] # 3D points in real world space
         self.imgpoints=[] # 2D points in image plane

         # Prepare object points (x,y,z) like (0,0,0),(1,0,0), ... (8,5,0)4036
         #  to represent all the chestboard corners.
         #  There are nx * ny points, each with 3 coordinates (x,y,z) to store
         #  for the x,y,z. x & y we will fill in, but Z will be 0 for all.
         self.objp = np.zeros((ny*nx,3), np.float32)
         self.objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2) # x,y coordinates 

         # The camera calibration information
         # self.ret, self.rvecs, self.tvecs = None, None, None
         self.mtx, self.dist = [], []

         # Keep an image for testing purposes
         self.test_image = []

         # Calibrate the camera
         self.calibrate()

    # method to calibrate the camera
    def calibrate(self):
        # Re-initialize our import pickleobjpoints and imagepoints  
        self.objpoints=[] # 3D points in real world space
        self.imgpoints=[] # 2D points in image plane

        # Read in filenames and make a list of calibration images
        images = glob.glob(self.dir + self.globFn)
        print("\n Calibratng Camera with chess board images")

        # For all the calibration images in our list4036
        for fn in images: 
            # Read in an image - using matplotlib image read so get a RGB image
            img = mpimg.imread(fn)
 
            # keep the lastimage for testing
            self.test_image = img
    
            # 1) Convert to grayscale. original image is RGB
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            #print("shape :", gray.shape)

            # 2) Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (self.nx, self.ny), None)
    
            # 3) If corners are found, add object points & image points to the lists
            if ret == True:
                # print("\n Chessboard corners found for image: ", fn)
                self.imgpoints.append(corners) # 2D
                self.objpoints.append(self.objp) # 3D , same for all chestboards

                # Draw and display the corners (we will only see the last one) 
                cv2.drawChessboardCorners(img, (self.nx, self.ny), corners, ret)
                plt.ion()
                plt.imshow(img)

        # Calibrate the  camera using the size of the last image     
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, 
                                             self.imgpoints, 
                                             gray.shape[::-1],        
                                             None, None)

        self.mtx = mtx
        self.dist = dist

        # Save the camera calibration result for later 
        dist_pickle = {}
        dist_pickle["mtx"] = mtx
        dist_pickle["dist"]= dist
        pickle.dump(dist_pickle, open (self.dir + "camera_calibration.p", "wb"))

        print("\n Camera is calibrated. mtx & dist have been pickled") 

    # method to undistort images
    def undistort(self, image):
        # Make sure that the     
        if (self.mtx == [] or self.dist== []):
            print("ERROR: Camera not calibrated")
            return image
        # Undistort
        return cv2.undistort(image, self.mtx, self.dist, None, self.mtx)

    # test the undistortion on a test image
    def test_undistort(self):
        if self.test_image == []: 
            print("ERROR: No test image available")
            return  
        undist = self.undistort(self.test_image)
        # Display the original & undistorted image
        helpers.jkm_plot_results(self.test_image, "Original Image" , undist, "Undistorted Image") 


# Calibrate the camera
# Create the camera object. This also shows the chessboard corners in a test image
camera = Camera()

# Test the calibration on a test image
camera.test_undistort()

print("\n Program paused. Press Ctrl-D to continue.\n")
code.interact(local=dict(globals(), **locals()))
print(" ... continuing\n ")

print("Camera Calibration is complete.")        


