# Vehicle Detection - Helper Procedures
#   This script is called by other scripts to provide helper functions
#   used to display images to help with debug & verification, or to write 
#   text onto images.
#   

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def jkm_plot_results(orig_image, orig_title, trans_image, trans_title):
    # One routine to display the original image and the transformed image.
    #  code taken from quizzes
    # Plot the result
    plt.ion()
    # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 12))
    f.tight_layout()
    ax1.imshow(orig_image)
    ax1.set_title(orig_title, fontsize=25) #50
    ax2.imshow(trans_image, cmap='gray')
    #ax2.imshow(trans_image)
    ax2.set_title(trans_title, fontsize=25) #50
    plt.subplots_adjust(left=0.125, right=0.9, top=0.9, bottom=0.1)

    # plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


def print_my_image(image, title, orig_img_name, gray=True):
    '''
    To print the image
    '''
    if gray: 
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)
    plt.title(title + "_" + orig_img_name)
    plt.show() 


def print_text_on_image(image, text, position):
    # Function to print text on the image
    #  Prints in white.
    font=cv2.FONT_HERSHEY_DUPLEX
    colour=[255, 255, 255]
    image = cv2.putText(image, text, position, cv2.FONT_HERSHEY_DUPLEX, 2, color=colour, thickness=2)
    return image


# This is taken from the Udacity SDC Classroom Project Vehicle Detection and Tracking"
#   Lesson 15: Explore Color Spaces
def plot3d(pixels, colors_rgb,
        axis_labels=list("RGB"), axis_limits=[(0, 255), (0, 255), (0, 255)]):
    """Plot pixels in 3D."""
    #
    # Create figure and 3D axes
    fig = plt.figure(figsize=(8, 8))
    ax = Axes3D(fig)
    #
    # Set axis limits
    ax.set_xlim(*axis_limits[0])
    ax.set_ylim(*axis_limits[1])
    ax.set_zlim(*axis_limits[2])
    #
    # Set axis labels and sizes
    ax.tick_params(axis='both', which='major', labelsize=14, pad=8)
    ax.set_xlabel(axis_labels[0], fontsize=16, labelpad=16)
    ax.set_ylabel(axis_labels[1], fontsize=16, labelpad=16)
    ax.set_zlabel(axis_labels[2], fontsize=16, labelpad=16)
    #
    # Plot pixel values with colors given in colors_rgb
    ax.scatter(
        pixels[:, :, 0].ravel(),
        pixels[:, :, 1].ravel(),
        pixels[:, :, 2].ravel(),
        c=colors_rgb.reshape((-1, 3)), edgecolors='none')
    #
    return ax  # return Axes3D object for further manipulation


