# Advanced Lane Finding - Helper Procedures
#   This script is called by other scripts to provide helper functions
#   used to display images to help with debug & verification, or to write 
#   text onto images.
#   

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

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


def print_my_image(image, title, img, gray=True):
    '''
    To print the image
    '''
    if gray: 
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)
    plt.title(title + img)
    plt.show() 


def print_text_on_image(image, text, position):
    # Function to print text on the image
    #  Prints in white.
    font=cv2.FONT_HERSHEY_DUPLEX
    colour=[255, 255, 255]
    image = cv2.putText(image, text, position, cv2.FONT_HERSHEY_DUPLEX, 2, color=colour, thickness=2)
    return image

