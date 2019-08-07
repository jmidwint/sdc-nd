# This code is taken from the Udacity Classoom SDC "Project: Vehicle Detection & Tracking"
#  Modifications:
#      
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.ndimage.measurements import label

# Define a function that takes an image, a list of bounding boxes, 
# and optional color tuple and line thickness as inputs
# then draws boxes in that color on the output

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # make a copy of the image
    draw_img = np.copy(img)
    # draw each bounding box on your image copy using cv2.rectangle()
    # return the image copy with boxes drawn
    for x1y1, x2y2 in bboxes:
        cv2.rectangle(draw_img, x1y1, x2y2 , color, thick)
    return draw_img # Change this line to return image copy with boxes


# Uncomment this section to test this function.
# =============================================
# Add bounding boxes in this format, these are just example coordinates.
# bboxes = [((100, 100), (200, 200)), ((300, 300), (400, 400))]
# bboxes = [((250, 500), (390, 550)), ((850, 500), (1150, 650)) ]
# 
#image = mpimg.imread('../examples/bbox-example-image.jpg')
# Solution for image - hard coded. 
#bboxes = [((275, 572), (380, 510)), ((488, 563), (549, 518)), ((554, 543), (582, 522)), 
#          ((601, 555), (646, 522)), ((657, 545), (685, 517)), ((849, 678), (1135, 512))]          
#result = draw_boxes(image, bboxes)
#plt.imshow(result)

###################
#
#  This code is taken from the Udacity Classoom SDC "Project: Vehicle Detection & Tracking" 
#     - Lesson 37 Multiple Detections & False Positives



def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    #
    # Return updated heatmap
    return heatmap # Iterate through list of bboxes



    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

# Try a simple algorithm of just removing boxes that had a confidence factor 
#   less then the threshold
def apply_threshold_of_confidence(car_list, conf_list, thresh):
    keep_car, keep_conf = [], []
    for car, conf in zip(car_list, conf_list):
        if conf >= thresh:
            keep_car.append(car)
            keep_conf.append(conf)                
    return keep_car, keep_conf 


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img



