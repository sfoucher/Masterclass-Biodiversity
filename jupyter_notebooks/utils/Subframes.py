import os
from PIL import Image
from albumentations import Compose, BboxParams, Crop
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
import json
import time
import datetime
from datetime import date
import csv
from os.path import exists, join, basename, splitext
from mmdet.apis import inference_detector, init_detector
import collections
import pandas as pd
import shutil

class Subframes(object):
    ''' 
    Class allowing the visualisation and the cropping of a labeled 
    image (bbox) into sub-frames whose dimensions are specified 
    by the user.

    Attributes
    -----------
    img_name : str
        name of the image (with extension, e.g. "My_image.JPG").
    image : PIL
        PIL image.
    target : dict
        Must have 'boxes' and 'labels' keys at least.
        'boxes' must be a list in the 'coco' bounding box format :
        [[xmin, ymin, width, height], ...]
    width : int
        width of the sub-frames
    height : int
        height of the sub-frames
    strict : bool
        set to True get sub-frames of exact same size 
        (e.g width x height) (default: False)
    
    Methods
    --------
    getlist(overlap=False)
        Produces a results list containing, for each row :
        the sub-frame (3D list, dtype=uint8), the bboxes (2D list),
        the labels (1D list) and the filename (str).
    visualise(results)
        Displays ordered sub-frames of the entire image.
    topoints(results)
        Converts the bounding boxes into points annotations.
    displayobjects(results, points_results, ann_type='point')
        Displays only sub-frames containing objects.
    save(results, output_path, object_only=True)
        Saves sub-frames to a specific path.
    '''

    ####################################
    ### ORIGINAL '__init__' FUNCTION ###
    ####################################
    # def __init__(self, img_name, image, target, width, height, strict=False):
    #     '''
    #     Parameters
    #     -----------
    #     img_name : str
    #         name of the image (with extension, e.g. "My_image.JPG")
    #     image : PIL
    #         PIL image
    #     target : dict
    #         Must have 'boxes' and 'labels' keys at least.
    #     width : int
    #         width of the sub-frames
    #     height : int
    #         height of the sub-frames
    #     strict : bool
    #         set to True get sub-frames of exact same size 
    #         (e.g width x height) (default: False)
    #     '''

    #     self.img_name = img_name
    #     self.image = image
    #     self.target = target
    #     self.width = width
    #     self.height = height
    #     self.strict = strict

    #     self.img_width = image.size[0]
    #     self.img_height = image.size[1]

    #     self.x_sub = 1 + int((self.img_width - (self.img_width % width)) / width)
    #     self.y_sub = 1 + int((self.img_height - (self.img_height % height)) / height)
    ####################################
    ### MODIFIED '__init__' FUNCTION ###
    ####################################
    def __init__(self, img_name, img_pillow, img_target, sfm_width, sfm_height, sfm_strict_size=False):
        """
        Instantiates a 'Subframes' object.

        Args:       
            img_name (str):
                The name of the original unsliced image with its extension. (e.g. "My_image.JPG")
            img_pillow (PIL):
                An image opened with the PILLOW Python library.
            img_target (dict):
                The associated groundtruthes (i.e. target) of the original unsliced image. The dictionnary must contain at minimum the 'boxes' and 'labels' keys.
            sfm_width (int):
                The user-specified width of the sub-frames.
            sfm_height (int):
                The user-specified height of the sub-frames.
            sfm_strict_size (bool):
                If set to 'True', all the sub-frames will be of the same exact size. If set to 'False', the size of the sub-frames may differ from one another. (default: False)
        """
        # Specify the relevant attributes for the original unsliced image.
        self.img_name = img_name
        self.img_pillow = img_pillow
        self.img_width = img_pillow.size[0]
        self.img_height = img_pillow.size[1]
        self.img_target = img_target

        # Specify the relevant attributes for the sub-frames.
        self.sfm_width = sfm_width
        self.sfm_height = sfm_height
        self.sfm_strict_size = sfm_strict_size

        # TODO: What is this?
        self.x_sub = int(self.img_width // sfm_width) + 1
        self.y_sub = int(self.img_height // sfm_height) + 1
        
        # print("img_name: {}\n"
        #       "img_pillow: {}\n"
        #       "img_width : {}\n"
        #       "img_height: {}\n"
        #       "img_target: {}\n"
        #       "sfm_width : {}\n"
        #       "sfm_height: {}\n"
        #       "sfm_size_strict: {}\n"
        #       "x_sub: {}\n"
        #       "y_sub: {}\n"
        #       .format(self.img_name, self.img_pillow, self.img_width, self.img_height, self.img_target, self.sfm_width, self.sfm_height, self.sfm_strict_size, self.x_sub, self.y_sub))




    ###################################
    ### ORIGINAL 'getlist' FUNCTION ###
    ###################################
    # def getlist(self, overlap=False):
    #     '''
    #     Produces a results list containing, for each row :
    #     the sub-frame (3D list, dtype=uint8), the bboxes (2D list),
    #     the labels (1D list) and the filename (str).
    #     Parameters
    #     -----------
    #     overlap : bool, optional
    #         Set to True to get an overlap of 50% between 
    #         2 sub-frames (default: False)
    #     Returns
    #     --------
    #     list
    #     '''
    #     height = self.height
    #     width = self.width
    #     img_height = self.img_height
    #     img_width = self.img_width

    #     results = []

    #     # Image preprocessing      
    #     image_np = np.array(self.image)
    #     boxes = self.target['boxes']
    #     labels = self.target['labels']
    #     annotations = {'image':image_np,'bboxes':boxes,'labels':labels}

    #     # Crop lists
    #     if overlap is True:
    #         overlap = 0.5
    #         y_sub = int(np.round(height*overlap))
    #         x_sub = int(np.round(width*overlap))
    #         rg_ymax = img_height-y_sub
    #         rg_xmax = img_width-x_sub
    #     else:
    #         y_sub = height
    #         x_sub = width
    #         rg_ymax = img_height
    #         rg_xmax = img_width

    #     crops = []

    #     for y in range(0, rg_ymax, y_sub):
    #         if  y+height <= img_height:
    #             for x in range(0, rg_xmax, x_sub):
    #                 if  x+width <= img_width:
    #                     xmin, ymin = x, y
    #                     xmax, ymax = x+width, y+height
    #                 elif x+img_width%width <= img_width:
    #                     xmin, ymin = img_width - width, y
    #                     xmax, ymax = x+img_width%width, y+height

    #                 if self.strict is True:
    #                     crops.append([xmin, ymin, xmax, ymax])
    #                 else:
    #                     crops.append([x, y, xmax, ymax])
            
    #         elif  y+img_height%height <= img_height:
    #             for x in range(0, rg_xmax, x_sub):
    #                 if  x+width <= img_width:
    #                     xmin, ymin = x, img_height - height
    #                     xmax, ymax = x+width, y+img_height%height
    #                 elif x+img_width%width <= img_width:
    #                     xmin, ymin = img_width - width, img_height - height
    #                     xmax, ymax = x+img_width%width, y+img_height%height

    #                 if self.strict is True:
    #                     crops.append([xmin, ymin, xmax, ymax])
    #                 else:
    #                     crops.append([x, y, xmax, ymax])

    #     sub = 0
    #     for xmin, ymin, xmax, ymax in crops:
    #         transf = Compose([Crop(xmin, ymin, xmax, ymax, p=1.0)], 
    #                             bbox_params=BboxParams(format='coco',
    #                                                     min_visibility=0.25, 
    #                                                     label_fields=['labels']))
    #         augmented  = transf(**annotations)
    #         sub_name = self.img_name.rsplit('.')[0] + "_S" + str(sub) + ".JPG"
    #         results.append([augmented['image'],augmented['bboxes'],augmented['labels'],sub_name])
    #         sub += 1

    #     return results
    # ###################################
    # ### MODIFIED 'getlist' FUNCTION ###
    # ###################################
    # def getlist(self, sfm_overlap=False):
    #     """
    #     Function that creates a list containing, for each row, the following elements: the sub-frame (3D list, dtype=uint8), the annotations's labels (1D list), the annotation's bounding boxes (2D list) and the filename (str).

    #     Parameters:
    #         sf_size_overlap (bool, optional):
    #             If set to 'True', an overlap of 50 % will be considered between two consecutive sub-frames. If set to 'False', no overlap will be considered between two consecutive sub-frames. (default: False)
        
    #     Returns:
    #         results (list):
    #             TODO: Put description.
    #     """
    #     # Fetch the relevant variables from the 'Subframes' object.
    #     image_np = np.array(self.img_pillow)
    #     img_width = self.img_width
    #     img_height = self.img_height
    #     img_target_labels = self.img_target["anno_labels"]
    #     img_target_bboxes = self.img_target["anno_bboxes"]
    #     sfm_width = self.sfm_width
    #     sfm_height = self.sfm_height

    #     # Create empty lists to store relevant information.
    #     crops = []
    #     results = []
    #     x_axis_minimum = -9000
    #     row_cnt = 0
    #     y_axis_minimum = -9000
    #     col_cnt = 0

    #     # Define a dictionnary to store annotations information for each original unsliced images.
    #     annotations = {
    #         "image": image_np,
    #         "bboxes": img_target_bboxes,
    #         "labels": img_target_labels
    #     }

    #     # If a 50 % overlap between two consecutive sub-frames is wanted.
    #     if sfm_overlap is True:
    #         sfm_overlap = 0.5
    #         x_sub = int(np.round(sfm_width * sfm_overlap))      # Width of the 'X' step for each sub-frame.
    #         y_sub = int(np.round(sfm_height * sfm_overlap))     # Height of the 'Y' step for each sub-frame.
    #         rg_xmax = img_width - x_sub                         # Width of the image corrected for the 'X' step.
    #         rg_ymax = img_height - y_sub                        # Height of the image corrected for the 'Y' step.
    #     # If a 50 % overlap between two consecutive sub-frames is not wanted.
    #     else:
    #         x_sub = sfm_width                                   # Width of the 'X' step for each sub-frame.
    #         y_sub = sfm_height                                  # Height of the 'Y' step for each sub-frame.
    #         rg_xmax = img_width                                 # Width of the image corrected for the 'X' step.
    #         rg_ymax = img_height                                # Height of the image corrected for the 'Y' step.

    #     # Parse through all minimum upper left 'Y' coordinates.
    #     for y in range(0, rg_ymax, y_sub):
    #         #TODO: Verify if the following comment is factual or not.
    #         # Cases when the sub-frames' height is smaller than the image's height.
    #         if  (y + sfm_height) <= img_height:
    #             # Parse through all minimum upper left 'X' coordinates.
    #             for x in range(0, rg_xmax, x_sub):
    #                 #TODO : Verify if the following comment is factual or not.
    #                 # Cases when the sub-frames' width is smaller than the image's width.
    #                 if  (x + sfm_width) <= img_width:
    #                     xmin = x                                # Minimum upper left 'X' coordinate.
    #                     ymin = y                                # Minimum upper left 'Y' coordinate.
    #                     xmax = x + sfm_width                    # Maximum lower right 'X' coordinate.
    #                     ymax = y + sfm_height                   # Maximum lower right 'Y' coordinate.
    #                 #TODO : Verify if the following comment is factual or not.
    #                 # Cases when the sub-frames' width is bigger than the image's width.
    #                 elif (x + img_width % sfm_width) <= img_width:
    #                     xmin = img_width - sfm_width            # Minimum upper left 'X' coordinate.
    #                     ymin = y                                # Minimum upper left 'Y' coordinate.
    #                     xmax = x + img_width % sfm_width        # Maximum lower right 'X' coordinate.
    #                     ymax = y + sfm_height                   # Maximum lower right 'Y' coordinate.
    #                 # Add the sub-frames' upper left and lower right ('X' and 'Y') coordinates to a list.
    #                 if self.sfm_strict_size is True:
    #                     crops.append([xmin, ymin, xmax, ymax])
    #                 else:
    #                     crops.append([x, y, xmax, ymax])
    #         #TODO: Verify if the following comment is factual or not.
    #         # Cases when the sub-frames' height is bigger than the image's height.
    #         elif  (y + img_height % sfm_height) <= img_height:
    #             # Parse through all minimum upper left 'X' coordinates.
    #             for x in range(0, rg_xmax, x_sub):
    #                 #TODO : Verify if the following comment is factual or not.
    #                 # Cases when the sub-frames' width is smaller than the image's width.
    #                 if  (x + sfm_width) <= img_width:
    #                     xmin = x                                # Minimum upper left 'X' coordinate.
    #                     ymin = img_height - sfm_height          # Minimum upper left 'Y' coordinate.
    #                     xmax = x + sfm_width                    # Maximum lower right 'X' coordinate.
    #                     ymax = y + img_height % sfm_height      # Maximum lower right 'Y' coordinate.
    #                 #TODO : Verify if the following comment is factual or not.
    #                 # Cases when the sub-frames' width is smaller than the image's width.
    #                 elif (x + img_width % sfm_width) <= img_width:
    #                     xmin = img_width - sfm_width            # Minimum upper left 'X' coordinate.
    #                     ymin = img_height - sfm_height          # Minimum upper left 'Y' coordinate.
    #                     xmax = x + img_width % sfm_width        # Maximum lower right 'X' coordinate.
    #                     ymax = y + img_height % sfm_height      # Maximum lower right 'Y' coordinate.
    #                 # Add the sub-frames' upper left and lower right ('X' and 'Y') coordinates to a list.
    #                 if self.sfm_strict_size is True:
    #                     crops.append([xmin, ymin, xmax, ymax])
    #                 else:
    #                     crops.append([x, y, xmax, ymax])

    #     subframe_count = 0
    #     # Parse through all items contained within the 'crops' list.
    #     for xmin, ymin, xmax, ymax in crops:
    #         # print("img_name:{}   img_width: {}   img_height: {}".format(self.img_name, img_width, img_height))
    #         # Define the augmentation pipeline with the 'Compose' class of the Albumentations Python module.
    #         transf = Compose([
    #             # Crop a region from the image.
    #             Crop(
    #                 x_min = xmin,                            # Minimum upper left x coordinate.
    #                 y_min = ymin,                            # Minimum upper left y coordinate.
    #                 x_max = xmax,                            # Maximum lower right x coordinate.
    #                 y_max = ymax,                            # Maximum lower right y coordinate.
    #                 p = 1.0                                  # Probability of applying the transform.
    #             )],
    #             # Specify the settings for working with the image's associated annotation's bounding boxes.
    #             # For more information,  https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/
    #             bbox_params = BboxParams(
    #                 format = "coco",                        # format of the bounding boxes.
    #                 min_visibility = 0.25,                  # Value between 0 and 1. Controls what to do with the augmented bounding boxes if their size has changed after augmentation.
    #                 label_fields = ["labels"]               # Set names for all arguments in 'transf' that will contain label descriptions for bounding boxes (i.e. name of the classes).
    #             )
    #         )

    #         # Apply the augmentation pipeline on the images stored in the 'annotations' dictionnary.
    #         augmented  = transf(**annotations)

    #         # print(xmin, ymin, xmax, ymax)

    #         # Create a name for the newly generated sub-frame.
    #         # print("xmin: {}   ymin: {}   xmax: {}   ymax: {}".format(xmin, ymin, xmax, ymax))
    #         if y_axis_minimum < ymin:
    #             y_axis_minimum = ymin
    #             row_cnt += 1
    #             if x_axis_minimum < xmin:
    #                 x_axis_minimum = xmin
    #                 col_cnt += 1
    #                 subframe_name = self.img_name.rsplit(".")[0] + "_S" + str(subframe_count) + "_R" + str(row_cnt) + "_C" + str(col_cnt) + ".JPG"
    #             elif x_axis_minimum == xmin:
    #                 subframe_name = self.img_name.rsplit(".")[0] + "_S" + str(subframe_count) + "_R" + str(row_cnt) + "_C" + str(col_cnt) + ".JPG"
    #             else:
    #                 x_axis_minimum = -9000
    #                 col_cnt = 1
    #                 subframe_name = self.img_name.rsplit(".")[0] + "_S" + str(subframe_count) + "_R" + str(row_cnt) + "_C" + str(col_cnt) + ".JPG"
    #         elif y_axis_minimum == ymin:
    #             if x_axis_minimum < xmin:
    #                 x_axis_minimum = xmin
    #                 col_cnt += 1
    #                 subframe_name = self.img_name.rsplit(".")[0] + "_S" + str(subframe_count) + "_R" + str(row_cnt) + "_C" + str(col_cnt) + ".JPG"
    #             elif x_axis_minimum == xmin:
    #                 subframe_name = self.img_name.rsplit(".")[0] + "_S" + str(subframe_count) + "_R" + str(row_cnt) + "_C" + str(col_cnt) + ".JPG"
    #             else:
    #                 x_axis_minimum = -9000
    #                 col_cnt = 1
    #                 subframe_name = self.img_name.rsplit(".")[0] + "_S" + str(subframe_count) + "_R" + str(row_cnt) + "_C" + str(col_cnt) + ".JPG"
    #         else:
    #             print("MEGA ERROR")
    #             break
            
    #         # subframe_name = self.img_name.rsplit(".")[0] + "_S" + str(subframe_count) + ".JPG"
    #         # print("subframe_name: {}".format(subframe_name))

    #         # Append the results to the 'results' list.
    #         results.append(
    #             [augmented["image"], augmented["bboxes"], augmented["labels"], subframe_name]
    #         )

    #         # Increment the subframe count by one.
    #         subframe_count += 1

    #     return results
    #############################################
    ### MODIFIED 'getlist' FUNCTION VERSION 2 ###
    #############################################
    def getlist(self, sfm_overlap=False):
        """
        Function that creates a list containing, for each row, the following elements: the sub-frame (3D list, dtype=uint8), the annotations's labels (1D list), the annotation's bounding boxes (2D list) and the filename (str).

        Parameters:
            sf_size_overlap (bool, optional):
                If set to 'True', an overlap of 50 % will be considered between two consecutive sub-frames. If set to 'False', no overlap will be considered between two consecutive sub-frames. (default: False)
        
        Returns:
            results (list):
                TODO: Put description.
        """
        # Fetch the relevant variables from the 'Subframes' object.
        image_np = np.array(self.img_pillow)
        img_width = self.img_width
        img_height = self.img_height
        img_target_labels = self.img_target["anno_labels"]
        img_target_bboxes = self.img_target["anno_bboxes"]
        sfm_width = self.sfm_width
        sfm_height = self.sfm_height

        # Create empty lists to store relevant information.
        crops = []
        results = []
        x_axis_minimum = -9000
        row_cnt = 0
        y_axis_minimum = -9000
        col_cnt = 0

        # Define a dictionnary to store annotations information for each original unsliced images.
        annotations = {
            "image": image_np,
            "bboxes": img_target_bboxes,
            "labels": img_target_labels
        }

        if sfm_overlap is True:
            x_sub = int(np.round(sfm_width * sfm_overlap))  # width to move the subframe center
            y_sub = int(np.round(sfm_height * sfm_overlap))  # height to move the subframe center
            rg_xmax = img_width - (img_width % x_sub)  # new max width of image
            rg_ymax = img_height - (img_height % y_sub)  # new max height of image
        else:
            x_sub = sfm_width  # width to move the subframe center
            y_sub = sfm_height  # height to move the subframe center
            rg_xmax = img_width - (img_width % x_sub)  # new max width of image. The modulo is used to remove unwanted pixels.
            rg_ymax = img_height - (img_height % y_sub)
        # # If a 50 % overlap between two consecutive sub-frames is wanted.
        # if sfm_overlap is True:
        #     sfm_overlap = 0.5
        #     x_sub = int(np.round(sfm_width * sfm_overlap))      # Width of the 'X' step for each sub-frame.
        #     y_sub = int(np.round(sfm_height * sfm_overlap))     # Height of the 'Y' step for each sub-frame.
        #     rg_xmax = img_width - x_sub                         # Width of the image corrected for the 'X' step.
        #     rg_ymax = img_height - y_sub                        # Height of the image corrected for the 'Y' step.
        # # If a 50 % overlap between two consecutive sub-frames is not wanted.
        # else:
        #     x_sub = sfm_width                                   # Width of the 'X' step for each sub-frame.
        #     y_sub = sfm_height                                  # Height of the 'Y' step for each sub-frame.
        #     rg_xmax = img_width                                 # Width of the image corrected for the 'X' step.
        #     rg_ymax = img_height                                # Height of the image corrected for the 'Y' step.

        for y in range(0, rg_ymax, y_sub):
            if (y + sfm_height) <= img_height:
                for x in range(0, rg_xmax, x_sub):
                    if (x + sfm_width) <= img_width:
                        xmin, ymin = x, y
                        xmax, ymax = (x + sfm_width), (y + sfm_height)
                        if self.sfm_strict_size is True:
                            crops.append([xmin, ymin, xmax, ymax])
                        else:
                            crops.append([x, y, xmax, ymax])
        print("crops: {}".format(crops))
        # # Parse through all minimum upper left 'Y' coordinates.
        # for y in range(0, rg_ymax, y_sub):
        #     #TODO: Verify if the following comment is factual or not.
        #     # Cases when the sub-frames' height is smaller than the image's height.
        #     if  (y + sfm_height) <= img_height:
        #         # Parse through all minimum upper left 'X' coordinates.
        #         for x in range(0, rg_xmax, x_sub):
        #             #TODO : Verify if the following comment is factual or not.
        #             # Cases when the sub-frames' width is smaller than the image's width.
        #             if  (x + sfm_width) <= img_width:
        #                 xmin = x                                # Minimum upper left 'X' coordinate.
        #                 ymin = y                                # Minimum upper left 'Y' coordinate.
        #                 xmax = x + sfm_width                    # Maximum lower right 'X' coordinate.
        #                 ymax = y + sfm_height                   # Maximum lower right 'Y' coordinate.
        #             #TODO : Verify if the following comment is factual or not.
        #             # Cases when the sub-frames' width is bigger than the image's width.
        #             elif (x + img_width % sfm_width) <= img_width:
        #                 xmin = img_width - sfm_width            # Minimum upper left 'X' coordinate.
        #                 ymin = y                                # Minimum upper left 'Y' coordinate.
        #                 xmax = x + img_width % sfm_width        # Maximum lower right 'X' coordinate.
        #                 ymax = y + sfm_height                   # Maximum lower right 'Y' coordinate.
        #             # Add the sub-frames' upper left and lower right ('X' and 'Y') coordinates to a list.
        #             if self.sfm_strict_size is True:
        #                 crops.append([xmin, ymin, xmax, ymax])
        #             else:
        #                 crops.append([x, y, xmax, ymax])
        #     #TODO: Verify if the following comment is factual or not.
        #     # Cases when the sub-frames' height is bigger than the image's height.
        #     elif  (y + img_height % sfm_height) <= img_height:
        #         # Parse through all minimum upper left 'X' coordinates.
        #         for x in range(0, rg_xmax, x_sub):
        #             #TODO : Verify if the following comment is factual or not.
        #             # Cases when the sub-frames' width is smaller than the image's width.
        #             if  (x + sfm_width) <= img_width:
        #                 xmin = x                                # Minimum upper left 'X' coordinate.
        #                 ymin = img_height - sfm_height          # Minimum upper left 'Y' coordinate.
        #                 xmax = x + sfm_width                    # Maximum lower right 'X' coordinate.
        #                 ymax = y + img_height % sfm_height      # Maximum lower right 'Y' coordinate.
        #             #TODO : Verify if the following comment is factual or not.
        #             # Cases when the sub-frames' width is smaller than the image's width.
        #             elif (x + img_width % sfm_width) <= img_width:
        #                 xmin = img_width - sfm_width            # Minimum upper left 'X' coordinate.
        #                 ymin = img_height - sfm_height          # Minimum upper left 'Y' coordinate.
        #                 xmax = x + img_width % sfm_width        # Maximum lower right 'X' coordinate.
        #                 ymax = y + img_height % sfm_height      # Maximum lower right 'Y' coordinate.
        #             # Add the sub-frames' upper left and lower right ('X' and 'Y') coordinates to a list.
        #             if self.sfm_strict_size is True:
        #                 crops.append([xmin, ymin, xmax, ymax])
        #             else:
        #                 crops.append([x, y, xmax, ymax])

        subframe_count = 0
        # Parse through all items contained within the 'crops' list.
        for xmin, ymin, xmax, ymax in crops:
            # print("img_name:{}   img_width: {}   img_height: {}".format(self.img_name, img_width, img_height))
            # Define the augmentation pipeline with the 'Compose' class of the Albumentations Python module.
            transf = Compose([
                # Crop a region from the image.
                Crop(
                    x_min = xmin,                            # Minimum upper left x coordinate.
                    y_min = ymin,                            # Minimum upper left y coordinate.
                    x_max = xmax,                            # Maximum lower right x coordinate.
                    y_max = ymax,                            # Maximum lower right y coordinate.
                    p = 1.0                                  # Probability of applying the transform.
                )],
                # Specify the settings for working with the image's associated annotation's bounding boxes.
                # For more information,  https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/
                bbox_params = BboxParams(
                    format = "coco",                        # format of the bounding boxes.
                    min_visibility = 0.25,                  # Value between 0 and 1. Controls what to do with the augmented bounding boxes if their size has changed after augmentation.
                    label_fields = ["labels"]               # Set names for all arguments in 'transf' that will contain label descriptions for bounding boxes (i.e. name of the classes).
                )
            )

            # Apply the augmentation pipeline on the images stored in the 'annotations' dictionnary.
            augmented  = transf(**annotations)

            # print(xmin, ymin, xmax, ymax)

            # Create a name for the newly generated sub-frame.
            # print("xmin: {}   ymin: {}   xmax: {}   ymax: {}".format(xmin, ymin, xmax, ymax))
            if y_axis_minimum < ymin:
                y_axis_minimum = ymin
                row_cnt += 1
                if x_axis_minimum < xmin:
                    x_axis_minimum = xmin
                    col_cnt += 1
                    subframe_name = self.img_name.rsplit(".")[0] + "_S" + str(subframe_count) + "_R" + str(row_cnt) + "_C" + str(col_cnt) + ".JPG"
                    print("     {}".format(subframe_name))
                elif x_axis_minimum == xmin:
                    subframe_name = self.img_name.rsplit(".")[0] + "_S" + str(subframe_count) + "_R" + str(row_cnt) + "_C" + str(col_cnt) + ".JPG"
                    print("     {}".format(subframe_name))
                else:
                    x_axis_minimum = -9000
                    col_cnt = 1
                    subframe_name = self.img_name.rsplit(".")[0] + "_S" + str(subframe_count) + "_R" + str(row_cnt) + "_C" + str(col_cnt) + ".JPG"
                    print("     {}".format(subframe_name))
            elif y_axis_minimum == ymin:
                if x_axis_minimum < xmin:
                    x_axis_minimum = xmin
                    col_cnt += 1
                    subframe_name = self.img_name.rsplit(".")[0] + "_S" + str(subframe_count) + "_R" + str(row_cnt) + "_C" + str(col_cnt) + ".JPG"
                    print("     {}".format(subframe_name))
                elif x_axis_minimum == xmin:
                    subframe_name = self.img_name.rsplit(".")[0] + "_S" + str(subframe_count) + "_R" + str(row_cnt) + "_C" + str(col_cnt) + ".JPG"
                    print("     {}".format(subframe_name))
                else:
                    x_axis_minimum = -9000
                    col_cnt = 1
                    subframe_name = self.img_name.rsplit(".")[0] + "_S" + str(subframe_count) + "_R" + str(row_cnt) + "_C" + str(col_cnt) + ".JPG"
                    print("     {}".format(subframe_name))
            else:
                print("##########")
                print("MEGA ERROR")
                print("##########")
                break
            
            # subframe_name = self.img_name.rsplit(".")[0] + "_S" + str(subframe_count) + ".JPG"
            # print("subframe_name: {}".format(subframe_name))

            # Append the results to the 'results' list.
            results.append(
                [augmented["image"], augmented["bboxes"], augmented["labels"], subframe_name]
            )

            # Increment the subframe count by one.
            subframe_count += 1

        return results


    def visualise(self, results):
        '''
        Displays ordered sub-frames of the entire image.
        Parameters
        -----------
        results : list
            The list obtained by the method getlist().
        Returns
        --------
        matplotlib plot
        '''

        if len(results) > (self.x_sub*self.y_sub):
            x_sub = 2*self.x_sub - 2
            y_sub = 2*self.y_sub - 2
        else:
            x_sub = self.x_sub
            y_sub = self.y_sub

        plt.figure(1)
        plt.suptitle(self.img_name)
        sub = 1
        for line in range(len(results)):

            if self.img_width % self.width != 0:
                n_col = x_sub
                n_row = y_sub
            else:
                n_col = x_sub - 1
                n_row = y_sub - 1

            plt.subplot(n_row, n_col, sub, xlim=(0,self.width), ylim=(self.height,0))
            plt.imshow(Image.fromarray(results[line][0]))
            plt.axis('off')
            plt.subplots_adjust(wspace=0.1,hspace=0.1)

            text_x = np.shape(results[line][0])[1]
            text_y = np.shape(results[line][0])[0]

            if self.width > self.height:
                f = self.height*(self.y_sub/y_sub)
            else:
                f = self.width*(self.x_sub/x_sub)

            plt.text(0.5*text_x, 0.5*text_y, 
                    "S"+str(line),
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=0.02*f,
                    color='w')
            sub += 1

    def topoints(self, results):
        '''
        Converts the bounding boxes into points annotations.
        Parameters
        -----------
        results : list
            The list obtained by the method getlist().
        Returns
        --------
        list
            A 2D list with headers : "id", "filename", "count",
            "locations" where
            - "id" represents the unique id of the sub-frame within 
              the image
            - "filename" is the name of the sub-frame 
              (e.g. "My_image_S1.JPG")
            - "count" is the number of objects into the sub-frame
            - "points" is a list of tuple representing the 
              locations of the objects (y,x)
    
        '''

        points_results = [['id','filename','count','locations']]
        loc = []
        for line in range(len(results)):
            # Verify that bbox exists
            if results[line][1]:
                count = len(results[line][1])
                for bbox in range(len(results[line][1])):
                    boxe = results[line][1][bbox]
                    x = int(boxe[0]+(boxe[2])/2)
                    y = int(boxe[1]+(boxe[3])/2)
                    point = (y,x)
                    loc.append(point)
            
                sub_name = self.img_name.rsplit('.')[0] + "_S" + str(line) + ".JPG"
                points_results.append([line, sub_name, count, loc])
                loc = []

        return points_results

    def displayobjects(self, results, points_results, ann_type='point'):
        '''
        Displays only sub-frames containing objects.
        Parameters
        -----------
        results : list
            The list obtained by the method getlist().
        points_results : list
            The list obtained by the method topoints(results).
        ann_type : str, optional
            A string used to specify the annotation type. Choose
            between :
            - 'point' to visualise points
            - 'bbox' to visualise bounding boxes
            - 'both' to visualise both
            (default is 'point')
        Returns
        --------
        matplotlib plot
        '''

        sub_r = 0
        sub_c = 0

        n_row = int(np.round(math.sqrt(len(points_results)-1)))
        n_col = n_row

        if int(len(points_results)-1) > int(n_row*n_col):
            n_row += 1

        fig, ax = plt.subplots(nrows=n_row, ncols=n_col, squeeze=False)

        for r in range(n_row):
            for c in range(n_col):
                ax[r,c].axis('off')
                plt.subplots_adjust(wspace=0.1,hspace=0.1)

        for o in range(1,len(points_results)):

            id_object = points_results[o][0]
            patch_object = results[id_object][0]

            text_x = np.shape(results[id_object][0])[1]
            text_y = np.shape(results[id_object][0])[0]

            # Plot
            ax[sub_r,sub_c].imshow(Image.fromarray(patch_object))
            ax[sub_r,sub_c].text(0.5*text_x, 0.5*text_y, 
                    "S"+str(id_object),
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=15,
                    color='w',
                    alpha=0.6)

            if ann_type == 'point':
                points = points_results[o][3]
                for p in range(len(points)):
                    ax[sub_r,sub_c].scatter(points[p][1],points[p][0], color='r')
            
            elif ann_type == 'bbox':
                bboxes = results[id_object][1]
                for b in range(len(bboxes)):
                    rect = patches.Rectangle((bboxes[b][0],bboxes[b][1]),bboxes[b][2],bboxes[b][3], linewidth=1, edgecolor='r', facecolor='none')
                    ax[sub_r,sub_c].add_patch(rect)
                
            elif ann_type == 'both':
                points = points_results[o][3]
                bboxes = results[id_object][1]
                for b in range(len(bboxes)):
                    ax[sub_r,sub_c].scatter(points[b][1],points[b][0], color='b')
                    rect = patches.Rectangle((bboxes[b][0],bboxes[b][1]),bboxes[b][2],bboxes[b][3], linewidth=1, edgecolor='r', facecolor='none')
                    ax[sub_r,sub_c].add_patch(rect)

            else:
                raise ValueError('Annotation of type \'{}\' unsupported. Choose between \'point\',\'bbox\' or \'both\'.'.format(ann_type))
                
            if sub_c < n_col-1:
                sub_r = sub_r
                sub_c += 1
            else:
                sub_c = 0
                sub_r += 1
    
    ################################
    ### ORIGINAL 'save' FUNCTION ###
    ################################
    # def save(self, results, output_path, object_only=True):
    #     '''
    #     Saves sub-frames (.JPG) to a specific path.
    #     Parameters
    #     -----------
    #     results : list
    #         The list obtained by the method getlist().
    #     output_path : str
    #         The path to the folder chosen to save sub-frames.
    #     object_only : bool, optional
    #         A flag used to choose between :
    #         - saving all the sub-frames of the entire image
    #           (set to False)
    #         - saving only sub-frames with objects
    #           (set to True, default)
    #     Returns
    #     --------
    #     None
    #     '''

    #     for line in range(len(results)):
    #         if object_only is True:
    #             if results[line][1]:
    #                 subframe = Image.fromarray(results[line][0])
    #                 sub_name =  results[line][3]
    #                 subframe.save(os.path.join(output_path, sub_name))
                    
    #         elif object_only is not True:
    #             subframe = Image.fromarray(results[line][0])
    #             sub_name =  results[line][3]
    #             subframe.save(os.path.join(output_path, sub_name))
    
    ################################
    ### MODIFIED 'save' FUNCTION ###
    ################################
    def save(self, results, sfm_output_dir, sfm_object_only=True):
        """
        A Python method that saves the newly generated sub-frames (.JPG file) to a user-defined location.

        Parameters:
            results (list):
                The list generated by the 'Subframes.getlist()' Python method.
            sfm_output_dir (str):
                The path to the directory in which to save the newly generated sub-frames.
            sfm_object_only (bool, optional):
                If set to 'True', only the sub-frames that contain relevant targets (i.e. objects) will be saved. If set to 'False', all sub-frames will be saved. (default: True)
        
        Returns:
            None
        """
        # Parse through all sub-frames contained in the 'results' list created with the 'Subframes.getlist()' Python method.
        for sfm in range(len(results)):
            # Only the sub-frames containing relevant targets (i.e. objects) are to be saved.
            if sfm_object_only is True:
                # Checks whether or not the sub-frames has an associated bounding box information. This logical operation will return 'True' if the sub-frame has one or more associated relevant target (i.e. object).
                if results[sfm][1]:
                    # Create an image memory from an object exporting the array interface.
                    subframe = Image.fromarray(
                        obj = results[sfm][0],
                        mode = None
                    )
                    # Fetch the name of the sub-frame to be saved.
                    subframe_name = results[sfm][3]
                    # Save the sub-frame using the PIL 'Image.save()' method.
                    subframe.save(
                        fp = os.path.join(sfm_output_dir, subframe_name)
                    )
            
            # All newly generated sub-frames are to be saved, whether or not they contain relevant targets (i.e. objects).
            elif sfm_object_only is not True:
                # Create an image memory from an object exporting the array interface.
                subframe = Image.fromarray(
                    obj = results[sfm][0],
                    mode = None
                )
                # Fetch the name of the sub-frame to be saved.
                subframe_name =  results[sfm][3]
                # Save the sub-frame using the PIL 'Image.save()' method.
                subframe.save(
                    fp = os.path.join(sfm_output_dir, subframe_name)
                )

class CustomDataset(Dataset):
    
    ####################################
    ### ORIGINAL '__init__' FUNCTION ###
    ####################################
    # def __init__(self, img_root, ann_root, target_type='coco', transforms=None):

    #     self.img_root = img_root
    #     self.ann_root = ann_root
    #     self.target_type = target_type
    #     self.transforms = transforms

    #     with open(ann_root) as json_file:
    #         self.data = json.load(json_file)
    
    ####################################
    ### MODIFIED '__init__' FUNCTION ###
    ####################################
    def __init__(self, img_dir, img_anno_path, img_target_type, sfm_transforms=None):
        """
        Instantiates a 'CustomDataset' dataset.

        Parameters:
            img_dir (str):
                The directory in which the original unsliced images are located.
            img_ann_path (str):
                The annotation file for the original unsliced images with its associated extension (e.g. "My_File.json).
            img_target_type (str):
                The type of annotation file to create for the sub-frames. If set to 'coco', the bounding boxes coordinates for the annotations will be in a coco-style format. If set to 'pascal', the bounding boxes coordinates for the annotations will be in a pascal-style format.
            sfm_transforms (???):
                (default: None) #TODO: Add information.
        """
        self.img_dir = img_dir
        self.img_anno_path = img_anno_path
        self.img_target_type = img_target_type
        self.sfm_transforms = sfm_transforms

        with open(file = img_anno_path, mode = "r") as json_file:
            self.data = json.load(fp = json_file)
        
        # print("img_dir: {}\n"
        #       "img_anno_path: {}\n"
        #       "img_target_type: {}\n"
        #       "sfm_transforms: {}\n"
        #       "data: {}\n"
        #       .format(self.img_dir, self.img_anno_path, self.img_target_type, self.sfm_transforms, self.data))



    #######################################
    ### MODIFIED '__getitem__' FUNCTION ###
    #######################################
    """
    
    Parameters:
        idx (???):
            TODO: Add description.
    
    Returns:
        img_pillow (PIL):
            An image opened with the PILLOW Python library.
        img_target (dict):
            A dictionnary with the following relevant target information: the ID of the original unsliced image, the labels of the original unsliced image's annotations, the bounding boxes of the original unsliced image's annotations and the area of that bounding box.
    """
    def __getitem__(self, idx):

        img_id = self.data["images"][idx]["id"]
        img_name = self.data["images"][idx]["file_name"]
        img_path = os.path.join(self.img_dir, img_name)
        img_pillow = Image.open(img_path).convert("RGB")
        img_target = {}
        bboxes = []
        labels = []
        area = []

        # Parse through all annotations contained in the original annotation file for the original unsliced image.
        for anno in range(0, len(self.data["annotations"])):
            
            # If the ID of the original unsliced image matches the ID specified for a given annotation.
            if self.data["annotations"][anno]["image_id"] == img_id:

                # Fetch the annotation's label (i.e. class).
                labels.append(self.data["annotations"][anno]["category_id"])

                # Fetch the annotation's bounding box in the desired coco-style format.
                if self.img_target_type == "coco":
                    bboxes.append(self.data["annotations"][anno]["bbox"])

                # Fetch the annotation's bounding box in the desired pascal-style format.
                elif self.img_target_type == "pascal":
                    bndbox = self.data["annotations"][anno]["bbox"]
                    xmin = bndbox[0]
                    ymin = bndbox[1]
                    xmax = bndbox[0] + bndbox[2]
                    ymax = bndbox[1] + bndbox[3]
                    bboxes.append([xmin, ymin, xmax, ymax])

                # Fetch the annotation's bounding box area.
                area.append(self.data["annotations"][anno]["area"])

        # Build the dictionnary with the relevant desired information.
        img_target["image_id"] = img_id
        img_target["anno_labels"] = labels
        img_target["anno_bboxes"] = bboxes
        img_target["anno_area"] = area

        return img_pillow, img_target

    def __len__(self):
        return len(self.data["images"])

# Collate_fn
def collate_fn(batch):
    """
    TODO: Add information for this function.
    """
    return tuple(zip(*batch))

#####################################
### ORIGINAL 'subexport' FUNCTION ###
#####################################
# def subexport(img_root, ann_root, width, height, output_folder, 
#             overlap=False, strict=False ,pr_rate=50, 
#             object_only=True, export_ann=True):
#     '''
#     Function that exports sub-frames created on the basis of 
#     images loaded by a dataloader, and their associated new 
#     annotations.

#     This function uses the 'subframes' class for image processing.

#     Parameters
#     -----------
#     img_root : str
#         Path to images.

#     ann_root : str
#         Path to a coco-style dict (.json) containing annotations of 
#         the initial dataset.

#     width : int
#         Width of the sub-frames.
    
#     height : int
#         Height of the sub-frames.
    
#     output_folder : str
#         Output folder path where to save sub-frames and new annotations.
    
#     overlap : bool, optional
#         Set to True to get an overlap of 50% between 
#         2 sub-frames (default: False)
    
#     strict : bool, optional
#         Set to True get sub-frames of exact same size 
#         (e.g width x height) (default: False)

#     pr_rate : int, optional
#         Console print rate of image processing progress.
#         Default : 50
    
#     object_only : bool, optional
#         A flag used to choose between :
#             - saving all the sub-frames of the entire image
#             (set to False)
#             - saving only sub-frames with objects
#             (set to True, default)

#     export_ann : bool, optional
#         A flag used to choose between :
#             - not exporting annotations with sub-frames
#             (set to False)
#             - exporting annotations with sub-frames
#             (set to True, default
   
#     Returns
#     --------
#     list

#     a coco-type JSON file named 'coco_subframes.json'
#     is created inside the subframes' folder
    
#     '''

#     # Get annos
#     with open(ann_root) as json_file:
#         coco_dic = json.load(json_file)

#     # Dataset
#     dataset = CustomDataset(img_root, ann_root, target_type='coco')

#     # Sampler
#     sampler = torch.utils.data.SequentialSampler(dataset)

#     # Collate_fn
#     def collate_fn(batch):
#         return tuple(zip(*batch))

#     # Dataloader
#     dataloader = torch.utils.data.DataLoader(dataset, 
#                                             batch_size=1,
#                                             sampler=sampler,
#                                             num_workers=0,
#                                             collate_fn=collate_fn)

#     # Header
#     all_results = [['filename','boxes','labels','HxW']]

#     # intial time
#     t_i = time.time()

#     for i, (image, target) in enumerate(dataloader):

#         if i == 0:
#             print(' ')
#             print('-'*38)
#             print('Sub-frames creation started...')
#             print('-'*38)

#         elif i == len(dataloader)-1:
#             print('-'*38)
#             print('Sub-frames creation finished!')
#             print('-'*38)

#         image = image[0]
#         target = target[0]

#         # image id and name
#         img_id = int(target['image_id'])
#         for im in coco_dic['images']:
#             if im['id'] == img_id:
#                 img_name = im['file_name']

#         # Get subframes
#         sub_frames = Subframes(img_name, image, target, width, height, strict=strict)
#         results = sub_frames.getlist(overlap=overlap)

#         # Save
#         sub_frames.save(results, output_path=output_folder, object_only=object_only)
        
#         if object_only is True:
#             for b in range(len(results)):
#                 if results[b][1]:
#                     h = np.shape(results[b][0])[0]
#                     w = np.shape(results[b][0])[1]
#                     all_results.append([results[b][3],results[b][1],results[b][2],[h,w]])

#         elif object_only is not True:
#             for b in range(len(results)):
#                 h = np.shape(results[b][0])[0]
#                 w = np.shape(results[b][0])[1]
#                 all_results.append([results[b][3],results[b][1],results[b][2],[h,w]])

#         if i % pr_rate == 0:
#             print('Image [{:<4}/{:<4}] done.'.format(i, len(coco_dic['images'])))

#     # final time
#     t_f = time.time()

#     print('Elapsed time : {}'.format(str(datetime.timedelta(seconds=int(np.round(t_f-t_i))))))
#     print('-'*38)
#     print(' ')

#     return_var = np.array(all_results)[:,:3].tolist()

#     # Export new annos
#     if export_ann is True:
#         file_name = 'coco_subframes.json'
#         output_f = os.path.join(output_folder, file_name)

#         # Initializations
#         images = []
#         annotations = []
#         id_img = 0
#         id_ann = 0

#         for i in range(1,len(all_results)):
            
#             id_img += 1

#             h = all_results[i][3][0]
#             w = all_results[i][3][1]

#             dico_img = {
#                 "license": 1,
#                 "file_name": all_results[i][0],
#                 "coco_url": "None",
#                 "height": h,
#                 "width": w,
#                 "date_captured": "None",
#                 "flickr_url": "None",
#                 "id": id_img
#             }

#             images.append(dico_img)

#             # Bounding boxes
#             if all_results[i][1]:
                
#                 bndboxes = all_results[i][1]

#                 for b in range(len(bndboxes)):

#                     id_ann += 1

#                     bndbox = bndboxes[b]
                    
#                     # Convert 
#                     x_min = int(np.round(bndbox[0]))
#                     y_min = int(np.round(bndbox[1]))
#                     box_w = int(np.round(bndbox[2]))
#                     box_h = int(np.round(bndbox[3]))

#                     coco_box = [x_min,y_min,box_w,box_h]

#                     # Area
#                     area = box_w*box_h

#                     # Label
#                     label_id = all_results[i][2][b]

#                     # Store the values into a dict
#                     dico_ann = {
#                             "segmentation": [[]],
#                             "area": area,
#                             "iscrowd": 0,
#                             "image_id": id_img,
#                             "bbox": coco_box,
#                             "category_id": label_id,
#                             "id": id_ann
#                     }

#                     annotations.append(dico_ann)
        
#         # Update info
#         coco_dic['info']['date_created'] = str(date.today())
#         coco_dic['info']['year'] = str(date.today().year)

#         new_dic = {
#             'info': coco_dic['info'],
#             'licenses': coco_dic['licenses'],
#             'images': images,
#             'annotations': annotations,
#             'categories': coco_dic['categories']
#         }

#         # Export json file
#         with open(output_f, 'w') as outputfile:
#             json.dump(new_dic, outputfile)

#         if os.path.isfile(output_f) is True:
#             print('File \'{}\' correctly saved at \'{}\'.'.format(file_name, output_folder))
#             print(' ')
#         else:
#             print('An error occurs, file \'{}\' not found at \'{}\'.'.format(file_name, output_folder))

#     return return_var

######################################
### CORRECTED 'subexport' FUNCTION ###
######################################
def subexport(img_dir, img_anno_path, sfm_width, sfm_height, sfm_output_dir, sfm_overlap=False, sfm_strict=False, print_rate=50, sfm_object_only=True, sfm_anno_export=True):
    """
    Slices an image and its associated annotations in subframes that can be exported along with their newly generated annotations. This function uses the 'Subframes' class for image processing.

    Parameters:
        img_dir (str):
            The path to the directory containing the original unsliced images.
        img_anno_path (str):
            The path to the annotation file in the COCO format (.json) for the original unsliced images.
        sfm_width (int):
            The width of the newly generated sub-frames (i.e. sliced images).
        sfm_height (int):
            The height of the newly generated sub-frames (i.e. sliced images).
        sfm_output_dir (str):
            The path to the directory in which to save the newly generated sub-frames and annotations.
        sfm_overlap (bool, optional):
            If set to 'True', an overlap of 50 % will be considered between two newly generated consecutive sub-frames. (default: False) TODO: What if set to 'False'?
        sfm_strict (bool, optional):
            If set to 'True', newly generated subframes will be of the same exact size. (default: False) TODO: What if set to 'False'?
        print_rate (int, optional):
            The console print rate for the image processing progress. (default: 50)
        sfm_object_only (bool, optional):
            If set to 'True', only sub-frames containing objects will be saved. If set to 'False', all sub-frames will be saved. (default: True)
        sfm_anno_export (bool, optional):
            If set to 'True', newly generated annotations will be exported. If set to 'False', newly generated annotations will not be exported. (default: True)

    Returns:    
        return_var (list):
            A coco-type JSON file named 'coco_subframes.json' is created inside the subframes' folder.
    """
    # Open and load the annotation file for the original unsliced images.
    with open(file = img_anno_path, mode = "r") as json_file:
        coco_dic = json.load(fp = json_file)

    # Creating a custom dataset using the original unsliced images and their associated annotations.
    dataset = CustomDataset(
        img_dir = img_dir,
        img_anno_path = img_anno_path,
        img_target_type = "coco",
        sfm_transforms = None
    )

    # Creating a sampler using PyTorch's 'SequentialSampler' module.
    # A sampler sequentially samples elements for a given dataset, and always in the same order.
    # https://pytorch.org/docs/stable/_modules/torch/utils/data/sampler.html#SequentialSampler
    sampler = torch.utils.data.SequentialSampler(
        data_source = dataset
    )

    # Creating the dataloader using PyTorch's 'DataLoader' module.
    # A Data loader combines a dataset and a sampler, and provides an iterable over the given dataset.
    # https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    dataloader = torch.utils.data.DataLoader(
        dataset = dataset,
        batch_size = 1,
        shuffle = False,
        sampler = sampler,
        num_workers = 0,
        collate_fn = collate_fn
    )
 
    # Header TODO: What is this?
    all_results = [
        ["filename", "boxes", "labels", "HxW"]
    ]

    # Fetch the initial time at which the creation of the sub-frames process has started.
    t_i = time.time()

    for i, (image, target) in enumerate(dataloader):

        # Checks whether or not the creation process for the sub-frames has started.
        if i == 0:
            # Print an update message in the console.
            print("-" * 38)
            print("Starting creation of the sub-frames...")
            print("-" * 38)

        # Checks whether or not the creation process for the sub-frames has been completed.
        elif i == (len(dataloader) - 1):
            print("-" * 36)
            print("Finished creation of the sub-frames.")
            print("-" * 36)

        # Fetch the original unsliced image in the PIL format.
        image = image[0]
        # Fetch the original unsliced image's associated target (i.e. ground truth).
        target = target[0]
        # Fetch the original unsliced image's ID.
        img_id = int(target["image_id"])
        # Fetch the original unsliced image's file name.
        for im in coco_dic["images"]:
            if im["id"] == img_id:
                img_name = im["file_name"]
                print(img_name)

        # Create a 'subframes' object with the 'Subframes.__init__' Python method.
        sub_frames = Subframes(
            img_name = img_name,
            img_pillow = image,
            img_target = target,
            sfm_width = sfm_width,
            sfm_height = sfm_height,
            sfm_strict_size = sfm_strict
        )
        
        # Create a list of all the sub-frames for a given original unsliced image with the 'Subframes.getlist()' Python method.
        results = sub_frames.getlist(
            sfm_overlap = sfm_overlap
        )

        # Save the newly generated sub-frames with the 'Subframes.save()' Python method.
        sub_frames.save(
            results = results,
            sfm_output_dir = sfm_output_dir,
            sfm_object_only = sfm_object_only
        )
        
        ###TODO: The following if & elif statement is redundant with the process of the 'Subframes.save()' Python method.
        # Use only the sub-frames containing relevant targets (i.e. objects).
        if sfm_object_only is True:
            # Parse through all sub-frames contained in the 'results' list created with the 'Subframes.getlist()' Python method.
            for sfm in range(len(results)):
                # Checks whether or not the sub-frames has an associated bounding box information. This logical operation will return 'True' if the sub-frame has one or more associated relevant target (i.e. object).
                if results[sfm][1]:
                    # Fetch the width of the sub-frame.
                    sfm_width = np.shape(
                        a = results[sfm][0]
                    )[1]
                    # Fetch the height of the sub-frame.
                    sfm_height = np.shape(
                        a = results[sfm][0]
                    )[0]
                    # Add the relevant sub-frame information to a new 'all_results' list.
                    all_results.append([
                        results[sfm][3],
                        results[sfm][1],
                        results[sfm][2],
                        [sfm_height, sfm_width]
                    ])
        # Use all sub-frames, whether or not they contain relevant targets (i.e. objects).
        elif sfm_object_only is not True:
            # Parse through all sub-frames contained in the 'results' list created with the 'Subframes.getlist()' Python method.
            for sfm in range(len(results)):
                # Fetch the width of the sub-frame.
                sfm_width = np.shape(results[sfm][0])[1]
                # Fetch the height of the sub-frame.
                sfm_height = np.shape(results[sfm][0])[0]
                # Add the relevant sub-frame information to a new 'all_results' list.
                all_results.append([
                    results[sfm][3],
                    results[sfm][1],
                    results[sfm][2],
                    [sfm_height,sfm_width]
                ])

        # Checks whether or not an update message should be printed in the console.
        if i % print_rate == 0:
            # Print a update message in the console.
            print("Image [{:<4}/{:<4}] done.".format(i, len(coco_dic["images"])))

    # Fetch the final time at which the creation of the sub-frames process has been completed.
    t_f = time.time()
    # Print the total elapsed time for the creation of the sub-frames.
    print("-" * 38)
    print("Elapsed time : {}".format(str(datetime.timedelta(seconds = int(np.round(t_f - t_i))))))
    print("-" * 38 + "\n")

    #TODO: What is this?
    return_var = np.array(all_results)[:,:3].tolist()

    # Export new annos
    if sfm_anno_export is True:
        file_name = "coco_subframes.json"
        output_f = os.path.join(sfm_output_dir, file_name)

        # Create empty lists and counters for relevant variables.
        subframes = []
        sfm_annotations = []
        id_img = 0
        id_ann = 0

        # Parse through all sub-frames.
        for i in range(1, len(all_results)):
            # Create the subframe ID by incrementing the associated counter.
            id_img += 1
            # Fetch the height and width of the sub-frame.
            sfm_width = all_results[i][3][1]
            sfm_height = all_results[i][3][0]
            # Create a dictionnary to store all relevant information in relation to the newly generated sub-frames.
            sfm_dico = {
                "license": 1,
                "file_name": all_results[i][0],
                "coco_url": "None",
                "height": sfm_height,
                "width": sfm_width,
                "date_captured": "None",
                "flickr_url": "None",
                "id": id_img
            }
            # Append the dictionnary to the previously created empty list.
            subframes.append(sfm_dico)

            # Checks whether or not a sub-frame has an associated annotation.
            if all_results[i][1]:
                
                bndboxes = all_results[i][1]

                for b in range(len(bndboxes)):
                    # Create the annotation ID by incrementing the associated counter.
                    id_ann += 1

                    bndbox = bndboxes[b]
                    
                    # Fetch the coordinates for the bounding box of each annotation for a given sub-frame in the COCO-style format. 
                    x_min = int(np.round(bndbox[0]))
                    y_min = int(np.round(bndbox[1]))
                    box_w = int(np.round(bndbox[2]))
                    box_h = int(np.round(bndbox[3]))
                    coco_box = [x_min, y_min, box_w, box_h]
                    # Fetch the area of the bounding box of each annotation.
                    area = box_w * box_h
                    # Fetch the label (i.e. class) of the target (i.e. object) contained within a given sub-frame.
                    label_id = all_results[i][2][b]
                    # Create a dictionnary to store all relevant information in relation to the COCO-style format annotation created for the newly generated sub-frames.
                    dico_ann = {
                            "segmentation": [[]],
                            "area": area,
                            "iscrowd": 0,
                            "image_id": id_img,
                            "bbox": coco_box,
                            "category_id": label_id,
                            "id": id_ann
                    }
                    # Append the dictionnary to the previously created empty list.
                    sfm_annotations.append(dico_ann)
        
        # Updating the annotation file for the original unsliced images.
        coco_dic["info"]["date_created"] = str(date.today())
        coco_dic["info"]["year"] = str(date.today().year)

        # Creating a new dictionnary to eventually export as the new COCO-style annotation file for the newly generated sub-frames.
        new_dic = {
            "info": coco_dic["info"],
            "licenses": coco_dic["licenses"],
            "images": subframes,                            # The previously created 'subframes' dictionnary.
            "annotations": sfm_annotations,                 # The previously created 'sfm_annotations' dictionnary.
            "categories": coco_dic["categories"]
        }

        # Exporting the newly created dictionnary as a JSON file.
        with open(output_f, "w") as outputfile:
            json.dump(new_dic, outputfile)

        if os.path.isfile(output_f) is True:
            print("File \'{}\' correctly saved at \'{}\'.\n".format(file_name, sfm_output_dir))
        else:
            print("An error occurs, file \'{}\' not found at \'{}\'.\n".format(file_name, sfm_output_dir))

    return return_var

#################################
### ORIGINAL 'softnms' METHOD ###
#################################
# def softnms(preds, Nt, tresh, method='linear', sigma=0.5):
#     '''
#     Function for applying the Non-Maximum Suppression 
#     (NMS) filter and Soft-NMS.

#     Parameters
#     ----------
#     preds : dict
#         Contains, at least, 3 keys:
#           - 'boxes' : list, containing a list of 
#             predicted bounding boxes,
#           - 'labels' : int, containing a list of labels
#             associated to the bboxes,
#           - 'scores' : float, containing confidence 
#             scores associated to predictions.
    
#     Nt : float
#         IoU treshold to apply.

#     tresh : float
#         Scores treshold.
    
#     method : str, optional
#         Choose between:
#           - 'nms' for classical non-soft NMS
#           - 'linear' for linear Soft-NMS
#           - 'gaussian' for gaussian Soft-NMS

#         In this third case, it is possible to
#         specify the variance, by changing 'sigma'.

#         Default: 'linear'
    
#     sigma : float, optional
#         Variance of gaussian's curve.

#         Default: 0.5


#     Returns
#     -------
#     dict
#         Contains the 3 initial keys including filtered
#         values.

#     Notes
#     -----
#     Based on:
#       - https://github.com/DocF/Soft-NMS/blob/master/soft_nms.py
#       - https://github.com/bharatsingh430/soft-nms/blob/master/lib/nms/cpu_nms.pyx

#     '''

#     boxes = np.array(preds['boxes'])
#     labels = np.array(preds['labels'])
#     scores = np.array(preds['scores'])

#     boxes_f = boxes.copy()
#     labels_f = labels.copy()
#     scores_f = scores.copy()

#     if len(boxes)==0:
#         return []

#     if boxes.dtype.kind == "i":
# 		    boxes = boxes.astype("float")

#     N = boxes.shape[0]
#     ind = np.array([np.arange(N)])
#     boxes = np.concatenate((boxes, ind.T), axis=1)

#     x1 = boxes[:,0]
#     y1 = boxes[:,1]
#     x2 = boxes[:,2]
#     y2 = boxes[:,3]

#     areas = (x2 - x1 + 1) * (y2 - y1 + 1)

#     for i in range(N):

#         # temporary variables
#         t_boxes = boxes[i, :].copy()
#         t_score = scores[i].copy()
#         t_area = areas[i].copy()
#         pos = i + 1

#         if i != N-1:
#             max_score = np.max(scores[pos:], axis=0)
#             max_pos = np.argmax(scores[pos:], axis=0)

#         else:
#             max_score = scores[-1]
#             max_pos = 0

#         if t_score < max_score:
#             boxes[i, :] = boxes[max_pos + i + 1, :]
#             boxes[max_pos + i + 1, :] = t_boxes
#             t_boxes = boxes[i,:]

#             scores[i] = scores[max_pos + i + 1]
#             scores[max_pos + i + 1] = t_score
#             t_score = scores[i]

#             areas[i] = areas[max_pos + i + 1]
#             areas[max_pos + i + 1] = t_area
#             t_area = areas[i]

#         # compute IoU
#         xx1 = np.maximum(boxes[i, 0], boxes[pos:, 0])
#         yy1 = np.maximum(boxes[i, 1], boxes[pos:, 1])
#         xx2 = np.minimum(boxes[i, 2], boxes[pos:, 2])
#         yy2 = np.minimum(boxes[i, 3], boxes[pos:, 3])


#         w = np.maximum(0.0, xx2 - xx1 + 1)
#         h = np.maximum(0.0, yy2 - yy1 + 1)

#         # IoU
#         iou = (w * h) / (areas[i] + areas[pos:] - (w * h))

#         # Weigthing
#         # ---
#         # 1 - Linear
#         if method == 'linear':
#             weight = np.ones(iou.shape)
#             weight[iou > Nt] = weight[iou > Nt] - iou[iou > Nt]
#         # 2 - Gaussian
#         elif method == 'gaussian':
#             weight = np.exp(-(iou*iou)/sigma)
#         # 3 - Original
#         elif method == 'nms':
#             weight = np.ones(iou.shape)
#             weight[iou > Nt] = 0

#         scores[pos:] = weight * scores[pos:]
  
#     idx = boxes[:,4][scores > tresh]
#     pick = idx.astype(int)

#     return {'boxes':boxes_f[pick],'labels':labels_f[pick],'scores':scores_f[pick]}
#################################
### MODIFIED 'softnms' METHOD ###
#################################
def softnms(preds, Nt, tresh, method='linear', sigma=0.5):
    """
    A Python method for applying the Non-Maximum Suppression (NMS) filter and soft-NMS.

    Parameters:
        preds (dict):
            A dictionnary containing at least the following three keys:
                'boxes' (list) --> A list containing all the bounding boxes associated with the predictions of the model.
                'label' (int) --> A list of labels (i.e. classes) associated with the predictions of the model.
                'scores' (float) --> A list of confidence scores associated with the predictions of the model.
        Nt (float):
            Index Over Union (or IoU) threshold to apply.
        tresh (float):
            The confidence scores threshold.
        method (str, optional):
            A NMS method to apply to the bounding boxes. If set to 'nms', the classic non-soft NMS method will be applied. If set to 'linear', the linear soft-NMS method will be applied. If set to 'gaussian', the gaussian soft-NMS (sigma = 0.5) method will be applied. If 'gaussian' is specified, it is possible to specify the variance by changing 'sigma'. (default: 'linear')    
        sigma (float, optional):
            The variance of the gaussian curve if the 'method' parameter is set to 'gaussian'. (default: 0.5)

    Returns:
        XXX (dict): TODO: Add a name to this dictionnary.
            A dictionnary containing the three initial keys including filtered values.

    Notes:
        This method is based on the following information:
            https://github.com/DocF/Soft-NMS/blob/master/soft_nms.py
            https://github.com/bharatsingh430/soft-nms/blob/master/lib/nms/cpu_nms.pyx
    """
    # Fetch the prediction's bounding boxes, labels (i.e. classes) and confidence scores.
    boxes = np.array(preds['boxes'])
    labels = np.array(preds['labels'])
    scores = np.array(preds['scores'])

    # Create shallow copies of the prediction's bounding boxes, labels (i.e. classes) and confidence scores.
    boxes_f = boxes.copy()
    labels_f = labels.copy()
    scores_f = scores.copy()

    if len(boxes) == 0:
        return []

    if boxes.dtype.kind == "i":
		    boxes = boxes.astype("float")

    N = boxes.shape[0]
    ind = np.array([np.arange(N)])
    boxes = np.concatenate((boxes, ind.T), axis=1)

    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    for i in range(N):

        # temporary variables
        t_boxes = boxes[i, :].copy()
        t_score = scores[i].copy()
        t_area = areas[i].copy()
        pos = i + 1

        if i != N-1:
            max_score = np.max(scores[pos:], axis=0)
            max_pos = np.argmax(scores[pos:], axis=0)

        else:
            max_score = scores[-1]
            max_pos = 0

        if t_score < max_score:
            boxes[i, :] = boxes[max_pos + i + 1, :]
            boxes[max_pos + i + 1, :] = t_boxes
            t_boxes = boxes[i,:]

            scores[i] = scores[max_pos + i + 1]
            scores[max_pos + i + 1] = t_score
            t_score = scores[i]

            areas[i] = areas[max_pos + i + 1]
            areas[max_pos + i + 1] = t_area
            t_area = areas[i]

        # compute IoU
        xx1 = np.maximum(boxes[i, 0], boxes[pos:, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[pos:, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[pos:, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[pos:, 3])


        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)

        # IoU
        iou = (w * h) / (areas[i] + areas[pos:] - (w * h))

        # Weigthing
        # ---
        # 1 - Linear
        if method == 'linear':
            weight = np.ones(iou.shape)
            weight[iou > Nt] = weight[iou > Nt] - iou[iou > Nt]
        # 2 - Gaussian
        elif method == 'gaussian':
            weight = np.exp(-(iou*iou)/sigma)
        # 3 - Original
        elif method == 'nms':
            weight = np.ones(iou.shape)
            weight[iou > Nt] = 0

        scores[pos:] = weight * scores[pos:]
  
    idx = boxes[:,4][scores > tresh]
    pick = idx.astype(int)

    return {'boxes':boxes_f[pick],'labels':labels_f[pick],'scores':scores_f[pick]}



# def overlap_merging(img_path, coco_path, width, height, output_path, 
#                     mmdet_model, nms_method='nms', IoU=0.3, sc_tresh=0.5):
#     '''
#     Function to perform inference and stitching of the 
#     resulting predictions on a big-sized image cut into 
#     sub-frames with 50% overlap, in order to obtain the 
#     original image and its predictions.

#     To be used with the Subframes class.

#     Parameters
#     ----------
#     img_path : str
#         Path to the image.

#     coco_path : str
#         Path to the COCO-annotation type (JSON).
    
#     width : int
#         Width of the sub-frames.

#     height : int
#         Height of the sub-frames.
    
#     output_path : str
#         Sub-frames saving path.
    
#     mmdet_model : object
#         Model built from mmdet.apis.init_detector

#     nms_method : str, optional
#         NMS method to apply to bounding boxes.
#         Choose between:
#             - 'nms' : for classic NMS
#             - 'linear' : for linear soft-NMS
#             - 'gaussian' : for gaussian soft-NMS (sigma=0.5)
#         Default : 'nms'

#     IoU : float, optional
#         (Soft-)NMS treshold.
#         Default : 0.3
    
#     sc_tresh : float, optional
#         Scores treshold to applay on predictions.
#         Default : 0.5

#     Returns
#     -------
#     dict
#         Resulting predictions containing 3 keys:
#           - 'boxes' (2D array)
#           - 'labels' (1D array)
#           - 'scores' (1D array)
#     '''

#     t_i = time.time()

#     with open(coco_path,'r') as json_file:
#         coco_dic = json.load(json_file)

#     cls_names = []
#     for cls in coco_dic['categories']:
#         cls_names.append(cls['name'])

#     # PIL image
#     pil_img = Image.open(img_path)

#     # Infos
#     name = basename(img_path)

#     w_sub = int(pil_img.width/width)
#     h_sub = int(pil_img.height/height)

#     if pil_img.width % width != 0:
#         w_sub += 1
#     if pil_img.height % height != 0:
#         h_sub += 1


#     # Export folder
#     save_path = os.path.join(output_path,name)
#     if os.path.exists(save_path) is not True:
#         os.mkdir(save_path)

#     # Get annos
#     for image in coco_dic['images']:
#         if image['file_name'] == name:
#             img_id = image['id']

#     boxes = []
#     labels = []
#     for ann in coco_dic['annotations']:
#         if ann['image_id'] == img_id:
#             boxes.append(ann['bbox'])
#             labels.append(ann['category_id'])

#     gt = {'boxes':boxes, 'labels':labels}

#     # Subframes class instantiation
#     sub_img = Subframes(name, pil_img, gt, width, height)

#     # overlap
#     results = sub_img.getlist(overlap=True)

#     sub_img.save(results, save_path, object_only=False)

#     os.chdir(save_path)
#     files = os.listdir(save_path)
#     files.sort(key=os.path.getctime)

#     files_2D = np.reshape(np.array(files), (2*h_sub-1,2*w_sub-1))

#     # Initializations
#     w_offset = 0
#     h_offset = 0
#     global_boxes = []
#     global_labels = []
#     global_scores = []

#     for y in range(files_2D.shape[0]):
#         for x in range(files_2D.shape[1]):

#             # Image
#             image = files_2D[y,x]

#             # Predictions
#             predictions = inference_detector(mmdet_model, image)

#             # adapt
#             boxes = []
#             labels = []
#             scores = []
#             for n_class in range(len(cls_names)):
#                 for n_box in range(len(predictions[n_class])):
#                     box = list(predictions[n_class][n_box][:4])
#                     score = predictions[n_class][n_box][4]
#                     boxes.append(box)
#                     labels.append(n_class+1)
#                     scores.append(score)

#             predictions = {'boxes': boxes, 'labels': labels, 'scores': scores}

#             # Put into a global frame
#             i = 0
#             for box in predictions['boxes']:

#                 new_box = [box[0] + w_offset,
#                           box[1] + h_offset,
#                           box[2] + w_offset,
#                           box[3] + h_offset]            
                
#                 global_boxes.append(new_box)
#                 global_labels.append(predictions['labels'][i])
#                 global_scores.append(predictions['scores'][i])

#                 i += 1

#             w_offset += 0.5*width

#         w_offset = 0
#         h_offset += 0.5*height

#     global_preds = {
#         'boxes':global_boxes, 
#         'labels':global_labels,
#         'scores':global_scores
#         }

#     # Soft-NMS 
#     global_preds = softnms(global_preds, IoU, sc_tresh, method=nms_method)

#     t_f = time.time()

#     shutil.rmtree(save_path)

#     return global_preds

###########################################
### MODIFIED 'overlap_merging' FUNCTION ###
###########################################
def overlap_merging(img_path, img_anno_path, sfm_width, sfm_height, output_dir, mmdet_model, nms_method='nms', IoU=0.3, sc_tresh=0.5):
    """
    A Python method to perform inference and stitching of the resulting predictions on a big-sized image cut into sub-frames with 50 % overlap, in order to obtain the original image and its predictions.

    To be used with the Subframes class.

    Parameters:
        img_path (str):
            The path to an original unsliced image.
        img_anno_path (str):
            The path to the COCO-style annotation file associated with the original unsliced images.
        sfm_width (int):
            The width of the sub-frames.
        sfm_height (int):
            The height of the sub-frames.
        output_dir (str):
            The directory in which to save the sub-frames.
        mmdet_model (object):
            The MMDetection model built from the 'mmdet.apis.init_detector()' Python method.
        nms_method (str, optional):
            A NMS method to apply to the bounding boxes. If set to 'nms', the classic non-soft NMS method will be applied. If set to 'linear', the linear soft-NMS method will be applied. If set to 'gaussian', the gaussian soft-NMS (sigma = 0.5) method will be applied. If 'gaussian' is specified, it is possible to specify the variance by changing 'sigma'. (default: 'nms')
        IoU (float, optional):
            The (soft-)NMS threshold. (default: 0.3)
        sc_tresh (float, optional):
            The scores threshold to apply on the predictions. (default: 0.5)

    Returns:
        global_preds (dict):
            The model's resulting predictions containing the following three keys: 'boxes' (2D array), 'labels' (1D array) and 'scores' (1D array)
    """
    # Fetch the initial time.
    t_i = time.time()

    with open(file = img_anno_path, mode = "r") as json_file:
        coco_dic = json.load(fp = json_file)

    # Fetch the names of the classes and add them to a list.
    cls_names = []
    for cls in coco_dic["categories"]:
        cls_names.append(cls["name"])

    # Open an original unsliced image with the PILLOW Python library.
    img_pillow = Image.open(img_path)

    # Infos
    name = basename(img_path)

    #TODO: The following 'w_sub' and h_sub' calculations are redundant with the ones coded in the 'Subframes.__init__' Python method.
    w_sub = int(img_pillow.width / sfm_width)
    h_sub = int(img_pillow.height / sfm_height)

    if img_pillow.width % sfm_width != 0:
        w_sub += 1
    if img_pillow.height % sfm_height != 0:
        h_sub += 1
    # print("w_sub: {} ; h_sub: {}".format(w_sub, h_sub))


    # Export folder
    save_path = os.path.join(output_dir, name)
    if os.path.exists(path = save_path) is not True:
        os.mkdir(path = save_path)

    # Get annos
    for image in coco_dic["images"]:
        if image["file_name"] == name:
            img_id = image["id"]

    anno_labels = []
    anno_bboxes = []
    for ann in coco_dic["annotations"]:
        if ann["image_id"] == img_id:
            anno_labels.append(ann["category_id"])
            anno_bboxes.append(ann["bbox"])

    gt = {
        "anno_labels": anno_labels,
        "anno_bboxes": anno_bboxes
    }

    # Subframes class instantiation
    sub_img = Subframes(
        img_name = name,
        img_pillow = img_pillow,
        img_target = gt,
        sfm_width = sfm_width,
        sfm_height = sfm_height,
        sfm_strict_size = False
    )

    # print("img_width: {}; img_height: {}".format(img_pillow.width, img_pillow.height))
    # print("x_sub: {}; y_sub: {}".format(sub_img.x_sub, sub_img.y_sub))
    # print("w_sub: {}; h_sub: {}".format(w_sub, h_sub))

    #TODO: To be eventually removed.
    if int(w_sub) == int(sub_img.x_sub):
        if int(h_sub) == int(sub_img.y_sub):
            print("img_id: {} --- True".format(img_id))
        else:
            print("img_id: {} --- THERE IS A PROBLEM".format(img_id))
    else:
        print("img_id: {} --- THERE IS A PROBLEM".format(img_id))



    # overlap
    results = sub_img.getlist(
        sfm_overlap = True
    )

    sub_img.save(
        results = results,
        sfm_output_dir = save_path,
        sfm_object_only = False
    )

    os.chdir(save_path)
    files = os.listdir(save_path)
    files.sort(key=os.path.getctime)

    img_array = np.array(object = files)
    # print(img_array, img_array.shape)
    w_sub = 0
    h_sub = 0
    for item in img_array:
        row = int(item.split(".")[0].split("_")[-2][1:])
        col = int(item.split(".")[0].split("_")[-1][1:])
        if row > h_sub:
            h_sub = row
        if col > w_sub:
            w_sub = col
    files_2D = np.reshape(
        a = np.array(object = files),
        newshape = (h_sub, w_sub)
        # newshape = (2*h_sub-1, 2*w_sub-1)
    )

    # Initializations
    w_offset = 0
    h_offset = 0
    global_boxes = []
    global_labels = []
    global_scores = []

    # Parse through all rows of the 'files_2D' numpy array.
    for y in range(files_2D.shape[0]):
        # Parse through all columnms of the 'files_2D' numpy array.
        for x in range(files_2D.shape[1]):
            # Fetch one of the sub-frame for a given original unsliced image.
            image = files_2D[y, x]
            # For each sub-frames, inference image(s) with the detector to get predictions.
            predictions = inference_detector(
                model = mmdet_model,
                imgs = image
            )

            # Create empty lists to store a prediction's associated bounding box, label and confidence score.
            boxes = []
            labels = []
            scores = []
            # For each classes...
            for n_class in range(len(cls_names)):
                # Fetch the associated information for each prediction made by the model during the inference process.
                for n_box in range(len(predictions[n_class])):
                    # Fetch the prediction's bounding box.
                    bbox = list(predictions[n_class][n_box][:4])
                    # Fetch the prediction's confidence score.
                    score = predictions[n_class][n_box][4]
                    # Add the bounding box, label and confidence score associated to a prediction in empty lists to store the information.
                    boxes.append(bbox)
                    labels.append(n_class + 1)
                    scores.append(score)
            # Add the prediction's associated information to the 'predictions' dictionnary.
            predictions = {
                "boxes": boxes,
                "labels": labels,
                "scores": scores
            }

            # Put into a global frame
            i = 0 #TODO: Why is there a 'i' counter? Is doesn't seem to be used elsewhere.
            for bbox in predictions["boxes"]:

                new_bbox = [
                    bbox[0] + w_offset,
                    bbox[1] + h_offset,
                    bbox[2] + w_offset,
                    bbox[3] + h_offset
                ]
                
                global_boxes.append(new_bbox)
                global_labels.append(predictions["labels"][i])
                global_scores.append(predictions["scores"][i])

                i += 1

            w_offset += 0.5 * sfm_width

        w_offset = 0
        h_offset += 0.5 * sfm_height

    global_preds = {
        "boxes": global_boxes, 
        "labels": global_labels,
        "scores": global_scores
    }

    # Soft-NMS TODO: Verify this function.
    global_preds = softnms(
        preds = global_preds,
        Nt = IoU,
        tresh = sc_tresh,
        method = nms_method
    )

    t_f = time.time()

    shutil.rmtree(save_path)

    return global_preds


def compute_IoU(box_A, box_B):
    '''
    Function to compute Intersect-over-Union (IoU)
    between two lists of boxes.

    Arguments
    ---------
    box_A : list
        (dim=4) 2 points-style (upper-left (x1,y1) and 
        bottom-right (x2,y2))
    box_B : list
        (dim=4) 2 points-style (upper-left (x1,y1) and 
        bottom-right (x2,y2))
    
    Returns
    -------
    IoU : float
    '''

    xA = max(box_A[0],box_B[0])	
    yA = max(box_A[1],box_B[1])	
    xB = min(box_A[2],box_B[2])	
    yB = min(box_A[3],box_B[3])	

    area = max(0, xB - xA +1) * max(0, yB - yA +1)	

    area_A = (box_A[2] - box_A[0] + 1) * (box_A[3] - box_A[1] + 1)	
    area_B = (box_B[2] - box_B[0] + 1) * (box_B[3] - box_B[1] + 1)	

    IoU = area / float(area_A + area_B - area)	

    return IoU
    
def match(predictions, img_name, coco_path, IoU_tresh, with_scores=False):	
    '''	
    Function used to match the ground-truth bounding boxes 	
    to predicted ones. The outputs are used to construct a	
    confusion matrix.	

    Parameters	
    ----------	
    predictions : pd.DataFrame
        Pandas DataFrame with header :
        |'Image'|'x1'|'y1'|'x2'|'y2'|'Label'|'Score'|

    img_name : str	
        Image's name.	

    coco_path : str	
        Path to the COCO-style annotation file in JSON format.

    IoU_tresh : float	
        IoU treshold.		

    with_scores : bool, optional
        If True, a column with scores is concatenated to matching.
        Default : False

    Returns	
    -------	
    2D list	
        Matching between gt and predicted bbox
    '''	

    # Open JSON file with ground truth	
    with open(coco_path,'r') as json_file:	
        coco_dic = json.load(json_file)	

    # Get images id and ground truth infos
    id_img = [
        i['id']
        for i in coco_dic['images']
        if i['file_name']==img_name][0]
    
    gt_boxes = [
        [
            a['bbox'][0],
            a['bbox'][1],
            a['bbox'][0]+a['bbox'][2],
            a['bbox'][1]+a['bbox'][3]
        ]
        for a in coco_dic['annotations']
        if a['image_id']==id_img
    ]

    gt_labels = [a['category_id'] for a in coco_dic['annotations']
                if a['image_id']==id_img]

    gt = {	
        'boxes': gt_boxes,	
        'labels': gt_labels	
    }	

    # Get predictions
    preds = {
            'boxes': [[p['x1'],p['y1'],p['x2'],p['y2']] for i,p in predictions.iterrows()],
            'labels': [p['Label'] for i,p in predictions.iterrows()],
            'scores': [p['Score'] for i,p in predictions.iterrows()],
            'indices': [i for i,p in predictions.iterrows()]
        }

    p_boxes = preds['boxes']

	
    res = []	
    i = 0	

    for gt_box, gt_label in zip(gt['boxes'], gt['labels']):	

        id_gt = i	
        p_iou = []	

        # No detection ? => FN	
        if len(p_boxes)==0:	
            res.append([i, gt_label, int(0), int(0), int(0), int(0), None])	
            continue	

        for p_box in preds['boxes']:	
            	
            IoU = compute_IoU(gt_box, p_box)

            p_iou.append(IoU)	

        # Maximum correspondance with gt
        p_iou_max = float(max(p_iou))	
        index = p_iou.index(p_iou_max)	

        # Iou to low ? FN	
        if p_iou_max < IoU_tresh and p_iou_max > 0:	
            p_label = int(0)
            p_score = preds['scores'][index]	
        elif p_iou_max == 0:	
            p_label = int(0)
            p_score = 0
        else:	
            p_label = int(preds['labels'][index])	
            p_score = preds['scores'][index]

        # Index of pandas.DataFrame
        pandas_idx = preds['indices'][index]

        res.append([id_gt, gt_label, index, p_label, p_iou_max, p_score, pandas_idx])

        # Avoid double count
        del preds['boxes'][index]
        del preds['labels'][index]
        del preds['scores'][index]
        del preds['indices'][index]

        i += 1	

    # p_boxes without correspondance = FPs	
    p_not_used = list(range(len(preds['boxes'])))

    for k in p_not_used:		
        label = int(preds['labels'][k])	
        score = preds['scores'][k]
        res.append([i, int(0), k, label, int(0), score, preds['indices'][k]])	
        i += 1	

    matching = np.array(res)

    if with_scores is True:
        matching = np.delete(matching, [0,2], 1)	# with IoU
        # matching = np.delete(matching, [0,2,4], 1)	# without
    else:
        matching = np.delete(matching, [0,2,5], 1)	 # with IoU
        # matching = np.delete(matching, [0,2,4,5], 1) # without

    return matching