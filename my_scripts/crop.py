import numpy as np
import torch
from math import ceil
from helpers import numpy_to_tuples
import pdb

class Basic_Crop(object):
    def __init__(self, margin):
        self.bbox_with_margin = None
        self.margin = margin #sth like 0.5 would do
        self.image_bounds = None

    def copy_cropping_tool(self):
        new_basic_crop = Basic_Crop()
        new_basic_crop.bbox_with_margin = self.bbox_with_margin
        new_basic_crop.image_bounds = self.image_bounds

    def update_bbox(self, bbox):
        length =  max(bbox [2], bbox [3])
        new_width = int(length + length*self.margin)
        new_height = int(length + length*self.margin)
    
        self.bbox_with_margin = [bbox[0], bbox[1], new_width, new_height]

    def crop_image(self, image):
        image_size_x, image_size_y = image.shape[1], image.shape[0]
        crop_size_x, crop_size_y = self.bbox_with_margin[2], self.bbox_with_margin[3]

        bounds = self.find_bbox_bounds()
        img_min_x = max(0, bounds["min_x"])
        crop_min_x = 0
        if img_min_x == 0:
            crop_min_x = -bounds["min_x"]
        
        img_min_y = max(0, bounds["min_y"])
        crop_min_y = 0
        if img_min_y == 0:
            crop_min_y = -bounds["min_y"]
        
        img_max_x = min(image_size_x, bounds["max_x"])
        crop_max_x = crop_size_x
        if img_max_x == image_size_x:
            crop_max_x = image_size_x - bounds["min_x"]

        img_max_y = min(image_size_y, bounds["max_y"])
        crop_max_y = crop_size_y
        if img_max_y == image_size_y:
            crop_max_y = image_size_y - bounds["min_y"]
        

        self.image_bounds = [img_min_x, img_min_y]

        crop_frame = np.zeros(tuple((crop_size_y, crop_size_x, image.shape[2])), dtype=image.dtype)
        crop_frame[crop_min_y:crop_max_y, crop_min_x:crop_max_x, :] = image[img_min_y:img_max_y, img_min_x:img_max_x, :]
        return crop_frame

    def find_bbox_from_pose(self, pose_2d):
        bounds = {}
        bounds["min_x"] = torch.min(pose_2d[0,:])
        bounds["min_y"] = torch.min(pose_2d[1,:])
        bounds["max_x"] = torch.max(pose_2d[0,:])
        bounds["max_y"] = torch.max(pose_2d[1,:])

        width = int(bounds["max_x"] - bounds["min_x"])
        height = int(bounds["max_y"] - bounds["min_y"])
        length = max(width+width*self.margin, height+height*self.margin)
        center_x = int((bounds["max_x"] + bounds["min_x"])/2)
        center_y = int((bounds["max_y"] + bounds["min_y"])/2)
        bbox = [center_x, center_y, length, length]
        return bounds, bbox

    def is_pose_in_image(self, pose_2d, image):
        image_size_x, image_size_y = image.shape[1], image.shape[0]

        bounds = self.find_bbox_from_pose(pose_2d)
        in_image = True
        if bounds["min_x"] < 0 or  bounds["min_y"] < 0:
            in_image = False
        if bounds["max_x"] > image_size_x or bounds["max_y"] > image_size_y:
            in_image = False
        return in_image

    def base_bbox(self, size_x, size_y):
        return [int(size_x/2), int(size_y/2), int(size_x), int(size_y)]
        

    def find_bbox_bounds(self):
        bounds =    {"min_x" : int(self.bbox_with_margin[0] - self.bbox_with_margin[2]/2),
                    "max_x" : int(self.bbox_with_margin[0] + self.bbox_with_margin[2]/2),
                    "min_y" : int(self.bbox_with_margin[1] - self.bbox_with_margin[3]/2),
                    "max_y" : int(self.bbox_with_margin[1] + self.bbox_with_margin[3]/2)
        }
        return bounds

    def crop_pose(self, pose_2d_input):
        pose_2d = pose_2d_input.clone()
        pose_2d[0,:] = pose_2d[0,:] - self.image_bounds[0]
        pose_2d[1,:] = pose_2d[1,:] - self.image_bounds[1]
        return pose_2d

    def uncrop_pose(self, pose_2d_input):
        pose_2d = pose_2d_input.clone()
        pose_2d[0,:] = pose_2d[0,:] + self.image_bounds[0]
        pose_2d[1,:] = pose_2d[1,:] + self.image_bounds[1]
        return pose_2d

    def return_bbox_coord(self):
        bounds = self.find_bbox_bounds()

        bbox_corners_x = [  bounds["min_x"], 
                            bounds["max_x"], 
                            bounds["max_x"],  
                            bounds["min_x"], 
                            bounds["min_x"]]
        bbox_corners_y = [  bounds["max_y"],
                            bounds["max_y"], 
                            bounds["min_y"],  
                            bounds["min_y"], 
                            bounds["max_y"]] 
        return bbox_corners_x, bbox_corners_y
  