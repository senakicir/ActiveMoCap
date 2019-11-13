import numpy as np
import torch
from math import ceil
from helpers import numpy_to_tuples

def find_bbox_bounds(bbox):
    bounds =   {"min_x" : int(bbox[0] - bbox[2]/2),
                "max_x" : int(bbox[0] + bbox[2]/2),
                "min_y" : int(bbox[1] - bbox[3]/2),
                "max_y" : int(bbox[1] + bbox[3]/2)
    }
    return bounds

def find_bbox_from_pose(pose_2d, margin):
    bounds = {}
    bounds["min_x"] = torch.min(pose_2d[0,:])
    bounds["min_y"] = torch.min(pose_2d[1,:])
    bounds["max_x"] = torch.max(pose_2d[0,:])
    bounds["max_y"] = torch.max(pose_2d[1,:])

    width = int(bounds["max_x"] - bounds["min_x"])
    height = int(bounds["max_y"] - bounds["min_y"])
    length = max(width, height)
    new_length = int(length + length*margin)
    center_x = ceil((bounds["max_x"] + bounds["min_x"])/2)
    center_y = ceil((bounds["max_y"] + bounds["min_y"])/2)
    bbox = [center_x, center_y, new_length, new_length]

    return bounds, bbox

def find_rectangular_bounds_from_pose(pose_2d, margin):
    bounds = {}
    bounds["min_x"] = torch.min(pose_2d[0,:])
    bounds["min_y"] = torch.min(pose_2d[1,:])
    bounds["max_x"] = torch.max(pose_2d[0,:])
    bounds["max_y"] = torch.max(pose_2d[1,:])

    width = int(bounds["max_x"] - bounds["min_x"])
    height = int(bounds["max_y"] - bounds["min_y"])
    new_width =  int(width + width*margin)
    new_height = int(height + height*margin)
    center_x = ceil((bounds["max_x"] + bounds["min_x"])/2)
    center_y = ceil((bounds["max_y"] + bounds["min_y"])/2)
    bbox = [center_x, center_y, new_width, new_height]
    bounds = find_bbox_bounds(bbox)
    return bounds

def is_pose_in_image(pose_2d, image_size_x, image_size_y):
    bounds = find_rectangular_bounds_from_pose(pose_2d, margin= -0.5)
    in_image = True
    if bounds["min_x"] < 0 or  bounds["min_y"] < 0:
        in_image = False
    if bounds["max_x"] > image_size_x or bounds["max_y"] > image_size_y:
        in_image = False
    return in_image 

class Basic_Crop(object):
    def __init__(self, margin):
        self.bbox_with_margin = None
        self.margin = margin #sth like 0.5 would do
        self.image_bounds = None
        self.can_crop = True
        self.scales = [0.5, 0.75, 1]

    def copy_cropping_tool(self):
        new_basic_crop = Basic_Crop(self.margin)
        new_basic_crop.bbox_with_margin = self.bbox_with_margin
        new_basic_crop.image_bounds = self.image_bounds
        new_basic_crop.can_crop = self.can_crop
        return new_basic_crop

    def update_bbox(self, bbox):
        if self.can_crop:
            new_width = int(bbox[2] + bbox[2]*self.margin)
            new_height = int(bbox[3] + bbox[3]*self.margin)
            if new_width %2 != 0:
                new_width += 1
            if new_height %2 != 0:
                new_height += 1        
            self.bbox_with_margin = [bbox[0], bbox[1], new_width, new_height]

    def find_bbox_from_pose(self, pose_2d):
        return find_bbox_from_pose(pose_2d, self.margin)

    def crop_image(self, image):
        if self.can_crop:
            image_size_x, image_size_y = image.shape[1], image.shape[0]
            crop_size_x, crop_size_y = self.bbox_with_margin[2], self.bbox_with_margin[3]

            bounds = find_bbox_bounds( self.bbox_with_margin)
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
            if img_min_x == 0:
                self.image_bounds[0] = bounds["min_x"]
            if  img_min_y == 0:
                self.image_bounds[1] = bounds["min_y"]

            crop_frame = np.zeros(tuple((crop_size_y, crop_size_x, image.shape[2])), dtype=image.dtype)
            crop_frame[crop_min_y:crop_max_y, crop_min_x:crop_max_x, :] = image[img_min_y:img_max_y, img_min_x:img_max_x, :]
        else:
            crop_frame = image
        return crop_frame

    def base_bbox(self, size_x, size_y):
        return [int(size_x/2), int(size_y/2), int(size_x), int(size_y)]

    def disable_cropping(self):
        self.can_crop = False
        self.scales = [1, 1.25, 2, 2.5]
    
    def enable_cropping(self):
        self.can_crop = True
        self.scales = [0.5, 0.75, 1]

    def crop_pose(self, pose_2d_input):
        pose_2d = pose_2d_input.clone()
        if self.can_crop:
            pose_2d[0,:] = pose_2d[0,:] - self.image_bounds[0]
            pose_2d[1,:] = pose_2d[1,:] - self.image_bounds[1]
        return pose_2d

    def uncrop_pose(self, pose_2d_input):
        pose_2d = pose_2d_input.clone()
        if self.can_crop:
            pose_2d[0,:] = pose_2d[0,:] + self.image_bounds[0]
            pose_2d[1,:] = pose_2d[1,:] + self.image_bounds[1]
        return pose_2d

    def uncrop_heatmap(self, heatmap, image_size_x, image_size_y):
        new_heatmap = np.zeros(heatmap.shape)
        max_crop_x = heatmap.shape[1] +  self.image_bounds[0]
        if max_crop_x > image_size_x:
            max_crop_x = image_size_x
        max_crop_y = heatmap.shape[2] +  self.image_bounds[1]
        if max_crop_y > image_size_y:
            max_crop_y = image_size_y        
        if self.can_crop:
            new_heatmap[:, self.image_bounds[0]:max_crop_x, self.image_bounds[1]:max_crop_y] = heatmap
        return new_heatmap

    def return_bbox_coord(self):
        bounds = find_bbox_bounds(self.bbox_with_margin)

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
  