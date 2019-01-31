import numpy as np
from math import ceil
from square_bounding_box import *
from helpers import SIZE_X, SIZE_Y, numpy_to_tuples
import pdb

crop_alpha = 0.5
STABLE_FRAME = 20

class Crop:

    def __init__(self, bbox_init = [0,0,SIZE_X,SIZE_Y], loop_mode = 0):
        self.old_bbox = bbox_init
        self.bbox = bbox_init
        self.image_bounds = [0,0]
        self.scales= [1]
        self.bounding_box_calculator = BoundingBox(3)
        self.bounding_box_margin = 3
        self.unstable = True

        if loop_mode != 0:
            self.unstable = False
            global crop_alpha
            crop_alpha = 1
            self.bbox = [SIZE_X//2-250,SIZE_Y//2-250,500,500]
            self.bounding_box_margin = 3

    def crop(self, image, linecount):
        pdb.set_trace()
        if linecount >= STABLE_FRAME:
            self.unstable = False

        if self.unstable:
            print("unstable, will not crop")
            return image, [1]
        
        self.update_bbox_margin(1)

        orig_image_width = image.shape[1]
        orig_image_height = image.shape[0]

        self.bbox[0] = int(crop_alpha*self.bbox[0] + (1-crop_alpha)*self.old_bbox[0])
        self.bbox[1] = int(crop_alpha*self.bbox[1] + (1-crop_alpha)*self.old_bbox[1])
        self.bbox[2] = int(crop_alpha*self.bbox[2] + (1-crop_alpha)*self.old_bbox[2])
        self.bbox[3] = int(crop_alpha*self.bbox[3] + (1-crop_alpha)*self.old_bbox[3])        

        crop_shape = (self.bbox[3], self.bbox[2], image.shape[2])
        crop_frame = np.zeros(tuple(crop_shape), dtype=image.dtype)
        if not (self.bbox[0] > image.shape[1] or self.bbox[0] + self.bbox[2] < 0 or
                        self.bbox[1] > image.shape[0] or self.bbox[1] + self.bbox[3] < 0):
            img_min_x = max(0, self.bbox[0])
            img_max_x = min(self.bbox[0] + self.bbox[2], orig_image_width)
            img_min_y = max(0, self.bbox[1])
            img_max_y = min(self.bbox[1] + self.bbox[3], orig_image_height)

            crop_min_x = img_min_x-self.bbox[0]
            crop_max_x = img_max_x-self.bbox[0]
            crop_min_y = img_min_y-self.bbox[1]
            crop_max_y = img_max_y-self.bbox[1]

            if (crop_min_x > 0):
                min_bound_x = -crop_min_x
            else:
                min_bound_x = img_min_x

            if (crop_min_y > 0):
                min_bound_y = -crop_min_y
            else:
                min_bound_y = img_min_y
            
            self.image_bounds = [min_bound_x, min_bound_y]
        else:
            return None

        crop_frame[crop_min_y:crop_max_y, crop_min_x:crop_max_x, :] = image[img_min_y:img_max_y,
                                                                            img_min_x:img_max_x,
                                                                            :]

        self.scales = [0.75, 1, 1.25, 1.5,]

        return crop_frame, self.scales

    def uncrop_pose(self, pose_2d):
        if self.unstable:
            return pose_2d

        pose_2d[0,:] = pose_2d[0,:] + self.image_bounds[0]
        pose_2d[1,:] = pose_2d[1,:] + self.image_bounds[1]
        self.update_bbox(numpy_to_tuples(pose_2d))

        return pose_2d
    
    def update_bbox(self, pose_2d):
        new_bbox = self.bounding_box_calculator.get_bounding_box(pose_2d)
        self.old_bbox = self.bbox 
        self.bbox = new_bbox
    
    def update_bbox_margin(self, margin):
        self.bounding_box_calculator.update_margin(margin)
        self.bounding_box_margin = margin

    def crop_pose(self, pose_2d):
        pose_2d[0,:] = pose_2d[0,:] - self.image_bounds[0]
        pose_2d[1,:] = pose_2d[1,:] - self.image_bounds[1]
        return pose_2d
    