import numpy as np


def bounding_box(pose_array, model_type, margin):
    """
    Computes the bounding box around the joints of a person. The joints are given as an array.
    The array is of size (nb_joints x 3). Each row contains the x, y coordinates and
    the probability of the joint.
    A margin is added to the bounding box. The margin is computed as a percentage of each side
    of the tight bounding box. The margin is added in both directions such that the center of gravity
    of the joints remains in the middle.

    :param pose_array: Array containing the joints (nb_joints x 3)
    :param model_type: The type of skeleton that it represents ('MPI' or 'COCO')
    :param margin: Percentage of the side to be added
    :return: Bounding box with margin around the joints
    """
    masked_xy_array = np.ma.masked_equal(pose_array[..., :2], 0.0, copy=False)
    min_values = masked_xy_array.min(axis=0)
    max_values = masked_xy_array.max(axis=0)

    x = min_values[0]
    y = min_values[1]
    width = max_values[0] - min_values[0]
    height = max_values[1] - min_values[1]

    # Check if the legs exist
    if np.any(pose_array[10] > 0) or np.any(pose_array[13] > 0):
        # Feet exist so do nothing
        pass
    elif np.any(pose_array[9] > 0) or np.any(pose_array[12] > 0):
        # Knees exist. Add a third of the height
        height *= 4 / 3
    elif np.any(pose_array[8] > 0) or np.any(pose_array[11] > 0):
        # Hip exists so double the height
        height *= 2
    elif model_type == 'MPI' and np.any(pose_array[14] > 0):
        height *= 2
    else:
        # We only have the upper body so maybe times 2.5 of the height ?!?
        height *= 2.5

    # Add the margin as a percentage of the height and width
    height_margin = height * margin
    y -= height_margin / 2
    height *= 1 + margin

    width_margin = width * margin
    x -= width_margin / 2
    width *= 1 + margin

    return_values = [x, y, width, height]

    return return_values


def same_margin_bounding_box(pose_array, model_type, margin):
    """
    Computes the bounding box around the joints of a person. The joints are given as an array.
    The array is of size (nb_joints x 3). Each row contains the x, y coordinates and
    the probability of the joint.
    A margin is added to the bounding box. The margin is computed as a percentage of the biggest side
    of the tight bounding box. The margin is then added in both directions such that the center of gravity
    of the joints remains in the middle.

    :param pose_array: Array containing the joints (nb_joints x 3)
    :param model_type: The type of skeleton that it represents ('MPI' or 'COCO')
    :param margin: Percentage of the biggest side to be added
    :return: Bounding box with margin around the joints
    """
    masked_xy_array = np.ma.masked_equal(pose_array[..., :2], 0.0, copy=False)
    min_values = masked_xy_array.min(axis=0)
    max_values = masked_xy_array.max(axis=0)

    x = min_values[0]
    y = min_values[1]
    width = max_values[0] - min_values[0]
    height = max_values[1] - min_values[1]

    # Check if the legs exist    
    if (pose_array[10] > 0).any() or (pose_array[13] > 0).any():
        # Feet exist so do nothing
        pass
    elif (pose_array[9] > 0).any() or (pose_array[12] > 0).any():
        # Knees exist. Add a third of the height
        height *= 4 / 3
    elif (pose_array[8] > 0).any() or (pose_array[11] > 0).any():
        # Hip exists so double the height
        height *= 2
    elif model_type == 'MPI' and (pose_array[14] > 0).any():
        height *= 2
    else:
        # We only have the upper body so maybe times 2.5 of the height ?!?
        height *= 2.5

    # Add the margin as a percentage of the height and width
    margin = max(2, max(height, width) * margin)

    y -= margin / 2
    height += margin
    x -= margin / 2
    width += margin

    return_values = [x, y, width, height]

    return return_values


def crop_input(input_orig, bbox, square, pad, pad_value, channel_indices):
    """
    Crops the given input given the coordinates of the bounding box.
    A list containing the used channels has to be given. All other channels are removed.
    The bounding box is in the order [x, y, width, height]. The bounding box can be forced to be square.
    The output can also be forced to be square. In the case the bounding box is not square the extra space is filled
    with pad_value.

    :param input_orig: The input to crop
    :param bbox: The bounding box determining where to crop. Format [x, y, width, height].
    :param square: If True the cropped region is a square.
    :param pad: If True the returned cropped shape is a square. If the original bounding box is not square the extra region from the cropped array is filled with the given value.
    :param pad_value: Value with which to pad the output.
    :param channel_indices: List of channel indices. Only those channels will be returned.
    :return: The cropped region of the heatmap.
    """
    if pad:
        square = True
    width = input_orig.shape[1]
    height = input_orig.shape[0]

    filtered_input = input_orig.copy()
    filtered_input = filtered_input[..., np.array(channel_indices)]

    try:
        bbox = [int(elem) for elem in bbox]
    except:
        bbox = [0, 0, 0, 0]

    if bbox[3] == 0:
        bbox[3] = 1
    if bbox[2] == 0:
        bbox[2] = 1

    if square and not pad:
        if bbox[3] > bbox[2]:
            diff = bbox[3] - bbox[2]
            bbox[2] = bbox[3]
            bbox[0] -= diff // 2
        elif bbox[2] > bbox[3]:
            bbox[3] = bbox[2]
            diff = bbox[2] - bbox[3]
            bbox[1] -= diff // 2

    crop_shape = (bbox[3], bbox[2], filtered_input.shape[2])
    crop_data = np.full(tuple(crop_shape), pad_value, dtype=filtered_input.dtype)

    if not (bbox[0] > width or bbox[0] + bbox[2] < 0 or
            bbox[1] > height or bbox[1] + bbox[3] < 0):
        img_min_x = max(0, bbox[0])
        img_max_x = min(bbox[0] + bbox[2], width)
        img_min_y = max(0, bbox[1])
        img_max_y = min(bbox[1] + bbox[3], height)

        crop_min_x = img_min_x - bbox[0]
        crop_max_x = img_max_x - bbox[0]
        crop_min_y = img_min_y - bbox[1]
        crop_max_y = img_max_y - bbox[1]
    else:
        raise Exception('Wrong bounding box value:', bbox[0], bbox[1], bbox[2], bbox[3])

    crop_data[crop_min_y:crop_max_y, crop_min_x:crop_max_x, :] = filtered_input[img_min_y:img_max_y, img_min_x:img_max_x, :]

    if pad:
        # Check that the crop shape is not already square
        if crop_shape[0] == crop_shape[1]:
            return crop_data

        # Create the new bounding box
        pad_dir = None
        diff = 0
        if bbox[3] > bbox[2]:
            pad_dir = 0
            diff = bbox[3] - bbox[2]
            bbox[2] = bbox[3]
            bbox[0] -= diff // 2
        elif bbox[2] > bbox[3]:
            pad_dir = 1
            diff = bbox[2] - bbox[3]
            bbox[3] = bbox[2]
            bbox[1] -= diff // 2

        new_crop_data = np.full((bbox[3], bbox[2], filtered_input.shape[2]), pad_value,  dtype=filtered_input.dtype)
        if pad_dir == 0:
            new_crop_data[:, diff//2:diff//2+crop_shape[1], :] = crop_data
        elif pad_dir == 1:
            new_crop_data[diff//2:diff//2+crop_shape[0], :, :] = crop_data

        return new_crop_data

    return crop_data
