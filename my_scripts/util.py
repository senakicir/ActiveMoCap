import cv2


def prepare_image(image_path, height):
    orig_image = cv2.imread(image_path)
    if height == -1:
        return orig_image
    # Compute the new width
    orig_image = orig_image[:,:,0:3]

    scale = height / orig_image.shape[0]
    scaled_image = cv2.resize(orig_image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return scaled_image
