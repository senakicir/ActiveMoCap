def find_single_person_bbox(predictions):
    """
        Returns square bounding box.
    """
    max_confidence = 0.5
    bounding_box = None
    for prediction in predictions:
        confidence = prediction[1]
        if (prediction[0] == b'person') and confidence>max_confidence:
            max_confidence = confidence
            bounding_box = list(prediction[2])
    if bounding_box is not None:
        length = max(bounding_box[2], bounding_box[3])
        bounding_box = [int(bounding_box[0]), int(bounding_box[1]), int(length), int(length)]
    return bounding_box, max_confidence
