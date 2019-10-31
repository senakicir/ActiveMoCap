def find_single_person_bbox(predictions, ):
    max_confidence = 0.5
    bounding_box = None
    for prediction in predictions:
        confidence = prediction[1]
        if (prediction[0] == b'person') and confidence>max_confidence:
            max_confidence = confidence
            bounding_box = prediction[2]
    return bounding_box, max_confidence
