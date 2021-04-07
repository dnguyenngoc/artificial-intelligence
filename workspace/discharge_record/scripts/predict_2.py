import os
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
import sys

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from tensorflow.python.platform import gfile
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.util import compat


# Path to frozen detection graph .pb file, which contains the model that is used
PATH_TO_CKPT = '/home/pot/Desktop/artificial_intelligence/workspace/discharge_record/exported_models/ssd_mobilenet_v2_320x320/saved_model/saved_model.pb'

# Path to label map file
PATH_TO_LABELS = '/home/pot/Desktop/artificial_intelligence/workspace/discharge_record/models/label_map.pbtxt'

# Path to image
PATH_TO_IMAGE = '/home/pot/Desktop/artificial_intelligence/workspace/discharge_record/images/test/1.jpg'

# Number of classes the object detector can identify
NUM_CLASSES = 13

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with gfile.FastGFile(PATH_TO_CKPT, 'rb') as fid:
        data = compat.as_bytes(fid.read())
        sm = saved_model_pb2.SavedModel()
        sm.ParseFromString(data)
#         od_graph_def.ParseFromString(fid.read(-1))
        tf.import_graph_def(sm.meta_graphs[0].graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Load image using OpenCV and
# expand image dimensions to have shape: [1, None, None, 3]
# i.e. a single-column array, where each item in the column has the pixel RGB value
image = cv2.imread(PATH_TO_IMAGE)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_expanded = np.expand_dims(image_rgb, axis=0)

# Perform the actual detection by running the model with the image as input
(boxes, scores, classes, num) = sess.run(
    [detection_boxes, detection_scores, detection_classes, num_detections],
    feed_dict={image_tensor: image_expanded})

# Draw the results of the detection (aka 'visulaize the results')

vis_util.visualize_boxes_and_labels_on_image_array(
    image,
    np.squeeze(boxes),
    np.squeeze(classes).astype(np.int32),
    np.squeeze(scores),
    category_index,
    use_normalized_coordinates=True,
    line_thickness=8,
    min_score_thresh=0.10)

cv2.imwrite('/home/pot/Desktop/artificial_intelligence/workspace/discharge_record/scripts/end.jpg', image_with_detections)