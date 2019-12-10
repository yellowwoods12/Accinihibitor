import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import requests

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import os
import sys

if len(sys.argv) != 3:
    print("Usage: python classifier.py [Input Image] [Output Image]")
    sys.exit(0)

input_image = sys.argv[1]

output_image = sys.argv[2]

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class AccidentsClassifier(object):
    def __init__(self):
        PATH_TO_MODEL = 'graph/frozen_inference_graph.pb'
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            # Works up to here.
            with tf.gfile.GFile(PATH_TO_MODEL, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.d_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.d_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.d_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_d = self.detection_graph.get_tensor_by_name('num_detections:0')
        
        self.sess = tf.Session(graph=self.detection_graph)

    def get_classification(self, img):
        # Bounding Box Detection.
        with self.detection_graph.as_default():
            # Expand dimension since the model expects image to have shape [1, None, None, 3].
            img_expanded = np.expand_dims(img, axis=0)  
            (boxes, scores, classes, num) = self.sess.run(
                [self.d_boxes, self.d_scores, self.d_classes, self.num_d],
                feed_dict={self.image_tensor: img_expanded}
            )
            """(scores, classes) = self.sess.run(
                [self.d_scores, self.d_classes],
                feed_dict={self.image_tensor: img_expanded}
            )"""
        return boxes, scores, classes, num

PATH_TO_LABELS = 'accidents.pbtxt'
NUM_CLASSES = 1

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

img = plt.imread(input_image)#[::-1,:,::-1]
img.setflags(write=1)

x = AccidentsClassifier()

boxes, scores, classes, num = x.get_classification(img)
#scores, classes = x.get_classification(img)

vis_util.visualize_boxes_and_labels_on_image_array(img, np.squeeze(boxes), np.squeeze(classes).astype(np.int32), np.squeeze(scores), category_index, use_normalized_coordinates=True, line_thickness=8)

plt.imsave(output_image, img)

url = "https://www.fast2sms.com/dev/bulk"
payload = "sender_id=FSTSMS&message=Accident has been detected at location https://www.google.com/maps/d/u/0/viewer?ie=UTF8&oe=UTF8&msa=0&mid=1CnQ93S-uOjPAfUt1DbxWHWPVw74&ll=28.533928000000017%2C77.34853999999996&z=17&language=english&route=p&numbers=7355780958"
headers = {'authorization': "wvJesBUI4mGl5AYpkEDjr1q9ZaFcz2oOChS3RXdtfixMVTHLQbxqUplsf9IAibQ23yzBVt6RCwkehjDS",
           'Content-Type': "application/x-www-form-urlencoded",
           'Cache-Control': "no-cache",}

response = requests.request("POST", url, data=payload, headers=headers)

#IMAGE_SIZE = (12, 8)
#plt.figure(figsize=IMAGE_SIZE)
#plt.imshow(img)
print("done")
print(response.text)
