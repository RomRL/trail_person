import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util

# Load the model
# Adjust the paths according to where you have stored the downloaded model

pipeline_config = '/home/romh/person_detection_/tensor_detect/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/pipeline.config'
model_dir = '/home/romh/person_detection_/tensor_detect/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/checkpoint'
configs = config_util.get_configs_from_pipeline_file(pipeline_config)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore('/home/romh/person_detection_/tensor_detect/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/checkpoint/ckpt-0').expect_partial()

@tf.function
def detect_fn(image):
    """Detect objects in image."""
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

# Load label map data (for plotting)
# Adjust the path to where you've stored the label map file
label_map_path = '/home/romh/person_detection_/tensor_detect/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/mscoco_label_map.pbtxt'
label_map = label_map_util.load_labelmap(label_map_path)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=label_map_util.get_max_label_map_index(label_map), use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load an image
image_path = '/home/romh/person_detection_/tensor_detect/000000115870.jpg'
image_np = np.array(Image.open(image_path))

# Run detection
input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
detections = detect_fn(input_tensor)
person_detections = detections['detection_classes'][0].numpy() == 1
person_scores = detections['detection_scores'][0].numpy()[person_detections]

for score in person_scores:
    print(f"Person detected with confidence: {score:.2%}")


# Visualize the results
label_id_offset = 1
image_np_with_detections = image_np.copy()

viz_utils.visualize_boxes_and_labels_on_image_array(
      image_np_with_detections,
      detections['detection_boxes'][0].numpy(),
      (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
      detections['detection_scores'][0].numpy(),
      category_index,
      use_normalized_coordinates=True,
      max_boxes_to_draw=200,
      min_score_thresh=.1,
      agnostic_mode=False)

plt.figure(figsize=(12, 8))
plt.imshow(image_np_with_detections)
#save the image
plt.savefig('output.jpg')
