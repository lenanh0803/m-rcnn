import cv2 
import sys
import os
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

from models import utils 
import models.model as modellib
from utils import visualize
from samples import coco

ROOT_DIR = os.getcwd()
sys.path.append(ROOT_DIR)
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()

class MaskModel(): 
    def __init__(self) -> None: 
        #load weight
        self.get_prediction()
        self.load_weight()
    
    def load_weight(self):
        if not os.path.exists(COCO_MODEL_PATH):
           utils.download_trained_weights(COCO_MODEL_PATH)

    def get_prediction(self, image_path):
        #load model
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
        model.load_weights('C:\\Users\\Admin\\Desktop/project AI\\mask_rcnn_coco.h5', by_name=True)

        #load img from the folder
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        #get prediction 
        results = model.detect([image], verbose=1)

        #visualize 
        r = results[0]
        mask_img = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
        return mask_img
