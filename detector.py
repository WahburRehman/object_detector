import numpy as np
from ultralytics import YOLO


# Wrapper for Ultralytics YOLO model.
# weights: path to .pt file
# - conf: confidence threshold
# - imgsz: inference size
class YoloDetector:
    def __init__(self, weights="yolov8s.pt", conf=0.25, imgsz=640, device=None):
       
        self.model = YOLO(weights)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self.names = self.model.names  # ID2NAME mapping

     
    # Run YOLO inference on an RGB numpy image.
    # Returns the raw Ultralytics result (res).
    def predict(self, img_rgb):
        res = self.model.predict(
            source=img_rgb,
            imgsz=self.imgsz,
            conf=self.conf,
            device=self.device,
            verbose=False
        )[0]
        return res
