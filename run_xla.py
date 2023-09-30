import os
os.environ["YOLO_XLA"] = "Env variable exists"

from ultralytics import YOLO

if __name__ == "__main__":
    # Load a model
    model = YOLO('yolov8n-cls.yaml')  # build a new model from YAML

    # Train the model with 2 GPUs
    results = model.train(data='mnist160', epochs=100, imgsz=64, device='xla')