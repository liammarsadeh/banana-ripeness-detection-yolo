## 🍌 Banana Detection using YOLOv8

A deep learning-based object detection project built using the Ultralytics YOLOv8 framework to detect bananas through custom training and evaluation pipelines.

This project explores the complete computer vision workflow including:

dataset preparation
model training
validation
continued fine-tuning
testing on unseen data

The model was trained incrementally using multiple training sessions while monitoring validation performance and loss reduction to improve generalization.

##    🚀 Features
YOLOv8-based object detection
Custom banana dataset training
Validation on separate test split
Incremental fine-tuning using saved checkpoints
GPU training support
Model evaluation using YOLO metrics
🛠️ Technologies Used
Python
Ultralytics YOLOv8
Jupyter Notebook
##    📁 Project Structure
banana dataset/
├── train/
├── val/
├── test/
└── data.yaml
##    🧠 Training Workflow

The project follows an iterative training strategy:

Initial training for a small number of epochs
Validation and loss monitoring
Reloading saved weights
Continued fine-tuning for additional epochs

Example training code:

from ultralytics import YOLO

model = YOLO('yolov8n.yaml')

model.train(
    data='banana dataset/data.yaml',
    epochs=5,
    imgsz=640,
    device=0
)

Continued training from saved checkpoints:

model = YOLO('runs/detect/train2/weights/last.pt')

model.train(
    data='banana dataset/data.yaml',
    epochs=30,
    imgsz=640,
    device=0
)
📊 Evaluation

The model is evaluated using the test split:

model.val(
    data='banana dataset/data.yaml',
    split='test'
)

##    Metrics include:

mAP
precision
recall
validation loss
##    🎯 Project Goal

The goal of this project is to gain hands-on experience with modern object detection systems by building a complete YOLOv8 training pipeline using a custom banana dataset.

The project focuses on understanding:

how YOLO training works
dataset structure and annotation
model improvement through fine-tuning
validation/testing workflows
practical computer vision experimentation
🔮 Future Improvements
Banana ripeness classification
Real-time webcam detection
Data augmentation improvements
Larger and more balanced datasets
Deployment as a web application
📜 License

This project is created for educational and research purposes.
