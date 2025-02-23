
# Custom TensorFlow Object Detection for Hand Gesture Recognition


This project implements an end-to-end pipeline for a custom object detection model using TensorFlow's Object Detection API. The model is designed to recognize a set of hand gestures (such as thumbs-up, thumbs-down, thank you, live long, and peace) from live video feeds.






## Project Overview

- **Dataset Collection:** The project includes scripts to capture images via a webcam, organize them into labeled directories, and generate TFRecords for training.

- **Model Configuration and Training:** A custom pipeline configuration is created to fine-tune a pre-trained SSD MobileNet model on the collected hand gesture dataset. The training setup leverages GPU acceleration for efficient training and real-time detection.

- **Evaluation and Real-Time Detection:** The trained model achieves robust detection performance, with an average precision of 85% on evaluation metrics. The project also integrates live detection using OpenCV to visualize the model’s predictions in real time.

- **Modular and Extensible:** The repository is structured with clearly defined scripts and modules, making it easy to adapt and extend for other custom object detection applications.










