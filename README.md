# Face Tracking using YOLOv10 and PID Algorithm

This repository contains the code for a face tracking system developed using YOLOv10 and a PID algorithm. The project utilizes a computer with a USB camera, two servo motors, and an ESP32C3-XIAO to control the servos.

## Project Overview

This project involves:
- Experimenting with Haar Cascade and YOLOv8 for initial face detection.
- Training a YOLOv10 model using a face detection dataset from Kaggle on Google Colab.
- Integrating the detection results into a face tracking system.
- Using the center point of the bounding box from YOLOv10 detections and the center point of the camera frame as references for servo movements.
- Implementing a PID algorithm to process these references and control the servo motors, ensuring smooth and accurate tracking.

## Project Workflow

1. **Initial Face Detection Experiments:**
   - Tried using OpenCV's Haar Cascade algorithm for face detection but found the results unsatisfactory.
   - Explored YOLOv8 using a pretrained model from [akanametov's YOLO-Face](https://github.com/akanametov/yolo-face) with promising results.

2. **Dataset and Model Training:**
   - Sourced a face detection dataset from Kaggle: [fareselmenshawii/face-detection-dataset](https://www.kaggle.com/datasets/fareselmenshawii/face-detection-dataset).
   - Trained the YOLOv10 model using Google Colab, leveraging the extensive dataset to fine-tune the model for better accuracy.

3. **Integrating Detection with Face Tracking:**
   - Integrated the face detection results from YOLOv10 into a face tracking system.
   - The center point of the bounding box from the detection results, along with the center point of the camera frame, were used as references for servo movements.
   - Implemented a PID algorithm to process these references and control the servo motors, ensuring smooth and accurate tracking.
   - Tested various parameters for Kp, Ki, and Kd to optimize the camera and servo movements.

## Results

The project demonstrates effective face tracking using YOLOv10 and a PID algorithm. The integration of the bounding box center points from YOLOv10 detections with servo movements provides smooth and accurate tracking.

## Repository Contents

- [training](https://github.com/fitranurmayadi/yolov10_face_tracking/tree/main/Training%20yolov10): Scripts for training the YOLOv10 model.
- [inference](https://github.com/fitranurmayadi/yolov10_face_tracking/tree/main/face_tracking_yolov10): Scripts for running inference on the trained model.
- [Control](https://github.com/fitranurmayadi/yolov10_face_tracking/tree/main/ServoXY_Serial): Code for integrating the detection results with servo control.

## Getting Started

To get started with this project, clone the repository and follow the instructions in the respective directories.

## Acknowledgments

I would like to thank the following for their inspiration and resources that greatly helped in the development of this project:
- [rizkydermawan1992's Face Detection](https://github.com/rizkydermawan1992/Face-Detection)
- [fareselmenshawii's face-detection-dataset on Kaggle](https://www.kaggle.com/datasets/fareselmenshawii/face-detection-dataset)
- [akanametov's YOLO-Face](https://github.com/akanametov/yolo-face)

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.

---

Feel free to check out the repository for more details and the complete code. Let's connect and discuss more about embedded systems, AI, and robotics!

#YOLOv10 #ComputerVision #FaceTracking #PIDControl #MachineLearning #Robotics #EmbeddedSystems
