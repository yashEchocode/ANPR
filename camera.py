import cv2
import numpy as np
from flask import Flask, render_template, request
import os 
from deeplearning import predictions
from flask_mysqldb import MySQL
import mysql.connector

# Load the object detection model (assuming 'net' is already defined)
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
net = cv2.dnn.readNetFromONNX('./static/models/best.onnx')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Open a connection to the camera (camera index 0 represents the default camera)
def openCamera():
    cap = cv2.VideoCapture(0)

    while True:
        # Capture a frame from the camera
        ret, frame = cap.read()

        result_img, text_list = predictions(frame, net, "")

        print(text_list)
        cv2.imshow('Object Detection', result_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
