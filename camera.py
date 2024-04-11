# importing OpenCV library
import cv2
import os
from deeplearning import predictions
import json
import requests
import os
from dotenv import load_dotenv
import time

load_dotenv()
headers = os.getenv("HEADERS")

print(headers)
headers = json.loads(headers)
url = os.getenv("URL")

data = {
    "providers": "amazon",
    "language": "en",
    "fallback_providers": ""
}

BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH, 'static/upload/')

INPUT_WIDTH =  640
INPUT_HEIGHT = 640
net = cv2.dnn.readNetFromONNX('./static/models/best.onnx')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def capture_image():
    # initialize the camera
    cam_port = 0  # If you have multiple cameras, change this value
    cam = cv2.VideoCapture(cam_port)

    while True:
        print("Capturing...")
        result, image = cam.read()
        print('result:', result)

        if result:
            # Pass the captured image to the predictions function
            index, boxes_np = predictions(image, net, '')

            if index is not None:
                for ind in index:
                    x, y, w, h = boxes_np[ind]
                    roi = image[y:y+h, x:x+w]
                    roi_filename = f'static/roi/roi_{x}_{y}.jpg'
                    cv2.imwrite(roi_filename, roi)
                    files = {"file": open(roi_filename, 'rb')}
                    response = requests.post(url, data=data, files=files, headers=headers)
                    result = json.loads(response.text)
                    text = result["amazon"]["text"]
                    print("Vehicle Number:", text)                    
                time.sleep(5)
            else:
                time.sleep(1)
        else:
            print("Failed to capture image from the camera.")
            time.sleep(1)