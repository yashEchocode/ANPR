# importing OpenCV library
import cv2
import os
from deeplearning import predictions
import json
import requests
import os
from dotenv import load_dotenv
import time
from flask_mysqldb import MySQL
import mysql.connector
from flask import Flask, render_template, request
# from app import render_camera_akjfdfafdsfafsdf
# from detect import detect

load_dotenv()
headers = os.getenv("HEADERS")

# print(headers)
headers = json.loads(headers)
url = os.getenv("URL")

data = {
    "providers": "amazon",
    "language": "en",
    "fallback_providers": "google"
}

BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH, 'static/upload/')

INPUT_WIDTH =  640
INPUT_HEIGHT = 640
net = cv2.dnn.readNetFromONNX('./static/models/best.onnx')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


db = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="Edith@2023",
    database="ANPR"
)

cur = db.cursor()

def drawings(image, boxes_np, confidences_np, index, path_save):
    text_list = []
    for ind in index:
        x, y, w, h =  boxes_np[ind]
        bb_conf = confidences_np[ind]
        conf_text = 'plate: {:.0f}%'.format(bb_conf * 100)
        # license_text = extract_text(image,boxes_np[ind])

        # text_width, text_height = cv2.getTextSize(license_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]

        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 0), 2)
        cv2.rectangle(image, (x, y-30), (x+w, y), (0, 0, 0), -1)
        # cv2.rectangle(image,(x,y+h),(x+max(text_width,w),y+h+40),(0,0,0),-1)

        cv2.putText(image, conf_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        # cv2.putText(image,license_text,(x,y+h+27),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),1)

        # text_list.append(license_text)

    return image, text_list

def detect(vehicleNumber):
    try:
        from app import render_camera
        cur.execute("SELECT * FROM detected_vehicle order by timeStamp DESC LIMIT 1")
        lastNumber = cur.fetchone()
        if vehicleNumber != lastNumber:
            cur.execute("SELECT vNO, roll FROM VEHICLEDB WHERE vNO = (%s)", (vehicleNumber,))
            vehicleData = cur.fetchall() # vehicleNumer, role 
            if vehicleData is not None: 
                print("detecting...")
                current_timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                sql_query = """
                    INSERT INTO detected_vehicle (vehicleNum, timeStamp, status, role)
                    VALUES (%s, %s, %s, %s)
                    """
                values = (vehicleNumber, current_timestamp, 'AUTHORIZED', 'vehicleData')
                cur.execute(sql_query, values)

                cur.execute("select * from detected_vehicle")
                detectedVehicleList = cur.fetchall()
                # detectedVehicleList = list(detectedVehicleList)

                url = 'http://localhost:5000/render_camera'  # Update URL if needed
                headers = {'Content-Type': 'application/json'}

                # render_template_before_route()
                # return requests.post(url, headers=headers, data=json.dumps(detectedVehicleList))
                # return render_template('camera.html', vehicle=detectedVehicleList)
                return render_camera(detectedVehicleList)
        return
    except Exception as err:
        print("error:", err)

def capture_image():
    # initialize the camera
    cam_port = 0  # If you have multiple cameras, change this value
    cam = cv2.VideoCapture(cam_port)

    while True:
        print("Capturing...")
        result, image = cam.read()
        # print('result:', result)

        if result:
            cv2.waitKey(1)
            index, boxes_np, confidences_np = predictions(image, net, '')

            if index is not None:
                image_with_boxes, _ = drawings(image.copy(), boxes_np, confidences_np, index, '')

                cv2.imshow('Annotated Image', image_with_boxes)
                cv2.waitKey(1) 
                for ind in index:
                    x, y, w, h = boxes_np[ind]
                    
                    roi = image[y:y+h, x:x+w]
                    roi_filename = f'static/roi/roi_{x}_{y}.jpg'
                    cv2.imwrite(roi_filename, roi)
                    files = {"file": open(roi_filename, 'rb')}
                    response = requests.post(url, data=data, files=files, headers=headers)
                    result = json.loads(response.text)
                    if 'amazon' in result:
                        vehicleNumber = result["amazon"]["text"]
                        # return vehicleNumber
                        # detect(vehicleNumber)
                        print("Vehicle Number:", vehicleNumber)
                        cur.execute("SELECT vNO, roll FROM VEHICLEDB WHERE vNO = (%s)", (vehicleNumber,))
                        vehicleData = cur.fetchall()
                        print("Vehicle Data:", vehicleData)
                        vehicle_data = {"number": vehicle_number, "additional_data": vehicleData}  # Modify this to include additional vehicle data
                        socketio.emit('vehicle_detected', vehicle_data)
                    elif 'google' in result:
                        vehicleNumber = result["google"]["text"]
                        # return vehicleNumber
                        # detect(vehicleNumber)
                        print("Vehicle Number:", vehicleNumber)
                        cur.execute("SELECT vNO, roll FROM VEHICLEDB WHERE vNO = (%s)", (vehicleNumber,))
                        vehicleData = cur.fetchall()
                        print("Vehicle Data:", vehicleData)
                    else:
                        print("")
                time.sleep(1)
            else:
                time.sleep(1)
        else:
            print("Failed to capture image from the camera.")
            time.sleep(1)

        # Close the camera feed window when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    cam.release()
    cv2.destroyAllWindows()