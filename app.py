from flask import Flask, render_template, request, jsonify
import os 
from deeplearning import object_detection, object_detection_camera
from flask_mysqldb import MySQL
import mysql.connector
import re
from dotenv import load_dotenv
import requests
import json
import cv2
import numpy as np
from flask_socketio import SocketIO
from deeplearning import predictions
import time
import threading  # Import threading module

app = Flask(__name__)
socketio = SocketIO(app)

BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH,'static/upload/')

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

if(db is not None):
    print("connected to database")

mysql = MySQL(app)

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

cur = db.cursor()
cur.execute("select vNO from VEHICLEDB")
dbCol = cur.fetchall()

# print(dbCol)


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

def start_camera():
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
                        vehicle_number = result["amazon"]["text"]
                        # return vehicleNumber
                        # detect(vehicleNumber)

                        def remove_non_alphanumeric(text):
                            return re.sub(r'[^a-zA-Z0-9]', '', text)

                        vehicle_number = ''.join([remove_non_alphanumeric(text) for text in vehicle_number])
                        print("Vehicle Number:", vehicle_number)

                        cur.execute("SELECT vNO, roll FROM VEHICLEDB WHERE vNO = (%s)", (vehicle_number,))
                        vehicleData = cur.fetchall()
                        print("Vehicle Data:", vehicleData)
                        vehicle_data = {"number": vehicle_number, "vehicleData": vehicleData}  # Modify this to include additional vehicle data
                        if not re.match(r'^[A-Za-z]+$', vehicle_number):
                            emit_vehicle_data(vehicle_data)
                    else:
                        print("")
                time.sleep(3)
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

@app.route('/')
@app.route('/first')
def first():
    return render_template('first.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/camera', methods=['POST','GET'])
def camera():
    # Start camera in a separate thread
    camera_thread = threading.Thread(target=start_camera)
    camera_thread.start()
    return render_template('camera.html')

def emit_vehicle_data(vehicle_data):
    socketio.emit('vehicle_detected', vehicle_data)

@app.route('/index',methods=['POST','GET'])
def index():

    if request.method == 'POST':
        upload_file = request.files['image_name']
        filename = upload_file.filename
        path_save = os.path.join(UPLOAD_PATH,filename)
        upload_file.save(path_save)
        # path_save, filename = capture_image()
        text_list = object_detection(path_save,filename)
        print('vehicle number',text_list)


        def remove_non_alphanumeric(text):
            return re.sub(r'[^a-zA-Z0-9]', '', text)

        text_str = ''.join([remove_non_alphanumeric(text) for text in text_list])

        cur = db.cursor()
        cur.execute("SELECT vNO, roll FROM VEHICLEDB WHERE vNO = (%s)", (text_str,))
        feachdata = cur.fetchall()

        if len(feachdata) > 0:
            feachdata = feachdata[0]

        print("feachdata",feachdata)

        if len(feachdata) == 0:
            print("feachdata....")
            feachdata = "No Data Found"


        return render_template('index.html',upload=True,upload_image=filename,text=text_list,no=len(text_list), rol=feachdata)

    return render_template('index.html',upload=False)

if __name__ =="__main__":
    socketio.run(app, debug=True)
