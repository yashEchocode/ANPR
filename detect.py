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

def detect(vehicleNumber):
    try:
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
                # return render_camera_akjfdfafdsfafsdf(detectedVehicleList)
        return
    except Exception as err:
        print("error:", err)