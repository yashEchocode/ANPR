from flask import Flask, render_template, request, jsonify
import os 
from deeplearning import object_detection, object_detection_camera
from flask_mysqldb import MySQL
import mysql.connector
import re
from dotenv import load_dotenv
import requests
import json
# from camera import openCamera
import cv2
import numpy as np
import base64
import matplotlib.pyplot as plt
import uuid
from camera import capture_image

app = Flask(__name__)

BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH,'static/upload/')

db = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="Edith@2023",
    database="ANPR"
)

if(db is not None):
    print("connected to database")

mysql = MySQL(app)

cur = db.cursor()
cur.execute("select vNO from VEHICLEDB")
dbCol = cur.fetchall()

# print(dbCol)

@app.route('/')
@app.route('/first')
def first():
    return render_template('first.html')

@app.route('/login')
def login():
    return render_template('login.html')

# @app.route('/camera')
# def camera():

#     # if request.method == 'POST':
#     #     path_save, filename = capture_image()
#     #     text_list = object_detection(path_save,filename)
#     #     print("text_list",text_list)
#     return render_template('index.html')

# @app.route('/camera')
# def camera():
#     path_save, filename = capture_image()
#     text_list = object_detection(path_save,filename)
#     print("text_str",text_list)


#     def remove_non_alphanumeric(text):
#         return re.sub(r'[^a-zA-Z0-9]', '', text)

#     text_str = ''.join([remove_non_alphanumeric(text) for text in text_list])

#     cur = db.cursor()
#     cur.execute("SELECT vNO, roll FROM VEHICLEDB WHERE vNO = (%s)", (text_str,))
#     feachdata = cur.fetchall()

#     if len(feachdata) > 0:
#         feachdata = feachdata[0]

#     print("feachdata",type(feachdata))

#     if len(feachdata) == 0:
#         print("feachdata....")
#         feachdata = "No Data Found"


#     return render_template('index.html',upload=True,upload_image=filename,text=text_list,no=len(text_list), rol=feachdata)


#     return render_template('camera.html')



@app.route('/camera')
def camera():
    vehicleNumber = capture_image()
    return render_template('camera.html')

# def render_template_before_route():
#     # Define your template string with Jinja2 syntax
#     template_string = """
#     <!DOCTYPE html>
#     <html lang="en">
#     <head>
#         <meta charset="UTF-8">
#         <title>Pre-rendered Template</title>
#     </head>
#     <body>
#         <h1>Hello, {{ name }}</h1>
#         <p>This template was rendered before the route execution.</p>
#     </body>
#     </html>
#     """

#     # Render the template string with some context data
#     rendered_template = render_template_string(template_string, name="John")

#     # In a real scenario, you might return the rendered template
#     # return rendered_template

#     # For demonstration, just print the rendered template
#     print(rendered_template)

# Call the function to render the template

# @app.route('/render_camera', methods=['POST'])
def render_camera(data):
    # if request.method == 'POST':
        # data = request.json
    print('render_camera', data)
    return render_template('camera.html', vehicle=data)

@app.route('/get_vehicle_data')
def get_vehicle_data():
    # if request.method == 'POST':
        # data = request.json
    cur.execute("select * from detected_vehicle")
    detectedVehicleList = cur.fetchall()
    data = list(detectedVehicleList)
    # print('render_camera', detectedVehicleList)
    return jsonify(detectedVehicleList)
    # return render_template('camera.html', vehicle=data)



@app.route('/index',methods=['POST','GET'])
def index():

    if request.method == 'POST':
        upload_file = request.files['image_name']
        filename = upload_file.filename
        path_save = os.path.join(UPLOAD_PATH,filename)
        upload_file.save(path_save)
        # path_save, filename = capture_image()
        text_list = object_detection(path_save,filename)


        def remove_non_alphanumeric(text):
            return re.sub(r'[^a-zA-Z0-9]', '', text)

        text_str = ''.join([remove_non_alphanumeric(text) for text in text_list])
        print(type(text_str))

        cur = db.cursor()
        cur.execute("SELECT vNO, roll FROM VEHICLEDB WHERE vNO = (%s)", (text_str,))
        feachdata = cur.fetchall()

        if len(feachdata) > 0:
            feachdata = feachdata[0]

        print("feachdata",type( feachdata))

        if len(feachdata) == 0:
            print("feachdata....")
            feachdata = "No Data Found"


        return render_template('index.html',upload=True,upload_image=filename,text=text_list,no=len(text_list), rol=feachdata)

    return render_template('index.html',upload=False)


# @app.route('/process_image', methods=['POST'])
# def process_image():
   
#     image_file = request.json['image_data']  # Extract image data from JSON payload
#     decoded_image_data = np.frombuffer(base64.b64decode(image_file.split(',')[1]), np.uint8)
#     image = cv2.imdecode(decoded_image_data, cv2.IMREAD_COLOR)
#     plt.imshow(image)
#     plt.show()

    
#     filename = str(uuid.uuid4()) + '.jpg' 
#     path_save = os.path.join(UPLOAD_PATH, filename)
#     cv2.imwrite(path_save, image)

#     text_list = object_detection(path_save,filename)

#     def remove_non_alphanumeric(text):
#         return re.sub(r'[^a-zA-Z0-9]', '', text)

#     text_str = ''.join([remove_non_alphanumeric(text) for text in text_list])
#     cur = db.cursor()
#     cur.execute("SELECT vNO, roll FROM VEHICLEDB WHERE vNO = (%s)", (text_str,))
#     feachdata = cur.fetchall()

#     if len(feachdata) > 0:
#         feachdata = feachdata[0]

#     print("feachdata",feachdata)

#     if len(feachdata) == 0:
#         print("feachdata....")
#         feachdata = "No Data Found"
    
   
#     return render_template('index.html',upload=True,upload_image=filename,text=text_list,no=len(text_list), rol=feachdata)


# @app.route('/run_script',methods=['POST','GET'])
# def camera():
#     openCamera()

if __name__ =="__main__":
    app.run(debug=True)
