from flask import Flask, render_template, request
import os 
from deeplearning import object_detection
from flask_mysqldb import MySQL
import mysql.connector
import re
from dotenv import load_dotenv
import requests
import json

app = Flask(__name__)

BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH,'static/upload/')

# load_dotenv()
# headers = os.getenv("API_HEADERS")

# print(headers)
# headers = json.loads(headers)
# url = os.getenv("API_URL")


# url = "https://api.edenai.run/v2/ocr/ocr"

# headers = {"Authorization" : "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiMmNkZWFhM2MtNzcyZS00ZGJjLTkyNGUtZTY0ZWU3YjYwODFmIiwidHlwZSI6ImFwaV90b2tlbiJ9.T2OV4eaNPtSkB6L5q2qG4TSXl3Yk77eveiUedLNAcAo"}


# data = {
#     "providers": "google",
#     "language": "en",
#     "fallback_providers": ""
# }

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

@app.route('/',methods=['POST','GET'])
def index():

    if request.method == 'POST':
        upload_file = request.files['image_name']
        filename = upload_file.filename
        path_save = os.path.join(UPLOAD_PATH,filename)
        upload_file.save(path_save)
        text_list = object_detection(path_save,filename)

        def remove_non_alphanumeric(text):
            return re.sub(r'[^a-zA-Z0-9]', '', text)

        text_str = ''.join([remove_non_alphanumeric(text) for text in text_list])

        cur = db.cursor()
        cur.execute("SELECT * FROM VEHICLEDB WHERE vNO = (%s)", (text_str,))
        feachdata = cur.fetchall()

        print("feachdata",type( feachdata))

        if len(feachdata) == 0:
            print("feachdata....")
            feachdata = "No Data Found"


        return render_template('index.html',upload=True,upload_image=filename,text=text_list,no=len(text_list), rol=feachdata)

    return render_template('index.html',upload=False)


if __name__ =="__main__":
    app.run(debug=True)
