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

@app.route('/index',methods=['POST','GET'])
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


if __name__ =="__main__":
    app.run(debug=True)
