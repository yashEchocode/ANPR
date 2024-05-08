import numpy as np
import cv2
import matplotlib.pyplot as plt
import pytesseract as pt
import re
import json
import requests
import os
from dotenv import load_dotenv

load_dotenv()
headers = os.getenv("HEADERS")

# print(headers)
headers = json.loads(headers)
url = os.getenv("URL")

data = {
    "providers": "google",
    "language": "en",
    "fallback_providers": "amazon"
}

# Loading MODEL
INPUT_WIDTH =  640
INPUT_HEIGHT = 640
net = cv2.dnn.readNetFromONNX('./static/models/best.onnx')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)



def get_detections(img,net):
    image = img.copy()
    row, col, d = image.shape

    max_rc = max(row,col)
    input_image = np.zeros((max_rc,max_rc,3),dtype=np.uint8)
    input_image[0:row,0:col] = image
    blob = cv2.dnn.blobFromImage(input_image,1/255,(INPUT_WIDTH,INPUT_HEIGHT),swapRB=True,crop=False) #gets predication
    net.setInput(blob)
    preds = net.forward()
    detections = preds[0]
    
    return input_image, detections

def filter_detection(input_image,detections):
    boxes = []
    confidences = []

    image_w, image_h = input_image.shape[:2]
    x_factor = image_w/INPUT_WIDTH
    y_factor = image_h/INPUT_HEIGHT

    for i in range(len(detections)):
        row = detections[i]
        confidence = row[4]
        if confidence > 0.4:
            class_score = row[5] # probability score of license plate
            if class_score > 0.25:
                cx, cy , w, h = row[0:4]

                left = int((cx - 0.5*w)*x_factor)
                top = int((cy-0.5*h)*y_factor)
                width = int(w*x_factor)
                height = int(h*y_factor)
                box = np.array([left,top,width,height])

                confidences.append(confidence)
                boxes.append(box)

    boxes_np = np.array(boxes).tolist()
    confidences_np = np.array(confidences).tolist()
    index = np.array(cv2.dnn.NMSBoxes(boxes_np,confidences_np,0.25,0.45)).flatten()
    
    return boxes_np, confidences_np, index

def extract_text(image,bbox):
    x,y,w,h = bbox
    
    roi = image[y:y+h, x:x+w]

    if 0 in roi.shape:
        return ''
    else:
        roi_bgr = cv2.cvtColor(roi,cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(roi_bgr,cv2.COLOR_BGR2GRAY)
        magic_color = apply_brightness_contrast(gray,brightness=40,contrast=70)
        text = pt.image_to_string(magic_color,lang='eng',config='--psm 6')
        text = text.strip()
        
        roi_filename = f'{"static/roi"}/roi_{x}_{y}.jpg'
        cv2.imwrite(roi_filename, roi)

        files = {"file": open(roi_filename, 'rb')}
        response = requests.post(url, data=data, files=files, headers=headers)
        result = json.loads(response.text)
        # print("Deep Api Result - ", result["google"]["text"])
        text = result["google"]["text"]

        def clean_vehicle_number(text):
            pattern = r'^(?:IND)?([^A-Z]*(\b[A-Z]{2}.+))'
            match = re.match(pattern, text)
            if match:
                return match.group(1)
            else:
                return ''

        text = clean_vehicle_number(text)

        return text


def drawings(image,boxes_np,confidences_np,index, path_save):
    text_list = []
    for ind in index:
        x,y,w,h =  boxes_np[ind]
        bb_conf = confidences_np[ind]
        conf_text = 'plate: {:.0f}%'.format(bb_conf*100)
        license_text = extract_text(image,boxes_np[ind])

        text_width, text_height = cv2.getTextSize(license_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]

        cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,0),2)
        cv2.rectangle(image,(x,y-30),(x+w,y),(0,0,0),-1)
        cv2.rectangle(image,(x,y+h),(x+max(text_width,w),y+h+40),(0,0,0),-1)


        cv2.putText(image,conf_text,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),1)
        cv2.putText(image,license_text,(x,y+h+27),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),1)
        
        text_list.append(license_text)

    return image,  text_list


# predictions
def predictions(img,net, path):
    ## detections
    input_image, detections = get_detections(img,net)
    boxes_np, confidences_np, index = filter_detection(input_image, detections)
    return index, boxes_np, confidences_np
    ## Drawings
    # result_img, text = drawings(img,boxes_np,confidences_np,index, path)
    # return result_img, text

def predictions_input(img,net, path):
    ## detections
    input_image, detections = get_detections(img,net)
    boxes_np, confidences_np, index = filter_detection(input_image, detections)
    ## Drawings
    result_img, text = drawings(img,boxes_np,confidences_np,index, path)
    print('drawing output', text)
    return result_img, text 


def object_detection(path,filename):
    # read image
    image = cv2.imread(path)
    # print(path)
    image = np.array(image,dtype=np.uint8)
    result_img, text_list = predictions_input(image,net,path)
    print('text list prediction',text_list)
    cv2.imwrite('./static/predict/{}'.format(filename),result_img)
    return text_list

def object_detection_camera(image):
    # read image
    # image = cv2.imread(path)
    # print(path)
    image = np.array(image,dtype=np.uint8)
    result_img, text_list = predictions(image,net,'')
    # cv2.imwrite('./static/predict/{}'.format(filename),result_img)
    return text_list



def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):
    
        if brightness != 0:
            if brightness > 0:
                shadow = brightness
                highlight = 255
            else:
                shadow = 0
                highlight = 255 + brightness
            alpha_b = (highlight - shadow)/255
            gamma_b = shadow
            
            buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
        else:
            buf = input_img.copy()
        
        if contrast != 0:
            f = 131*(contrast + 127)/(127*(131-contrast))
            alpha_c = f
            gamma_c = 127*(1-f)
            
            buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

        return buf