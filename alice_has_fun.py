from datetime import datetime
from distutils.command.build_scripts import first_line_re
import cv2
import numpy as np
from keras.models import load_model
import numpy as np
import time
import threading
from paho.mqtt import client as mqtt_client
import random
import json
import os

Config = None

mqttclient = None
input_sounrce = None
MQTT_BROKER = None
MQTT_PORT = 1883
client_id = f'alice_fun-mqtt-{random.randint(0, 1000)}'
mqtt_pub = True
DRAW_RECT = False
min_predict_threshold = 65
model = load_model('keras_model.h5')
facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
labels = open('labels.txt', 'r').read().splitlines()

FRAME_RECOGNITION_RATE = 4
recogintion_sleep_delay = 1/FRAME_RECOGNITION_RATE
frame_update_rate = 10
publish_min_dalay = 1

outputFrame = None
drawFrame = None
last_publish = datetime.now()
faces_dict = {}
summ_face_dict = {}
Face_Occuracy_Count = 0


def frame_update():
    global outputFrame, camera, frame_updated
    while True:
        ret, frame = camera.read()
        if not ret:
            check = False
            while not check:
                camera = cv2.VideoCapture(input_sounrce)
                ret, frame = camera.read()
                if ret:
                    check = True
                else:
                    time.sleep(0.5)
        outputFrame = frame.copy()
        # time.sleep(frame_update_sleep_delay)


def get_str_date_fname(dt_now):
    dir = './on_detect/'+dt_now.strftime("%Y-%m-%d")
    name = dt_now.strftime("%Y-%m-%d_%H_%M_%S")
    return [dir, name]


def save_image():
    dt_now = datetime.now()
    dir, name = get_str_date_fname(dt_now)
    if not os.path.exists(dir):
        os.mkdir(dir)
    name = dir+"/"+name+".jpg"
    print("Creating Images........." + name)
    cv2.imwrite(name, drawFrame)


def face_recogintion():
    global last_publish, summ_face_dict, Face_Occuracy_Count, drawFrame
    while True:
        if outputFrame is None:
            time.sleep(0.1)
            continue
        drawFrame = outputFrame.copy()
        faces = facedetect.detectMultiScale(
            drawFrame, scaleFactor=1.3, minNeighbors=5)
        if len(faces) == 0:
            time.sleep(0.1)
            continue
        dt_now = datetime.now()
        faces_dict = {}
        for x, y, w, h in faces:
            image = drawFrame[y:y+h, x:x+h]
            if image.size == 0:
                continue
            image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
            image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
            image = (image / 127.5) - 1
            probabilities = model.predict(image)
            max_label = ""
            max_prob = -1
            Face_Occuracy_Count += 1
            faces_dict = dict.fromkeys(faces_dict, 0)
            for i in range(0, len(probabilities[0])):
                faces_dict[labels[i]] = round(probabilities[0][i]*100, 2)
                # if labels[i] not in summ_face_dict:
                #     summ_face_dict[labels[i]] = 0
                prob = round(probabilities[0][i]*100, 2)
                # summ_face_dict[labels[i]] += prob
                if prob > min_predict_threshold:
                    if labels[i] not in summ_face_dict:
                        summ_face_dict[labels[i]] = 0
                    if summ_face_dict[labels[i]] < prob:
                        summ_face_dict[labels[i]] = prob
                if DRAW_RECT:
                    cv2.rectangle(drawFrame, (x, y),
                                  (x+w, y+h), (0, 255, 0), 2)
                    if prob > min_predict_threshold:
                        cv2.rectangle(drawFrame, (x, y-40),
                                      (x+w, y), (0, 255, 100), -2)
                        cv2.putText(drawFrame, str(prob)+' '+labels[i], (x, y-10), cv2.FONT_HERSHEY_COMPLEX,
                                    0.75, (0, 10, 10), 1, cv2.LINE_AA)
            # print(f"{summ_face_dict}")
        diff_time = dt_now-last_publish
        if Config['save_on_detect'] and diff_time.seconds > publish_min_dalay and len(summ_face_dict) > 0:
            save_save_image_thread = threading.Thread(target=save_image)
            save_save_image_thread.start()
        if diff_time.seconds > publish_min_dalay:
            faces_json = ""
            # for key in summ_face_dict.keys():
            #     summ_face_dict[key] /= Face_Occuracy_Count
            #     if summ_face_dict[key] > max_prob:
            #         max_prob = summ_face_dict[key]
            #         max_label = key
            faces_json = json.dumps(summ_face_dict, ensure_ascii=False)
            # summ_face_dict = dict.fromkeys(summ_face_dict, 0)
            summ_face_dict = {}
            if mqtt_pub:
                mqtt_publish("for_alice/peoples", faces_json)
            else:
                print(f"{Face_Occuracy_Count}, {faces_json}")
                #mqtt_publish("for_alice/peoples", faces_json)
            Face_Occuracy_Count = 0
            last_publish = dt_now

        time.sleep(recogintion_sleep_delay)


def connect_mqtt(username, password, broker, port, client_id):
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)
    # Set Connecting Client ID
    client = mqtt_client.Client(client_id)
    client.username_pw_set(username, password)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client


def mqtt_publish(topic, message):
    global mqttclient
    result = mqttclient.publish(topic, message)
    status = result[0]
    if status == 0:
        print(f"Send `{message}` to topic `{topic}`")
    else:
        print(f"Failed to send message to topic {topic}")
        print("Try reconnect...")
        mqttclient = connect_mqtt("", "", MQTT_BROKER, MQTT_PORT, client_id)
        result = mqttclient.publish(topic, message)
        status = result[0]
        if status == 0:
            print(f"Send `{message}` to topic `{topic}`")


if __name__ == '__main__':
    with open('config.json') as json_file:
        Config = json.load(json_file)
    input_sounrce = Config['input_sounrce']
    camera = cv2.VideoCapture(input_sounrce)
    camera.set(3, Config['camera_w'])
    camera.set(4, Config['camera_h'])
    DRAW_RECT = Config['draw_rect']
    MQTT_BROKER = Config['mqtt_broker']
    mqtt_pub = Config['mqtt_pub']
    MQTT_PORT = Config['mqtt_port']
    FRAME_RECOGNITION_RATE = Config['frame_recognition_rate']
    min_predict_threshold = Config['min_predict_threshold']
    publish_min_dalay = Config['publish_min_dalay']
    mqttclient = connect_mqtt("", "", MQTT_BROKER, MQTT_PORT, client_id)
    frame_update_thread = threading.Thread(target=frame_update)
    frame_update_thread.start()
    face_recognition_thread = threading.Thread(target=face_recogintion)
    face_recognition_thread.start()
    while True:
        if drawFrame is None:
            time.sleep(0.2)
            continue
        if Config['show_camera']:
            cv2.imshow('Camera', drawFrame)
        keyboard_input = cv2.waitKey(1)
        if keyboard_input == 27:
            break

    camera.release()
    cv2.destroyAllWindows()
