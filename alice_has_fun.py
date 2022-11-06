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
DEBUG = True
model = load_model('keras_model.h5')
facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
labels = open('labels.txt', 'r').read().splitlines()

frame_recognition_rate = 4
recogintion_sleep_delay = 1/frame_recognition_rate
frame_update_rate = 10
publish_min_dalay = 1

outputFrame = None
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


def face_recogintion():
    global faces_dict, last_publish, summ_face_dict, Face_Occuracy_Count
    while True:
        if outputFrame is None:
            time.sleep(0.1)
            continue
        _image = outputFrame.copy()
        image = _image
        faces = facedetect.detectMultiScale(image, 1.3, 5)
        if len(faces) == 0:
            time.sleep(0.1)
            continue
        for x, y, w, h in faces:
            image = image[y:y+h, x:x+h]
            if image.size == 0:
                continue
            image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
            image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
            image = (image / 127.5) - 1
            probabilities = model.predict(image)
            faces_dict = dict.fromkeys(faces_dict, 0)
            max_label = ""
            max_prob = -1
            Face_Occuracy_Count += 1
            for i in range(0, len(probabilities[0])):
                faces_dict[labels[i]] = round(probabilities[0][i]*100, 2)
                if labels[i] not in summ_face_dict:
                    summ_face_dict[labels[i]] = 0
                summ_face_dict[labels[i]] += faces_dict[labels[i]]
            dt_now = datetime.now()
            diff_time = dt_now-last_publish
            faces_json = ""
            if diff_time.seconds > publish_min_dalay:
                for key in summ_face_dict.keys():
                    summ_face_dict[key] /= Face_Occuracy_Count
                    if summ_face_dict[key] > max_prob:
                        max_prob = summ_face_dict[key]
                        max_label = key
                faces_json = json.dumps(summ_face_dict, ensure_ascii=False)
                Face_Occuracy_Count = 0
                summ_face_dict = dict.fromkeys(summ_face_dict, 0)
                if Config['save_on_detect']:
                    dir, name = get_str_date_fname(dt_now)
                    if not os.path.exists(dir):
                        os.mkdir(dir)
                    name = dir+"/"+name+"_"+max_label+"_"+str(max_prob)+".jpg"
                    print("Creating Images........." + name)
                    cv2.rectangle(_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.rectangle(_image, (x, y-40), (x+w, y), (0, 255, 0), -2)
                    cv2.putText(_image, max_label, (x, y-10), cv2.FONT_HERSHEY_COMPLEX,
                                0.75, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.imwrite(name, _image)
                last_publish = dt_now
                if DEBUG:
                    print(faces_dict)
                    # if diff_time.seconds > 1:
                    #     mqtt_publish("for_alice/peoples", faces_json)
                else:
                    if diff_time.seconds > 1:
                        mqtt_publish("for_alice/peoples", faces_json)

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
    MQTT_BROKER = Config['mqtt_broker']
    DEBUG = Config['debug']
    MQTT_PORT = Config['mqtt_port']
    frame_recognition_rate = Config['frame_recognition_rate']
    publish_min_dalay = Config['publish_min_dalay']
    mqttclient = connect_mqtt("", "", MQTT_BROKER, MQTT_PORT, client_id)
    frame_update_thread = threading.Thread(target=frame_update)
    frame_update_thread.start()
    face_recognition_thread = threading.Thread(target=face_recogintion)
    face_recognition_thread.start()
    while True:
        if outputFrame is None:
            time.sleep(0.2)
            continue
        if Config['show_camera']:
            image = outputFrame.copy()
            cv2.imshow('Camera', image)
        keyboard_input = cv2.waitKey(1)
        if keyboard_input == 27:
            break

    camera.release()
    cv2.destroyAllWindows()
