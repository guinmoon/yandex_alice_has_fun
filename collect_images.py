import cv2
import os
import threading
import time
import argparse


parser = argparse.ArgumentParser(
    prog='CollectImages',
    description='',
    epilog='')
# option that takes a value
parser.add_argument('-i', '--input', default=0)
parser.add_argument('-n', '--name', default="test")
args = parser.parse_args()
input_source = args.input
if input_source.isnumeric():
    input_source = int(input_source)
video = cv2.VideoCapture(input_source)

# video = cv2.VideoCapture("raw/IMG_2404.MOV")
# video = cv2.VideoCapture(0)
# video = cv2.VideoCapture("rtsp://192.168.1.86:8554/unicast")

front_facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
profile_facedetect = cv2.CascadeClassifier('haarcascade_profileface.xml')


# nameID = str(input("Enter Your Name: ")).lower()
nameID = args.name

path = 'detected/'+nameID

isExist = os.path.exists(path)

if isExist:
    print("Name Already Taken")
    # nameID = str(input("Enter Your Name Again: "))
else:
    os.makedirs(path)

frame_rate = 10000
sleep_delay = 1/frame_rate
outputFrame = None
cur_x = 0
cur_y = 0
cur_w = 0
cur_h = 0


def frame_update():
    global outputFrame, video, frame_updated
    # video = cv2.VideoCapture(rtsp_link)
    while True:
        ret, frame = video.read()
        outputFrame = frame.copy()
        frame_updated = True
        # time.sleep(sleep_delay)


def face_detect():
    global cur_x, cur_y, cur_w, cur_h
    count = 0
    while True:
        if outputFrame is None:
            time.sleep(0.1)
            continue
        frame = outputFrame.copy()
        faces = front_facedetect.detectMultiScale(frame, 1.3, 5)
        for x, y, w, h in faces:
            cur_x, cur_y, cur_w, cur_h = x, y, w, h
            count = count+1
            name = './detected/'+nameID+'/f' + str(count) + '.jpg'
            print("Creating Images........." + name)
            cv2.imwrite(name, frame[y:y+h, x:x+w])
        faces = profile_facedetect.detectMultiScale(frame, 1.3, 5)
        for x, y, w, h in faces:
            cur_x, cur_y, cur_w, cur_h = x, y, w, h
            count = count+1
            name = './detected/'+nameID+'/p' + str(count) + '.jpg'
            print("Creating Images........." + name)
            cv2.imwrite(name, frame[y:y+h, x:x+w])
        if count > 500:
            break
        time.sleep(0.1)


if __name__ == '__main__':
    frame_update_thread = threading.Thread(target=frame_update)
    frame_update_thread.start()
    face_detect_thread = threading.Thread(target=face_detect)
    face_detect_thread.start()
    while True:
        if outputFrame is None:
            time.sleep(0.1)
            continue
        cv2.rectangle(outputFrame, (cur_x, cur_y),
                      (cur_x+cur_w, cur_y+cur_h), (0, 255, 0), 3)
        cv2.imshow("WindowFrame", outputFrame)
        cv2.waitKey(1)
    video.release()
    cv2.destroyAllWindows()
