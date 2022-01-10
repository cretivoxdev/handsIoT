import math
import pyttsx3

import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

import face_recognition
from PIL import Image

engine = pyttsx3.init()



Wcam, Hcam = 640, 480
frameR = 100
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
# camera = cv2.VideoCapture(0);

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('database/training.xml')
cap = cv2.VideoCapture(0)
cap.set(3, Wcam)
cap.set(4, Hcam)
pTime = 0
detector = htm.handDetector(maxHands=1)
wScr, hScr = autopy.screen.size()
smoothening = 5
# print(wScr, hScr)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
# print(volume.GetVolumeRange())
volRange = volume.GetVolumeRange()
volume.SetMasterVolumeLevel(0, None)
minVol = volRange[0]
maxVol = volRange[1]
# vol = 0
# volBar = 400
# volPer = 0

plocX, plocY = 0, 0
clocX, clocY = 0, 0


while True:
    #1 find hand landmark
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, conf = recognizer.predict(gray[y:y+h, x:x+w])
        if (id == 1) :
            id = "rohim"
        if (id == 2) :
            id = "agis"
        if (id == 3):
            id = "Bebed"

        say1 = "Hello " + id


        cv2.putText(img, str(id), (x+40,y-10), cv2.FONT_HERSHEY_DUPLEX,1,(0,255))
        cv2.putText(img, say1, (50, 80),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)


    # imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    # imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    #
    # facesCurFrame = face_recognition.face_locations(imgS)
    # encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
    # for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
    #     y1, x2, y2, x1 = faceLoc
    #     y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
    #     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #     #cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
    #print(lmList)
    #2 get index value fingers
    if len(lmList)!=0:
        x0 , y0 = lmList[4][1:]
        x1 , y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]


        # print(x1, y1, x2, y2)
        #3 check finger up
        fingers = detector.fingersUp()
        # print(fingers)

        #4 moving mode
        if fingers[1] == 1 and fingers [2] == 0 and fingers[0] == 0:

            #5 convert coordinates
            cv2.rectangle(img,(frameR,frameR), (Wcam-frameR, Hcam - frameR),
                          (255, 0, 255), 2)
            x3 = np.interp(x1, (frameR, Wcam-frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, Hcam-frameR), (0, hScr))

            #6 smoothen value
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening
            #7 move mouse
            autopy.mouse.move(wScr - clocX, clocY)
            cv2.circle(img,(x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX , clocY
        #8 both index finger
        if fingers[1] == 1 and fingers[2] == 1 and fingers[0] == 0 and fingers[3] == 0 and fingers[4] == 0:
            length, img, lineInfo = detector.findDistance(8, 12, img)
            #cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            #print(length)
            if length < 35.0:
                cv2.circle(img,(lineInfo[4], lineInfo[5]),
                           15, (0, 255, 255), cv2.FILLED)
                autopy.mouse.click()
        if fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0 :
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x0, y0), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img,(x1, y1), (x0, y0), (255, 0, 255), 3)
            lengthsound = math.hypot(x1 - x0, y1 - y0)
            lengthsound += 200
            vol = np.interp(lengthsound, [20, 300], [minVol, maxVol])
            print(int(lengthsound),vol)
            volume.SetMasterVolumeLevel(vol, None)
            #print(lengthsound)
    #9
    #10
    #11 FRAME RATE
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img,str(int(fps)), (20,50),cv2.FONT_HERSHEY_PLAIN,3,
    (255,0,0), 3)
    #12 DISPLAY
    cv2.imshow("Image", img)
    cv2.waitKey(1)