import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy

Wcam, Hcam = 640, 480
frameR = 100

cap = cv2.VideoCapture(0)
cap.set(3,Wcam)
cap.set(4, Hcam)
pTime = 0
detector = htm.handDetector(maxHands=1)
wScr, hScr = autopy.screen.size()
smoothening = 5
# print(wScr, hScr)

plocX, plocY = 0, 0
clocX, clocY = 0, 0


while True:
    #1 find hand landmark
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)
    # print(lmList)
    #2 get index value fingers
    if len(lmList)!=0:
        x1 , y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # print(x1, y1, x2, y2)
        #3 check finger up
        fingers = detector.fingersUp()
        # print(fingers)

        #4 moving mode
        if fingers[1] == 1 and fingers [2] == 0:

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
        if fingers[1] == 1 and fingers[2] == 1:
            length, img, lineInfo = detector.findDistance(8, 12, img)
            print(length)
            if length < 10:
                cv2.circle(img,(lineInfo[4], lineInfo[5]),
                           15, (0, 255, 255), cv2.FILLED)
            autopy.mouse.click()
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