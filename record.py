import cv2

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
camera = cv2.VideoCapture(0);
Wcam, Hcam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, Wcam)
cap.set(4, Hcam)
id = input('Input ID : ')
a = 0
while True:
    #1 find hand landmark
    a += 1
    success, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5);
    for (x, y, w, h) in faces:
        cv2.imwrite('database/user.'+str(id)+'.'+str(a)+'.jpg',gray[y:y+h, x:x+w])
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    if (a>29):
        break
    cv2.imshow("Image", img)
    cv2.waitKey(1)

