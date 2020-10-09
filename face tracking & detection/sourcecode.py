import cv2
alg="haarcascade_frontalface_default.xml"# importing algorithm
har=cv2.CascadeClassifier(alg)# reading & storing the algorithm in a variable
cam=cv2.VideoCapture(0)
while True:
    _,img=cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=har.detectMultiScale(gray,1.3,4)#detecting the face and scakling the image
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow("face detection",img)
    key=cv2.waitKey(10)
    if key == 27:# press escape button to exit
        break
cam.release()
cv2.destroyAllWindows()
