import cv2
import os # handling directories
alg="haarcascade_frontalface_default.xml"# importing algorithm
har=cv2.CascadeClassifier(alg)# reading & storing the algorithm in a variable
cam=cv2.VideoCapture(0)
dataset="dataset"
name="sociallion"
path=os.path.join(dataset,name)
if not os.path.isdir(path):
    os.makedirs(path)#creates a new directory for the sequence of folder
# resizing image using cv2
(width,height)=(150,150)
count=0
n=int(input("enter number of pictures to be taken"))
while (count<=n):
    _,img=cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=har.detectMultiScale(gray,1.3,4)#detecting the face and scakling the image
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        only_face=gray[y:y+h,x:x+w]#only used to process and store the face part from image
        res=cv2.resize(only_face,(width,height))
        cv2.imwrite("%s/%s.jpg" %(path,count),res)# having thwo %s , 1%s-path,2.%s-represents the number(count)
        count+=1
        print(count)
    cv2.imshow("face detection",img)
    key=cv2.waitKey(10)
    if key == 27:# press escape button to exit
        break
cam.release()
cv2.destroyAllWindows()
