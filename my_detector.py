from imutils.video import VideoStream
import playsound
import argparse
import numpy as np
import cv2
import time
from imutils import face_utils
import dlib
from threading import Thread
from scipy.spatial import distance as dist
from PIL import Image
from PIL import ImageEnhance

ap=argparse.ArgumentParser()
ap.add_argument("-p","--shape-predictor",type=str,default="shape_predictor_68_face_landmarks.dat",help="hepls to find shape of face")
ap.add_argument("-w","--webcam",type=int,default=0,help="index to web camera")
ap.add_argument("-a","--sound",type=str,default="alarm.wav",help="play required sound")
args=vars(ap.parse_args())

def sound_alarm(path):
    playsound.playsound(path)
def eye_aspect_ratio(eye):
    A=dist.euclidean(eye[1],eye[5])
    B=dist.euclidean(eye[2],eye[4])
    C=dist.euclidean(eye[0],eye[3])
    ear=(A+B)/(2*C)
    return ear
EYE_ASPECT_RATIO=0.3
NO_OF_FRAMES=15
COUNTER=0
ALARM=False
print("[INFO] loading facial mark predictor..... ..")
detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor(args["shape_predictor"])
(lStart,lEnd)=face_utils.FACIAL_LANDMARKS_IDXS ["left_eye"]
(rStart,rEnd)=face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
print("[INFO] loading videostream ..... ..")
vs=VideoStream(src=args["webcam"]).start()
time.sleep(1.0)
while True:
    frame=vs.read()
    frame=cv2.resize(frame,(450,300))
        #a = np.double(frame)
        #b = a + 10
        #frame= np.uint8(b)
    
    #frame = Image.open(frame)
    enhancer_object = ImageEnhance.Brightness(frame)
    frame= enhancer_object.enhance(1.7)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #cv2.imshow("gray scaling",gray)
    rects=detector(gray,0)
    for rect in rects:
        shape=predictor(gray,rect)
        shape=face_utils.shape_to_np(shape)
        leftEye=shape[lStart:lEnd]
        rightEye=shape[rStart:rEnd]
        leftEAR=eye_aspect_ratio(leftEye)
        rightEAR=eye_aspect_ratio(rightEye)
        ear=(leftEAR+rightEAR)/2.0
        leftEyeHull=cv2.convexHull(leftEye)
        rightEyeHull=cv2.convexHull(rightEye)
        cv2.drawContours(frame,[leftEyeHull],-1,(0,255,0),1)
        cv2.drawContours(frame,[rightEyeHull],-1,(0,255,0),1)
        if ear<EYE_ASPECT_RATIO:
            COUNTER+=1
            if COUNTER>NO_OF_FRAMES:
                if not ALARM:
                    ALARM=True
                    if args["sound"] !="":
                        t=Thread(target=sound_alarm,args=(args['sound'],))
                        t.daemon=True
                        t.start()
                        cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        else:
            COUNTER=0
            ALARM=False
            cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
            
    cv2.imshow("frame",frame)

    key=cv2.waitKey(1) & 0xff
    if key==ord("q"):
        break



