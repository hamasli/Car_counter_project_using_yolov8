import numpy as np;
import cv2;
from ultralytics import YOLO
import cvzone
import math;
from sort import*

#for video
cap=cv2.VideoCapture('../Videos/cars.mp4');
#1200 720
#for webcam
# cap=cv2.VideoCapture(0);
# cap.set(3,1200);
# cap.set(4,720);

#loading model
model=YOLO('../yolov8-weights/yolov8n.pt');

#loading class names
# Read the coco.names file and extract class names into a list
classfile='coco.names';
classNames=[];
with open(classfile,'rt') as f:
    classNames=f.read().rstrip('\n').split('\n');
print(classNames);

#import mask
mask=cv2.imread('mask.png');

#crating sort instance
tracker=Sort(max_age=20,min_hits=3,iou_threshold=0.3);
limits=[400,297,673,297];



totalCount=[];
while True:
    success,image=cap.read();
    #here we are overlaying the mask on the video to get extract data of particular region
    imgRegion=cv2.bitwise_and(image,mask);
    # img = cvzone.overlayPNG(image, imgGraphics, (0, 0))
    results=model(imgRegion,stream=True);
    detection=np.empty((0,5));
    for r in results:
        boxes=r.boxes;
        for box in boxes:
            #bounding box
            #getting coordinates of every box;
            # we can take x,y,w,h or x1,x2,y2,y2
            x1,y1,x2,y2=box.xyxy[0];
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2);
            # cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 3);
            # print(x1, y1, x2, y2);
            w,h=x2-x1,y2-y1;
            bbox=int(x1),int(y1),int(w),int(h);
            # cvzone.cornerRect(image,bbox,l=9,rt=5);
            #confidence
            conf=math.ceil((box.conf[0]*100))/100;
            # print(conf);
            cls=int(box.cls[0]);
            currentClass=classNames[cls];
            if currentClass=="car" or currentClass=="bus" or currentClass=="truck" or currentClass=="motorbike"  and conf>0.3:
                # cvzone.putTextRect(image, f'{currentClass} {conf}', (max(0, x1), max(35, y1)), offset=5, thickness=2,
                #                    scale=1.5);
                # cvzone.cornerRect(image,(x1,y1,w,h),l=9,rt=5);
                currentArray=np.array(([x1,y1,x2,y2,conf]));
                detection=np.vstack((detection,currentArray));


    resultsTracker=tracker.update(detection);
    cv2.line(image,(limits[0],limits[1]),(limits[2],limits[3]),(0,0,255),5);
    for result in resultsTracker:
        x1,y1,x2,y2,Id=result;
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2);
        print(result);
        w, h = x2 - x1, y2 - y1;

        cvzone.cornerRect(image,(x1,y1,w,h),l=9,rt=2,colorR=(255,0,255));
        cvzone.putTextRect(image, f'{int(Id)}', (max(0, x1), max(35, y1)), offset=10, thickness=3,
                           scale=2);
        #formula for getting the center of object using x,y,w,h of bounding box to detect center of object.
        cx,cy=x1+w//2,y1+h//2;
        cv2.circle(image,(cx,cy),5,(255,0,255),cv2.FILLED);
        limits = [400, 297, 673, 297];
        #now we are checking that each object crossed the line or not. using limits x1,x2,y1,y1
        if limits[0]<cx<limits[2] and limits[1]-15<cy<limits[1]+15:
            if totalCount.count(Id)==0:
                totalCount.append(Id)
                cv2.line(image, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5);



    cvzone.putTextRect(image, f'{len(totalCount)}', (50, 50));
    cv2.imshow("image",image);
    # cv2.imshow("ImageRegion",imgRegion);
    cv2.waitKey(0);
