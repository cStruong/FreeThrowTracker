from imutils.video import VideoStream
from collections import deque
import numpy as np
import argparse
import cv2
import imutils
import time
import math


# constructs args and parses them
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
                help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
                help="max buffer size")
args = vars(ap.parse_args())

# if a video path isnt supplies grab webcam reference
if not args.get("video", False):
    vs = VideoStream(src=0).start()

# otherwise grab a refeence to the video file
else:
    vs = cv2.VideoCapture(args["video"])
# allow camera or vid file to warm up
time.sleep(2.0)


#initiate font
# font = cv2.InitFont(cv2.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 3, 8)

#instantiate images
# hsv_img=cv2.CreateImage(cv2.GetSize(cv2.QueryFrame(self.capture)),8,3)
# threshold_img1 = cv2.CreateImage(cv2.GetSize(hsv_img),8,1)
# threshold_img1a = cv2.CreateImage(cv2.GetSize(hsv_img),8,1)
# threshold_img2 = cv2.CreateImage(cv2.GetSize(hsv_img),8,1)
# i=0
# writer=cv2.CreateVideoWriter('angle_tracking.avi',cv2.CV_FOURCC('M','J','P','G'),30,cv2.GetSize(hsv_img),1)


#blue ball
hsvLower1 = (62, 49, 51)
hsvUpper1 = (112, 253, 255)
#green pillow for now possibly rim later
hsvLower2 = (25, 84, 56)
hsvUpper2 = (78, 255, 255)

fontface = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 1
fontcolor = (0, 0, 0)
thickness = 3

pts = deque(maxlen=args["buffer"])

while True:

#capture the image from the cam
# img=cv2.QueryFrame(self.capture)
    frame = vs.read()
    # handle the frame from VideoCapture or VideoStream
    frame = frame[1] if args.get("video", False) else frame

    # if we are viewing a video and we did not grab a frame,
    # then we have reached the end of the video
    if frame is None:
        break


    #convert the image to HSV
    # cv2.CvtColor(img,hsv_img,cv2.CV_BGR2HSV)

    # resize the frame, blur it, and convert it to the HSV/RGB color space
    frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    rgb = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    #threshold the image to isolate two colors
    # cv2.InRangeS(hsv_img,(165,145,100),(250,210,160),threshold_img1) #red
    # cv2.InRangeS(hsv_img,(0,145,100),(10,210,160),threshold_img1a) #red again
    # cv2.Add(threshold_img1,threshold_img1a,threshold_img1) #this is combining the two limits for red
    # cv2.InRangeS(hsv_img,(105,180,40),(120,260,100),threshold_img2) #blue

    ball_frame_thresh = cv2.inRange(hsv, hsvLower1,hsvUpper1)
    hoop_frame_thresh = cv2.inRange(hsv, hsvLower2,hsvUpper2)


    #determine the moments of the two objects
    # threshold_img1=cv2.GetMat(threshold_img1)
    # threshold_img2=cv2.GetMat(threshold_img2)
    moments1=cv2.moments(ball_frame_thresh,0)
    moments2=cv2.moments(hoop_frame_thresh,0)
    # area1=cv2.GetCentralMoment(moments1,0,0)
    # area2=cv2.GetCentralMoment(moments2,0,0)
    area1 = moments1['m00']
    area2 = moments2['m00']

    #initialize x and y
    x1,y1,x2,y2=(1,2,3,4)
    coord_list=[x1,y1,x2,y2]
    for x in coord_list:
        x=0

    #there can be noise in the video so ignore objects with small areas
    if (area1 >200000):
        #x and y coordinates of the center of the object is found by dividing the 1,0 and 0,1 moments by the area
        # x1=int(cv2.GetSpatialMoment(moments1,1,0)/area1)
        # y1=int(cv2.GetSpatialMoment(moments1,0,1)/area1)
        x1 = moments1['m10'] / area1
        y1 = moments1['m01'] / area1

        #draw circle
        cv2.circle(frame, (int(x1), int(y1)), 2, (0,255,0), 20)

        #write x and y position
        # cv2.putText(frame,str(x1)+','+str(y1),(x1,y1+20),font, 255) #Draw the text
        cv2.putText(frame, str(round(x1, 1)) + ',' + str(round(y1, 1)), (int(x1), int(y1) + 20),
                    fontface, fontscale, fontcolor, thickness)

    if (area2 >200000):
        #x and y coordinates of the center of the object is found by dividing the 1,0 and 0,1 moments by the area
        # x2=int(cv2.GetSpatialMoment(moments2,1,0)/area2)
        # y2=int(cv2.GetSpatialMoment(moments2,0,1)/area2)
        x2 = moments2['m10'] / area2
        y2 = moments2['m01'] / area2

        #draw circle
        cv2.circle(frame,(int(x2),int(y2)),2,(0,255,0),20)
        # cv2.putText(frame,str(x2)+','+str(y2),(x2,y2+20),font, 255) #Draw the text
        cv2.putText(frame, str(round(x2, 1)) + ',' + str(round(y2, 1)), (int(x2), int(y2) + 20),
                    fontface, fontscale, fontcolor, thickness)
        # draws line between object
        cv2.line(frame,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),4)

        #draw line and angle
        # draws horizontal line
        cv2.line(frame,(int(x1),int(y1)),(frame.shape[1],int(y1)),(255,255,255,255),4)
        x1=float(x1)
        y1=float(y1)
        x2=float(x2)
        y2=float(y2)
        angle = int(math.atan((y1-y2)/(x2-x1))*180/math.pi)
        cv2.putText(frame,str(angle),(int(x1)+50, int((int(y2)+int(int(y1))/2))),
                    fontface, fontscale, fontcolor, thickness)

    #cv2.WriteFrame(writer,img)

    #display frames to users
    # cv2.ShowImage('Target',img)
    # cv2.ShowImage('Threshold1',threshold_img1)
    # cv2.ShowImage('Threshold2',threshold_img2)
    # cv2.ShowImage('hsv',hsv_img)

# # update the points queue
#     pts.appendleft(area1)
#
#     # loop over the set of tracked points
#     for i in range(1, len(pts)):
#         # if either of the tracked points are None, ignore them
#         if pts[i-1] is None or pts[i] is None:
#             continue
#
#         # otherwise, compute the thickness of the line and draw the connecting lines
#         thickness1 = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
#         cv2.line(frame, pts[i - 1], pts[i], (44, 255, 20), thickness1)

    cv2.imshow('Frame', frame)
    # Listen for ESC or ENTER key
    key = cv2.waitKey(1) & 0xFF

    # if the 'q' key is prssed, stop the loop/program
    if key == ord("q"):
        break

# if not using video file, stop the camera stream
if not args.get("video", False):
    vs.stop()

# else release camera
else:
    vs.release()

# close all windows
cv2.destroyAllWindows()