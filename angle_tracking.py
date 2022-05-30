#package imports

from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time

# print object coordinates
# def mapObjectPosition (frame, x1, y1):
#     cv2.PutText(frame, str(x1) + ',' + str(y1), (x1, y1 + 20), 255)  # Draw the text

# constructs args and parses them
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
                help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
                help="max buffer size")
args = vars(ap.parse_args())

# defines color limits of the basketball in RGB color code
# then initializes list of tracked color points
rgb_ball_color_lower = (95, 46, 44)
rgb_ball_color_upper = (101, 55, 68)
#blue ball
hsvLower1 = (62, 49, 51)
hsvUpper1 = (112, 253, 255)
#red pillow for now possibly rim later
hsvLower2 = (62, 49, 51)
hsvUpper2 = (112, 253, 255)

pts = deque(maxlen=args["buffer"])

# if a video path isnt supplies grab webcam reference
if not args.get("video", False):
    vs = VideoStream(src=0).start()

# otherwise grab a refeence to the video file
else:
    vs = cv2.VideoCapture(args["video"])
# allow camera or vid file to warm up
time.sleep(2.0)

while True:
    # grab current frame
    frame = vs.read()

    # handle the frame from VideoCapture or VideoStream
    frame = frame[1] if args.get("video", False) else frame

    # if we are viewing a video and we did not grab a frame,
    # then we have reached the end of the video
    if frame is None:
        break

    # resize the frame, blur it, and convert it to the RGB color space
    frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    rgb = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # construct a mask1 for the ball color
    # then perform a series of dilations and erosions to remove any small blobs left in the mask1
    mask1 = cv2.inRange(hsv, hsvLower1, hsvUpper1)
    mask1 = cv2.erode(mask1, None, iterations=2)
    mask1 = cv2.dilate(mask1, None, iterations=2)

    # construct a mask2 for the ball color
    # then perform a series of dilations and erosions to remove any small blobs left in the mask2
    mask2 = cv2.inRange(hsv, hsvLower2, hsvUpper2)
    mask2 = cv2.erode(mask1, None, iterations=2)
    mask2 = cv2.dilate(mask1, None, iterations=2)

    # find countours in the mask1/2 and initialize the current
    # (x, y) center of the balls
    cnts1 = cv2.findContours(mask1.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts1 = imutils.grab_contours(cnts1)

    cnts2 = cv2.findContours(mask1.copy(), cv2.RETR_EXTERNAL,
                             cv2.CHAIN_APPROX_SIMPLE)
    cnts2 = imutils.grab_contours(cnts2)
    center = None
    center2 = None

    # only proceed if at least one countour was found
    if len(cnts1) > 0 & len(cnts2) > 0:
        # find largest contour in the mask1
        # use it compute min enclosing circle and centroid
        c1 = max(cnts1, key=cv2.contourArea)
        ((x1, y1), radius) = cv2.minEnclosingCircle(c1)
        M = cv2.moments(c1)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # find largest contour in the mask2
        # use it compute min enclosing circle and centroid
        c2 = max(cnts2, key=cv2.contourArea)
        ((x2, y2), radius2) = cv2.minEnclosingCircle(c2)
        M2 = cv2.moments(c2)
        center2 = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # only proceed if the radius meets a minimum size of 10
        if radius > 10:

            # draw circle and centroid on the frame then update list of tracked points
            cv2.circle(frame, (int(x1), int(y1)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, (int(x2), int(y2)), int(radius2), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            cv2.circle(frame, center2, 5, (0, 0, 255), -1)
            
            # Draw the text
            fontface = cv2.FONT_HERSHEY_SIMPLEX
            fontscale = 1
            fontcolor = (255, 255, 255)
            cv2.putText(frame,str(round(x1, 1))+','+str(round(y1, 1)), (int(x1), int(y1)+20),
                        fontface, fontscale, fontcolor)
            cv2.putText(frame, str(round(x2, 1)) + ',' + str(round(y2, 1)), (int(x2), int(y2) + 20),
                        fontface, fontscale, fontcolor)

            # start_x = 0
            # start_y = int(y)
            # end_x = int(x)+200
            # end_y = int(y)

            # #draw horizontal line
            # #cv2.line(frame, (start_x, start_y), (int(frame.shape[1] / 2), frame.shape[0]), (0, 0, 0), 5)
            # cv2.line(frame, (start_x,start_y),(end_x,end_y), (0,0,0), 3)

    # update the points queue
    pts.appendleft(center)

    # loop over the set of tracked points
    for i in range(1, len(pts)):
        # if either of the tracked points are None, ignore them
        if pts[i-1] is None or pts[i] is None:
            continue

        # otherwise, compute the thickness of the line and draw the connecting lines
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (44, 255, 20), thickness)

    # show the frame to our screen
    cv2.imshow("Frame", frame)
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