#package imports

from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time

# print object coordinates
def mapObjectPosition (x, y):
    # x_lst = []
    # y_lst = []
    # x_lst.append(x)
    # y_lst.append(y)
    # xmax = max(x_lst)
    # ymax = max(y_lst)
    # xmin = min(x_lst)
    # ymin = min(y_lst)
    print ("[INFO] Object Center coordenates at X0 = {0} and Y0 =  {1}".format(x, y))

def get_max_x (x):
    x_lst = []
    x_lst.append(x)
    xmax = max(x_lst)
    return xmax
def get_max_y (y):
    y_lst = []
    y_lst.append(y)
    ymax = max(y_lst)
    return ymax

def get_min_x (x):
    x_lst = []
    x_lst.append(x)
    xmin = min(x_lst)
    return xmin

def get_min_y (y):
    y_lst = []
    y_lst.append(y)
    ymin = min(y_lst)
    return ymin



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
hsvLower = (62, 49, 51)
hsvUpper = (112, 253, 255)
real_color_ball_low = (0,117,0)
real_color_ball_high = (25, 163, 170)
# hsvLower = (30, 46, 45)
# hsvUpper = (92, 166, 98)
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
    frame = imutils.resize(frame, width=600, height=350)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    rgb = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # construct a mask for the ball color
    # then perform a series of dilations and erosions to remove any small blobs left in the mask
    mask = cv2.inRange(hsv, hsvLower, hsvUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # find countours in the mask and initialize the current
    # (x, y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None

    # only proceed if at least one countour was found
    if len(cnts) > 0:
        # find largest contour in the mask
        # use it compute min enclosing circle and centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # only proceed if the radius meets a minimum size of 10
        if radius > 10:
            # draw circle and centroid on the frame then update list of tracked points
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

            mapObjectPosition(int(x), int(y))


            start_x = 0
            start_y = int(y)
            end_x = int(x)+200
            end_y = int(y)

            #draw horizontal line if the position is below halfway through the screen
            if int(y) > 200 & int(y) < 210:
                cv2.line(frame, (int(x), int(y)), (frame.shape[1], int(y)), (0, 0, 0), 5)

            # cv2.line(frame, (start_x,start_y),(end_x,end_y), (0,0,0), 3)

            # ymax = get_max_y(int(y))
            # ymin = get_min_y(int(y))
            # xmin = get_min_x(int(x))
            # xmax = get_max_x(int(x))
            #
            # cv2.line(frame, (int(xmin),int(ymin)), (int(xmax),int(ymax)), (0, 0, 0), thickness=2)

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