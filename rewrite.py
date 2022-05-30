import cv2


cap = cv2.VideoCapture('Demo Media/test.mp4')

filename = "out.mp4"
codec = cv2.VideoWriter_fourcc(*'XVID')
framerate = 29
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
resolution = (frame_width, frame_height)

# VideoOutPut = cv2.VideoWriter(filename, codec, framerate, resolution)
out = cv2.VideoWriter('output.avi',codec , 20.0, resolution)

if cap.isOpened():
    ret, frame = cap.read()

else:
    ret = False

while ret:
    ret, frame = cap.read()

    cv2.circle(frame, (200, 200), 80, (0, 255, 0), -1)

    out.write(frame)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
out.release()
cap.release()