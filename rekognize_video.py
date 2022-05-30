from _collections import deque
import boto3
import math
import cv2
from datetime import datetime


def analyzeVideo(video, model, min_confidence):

    global left, top, height, width, x_basket, y_basket, basket_box, left_basket, x_ball, width_basket, top_basket, y_ball, height_basket
    rekognition = boto3.client('rekognition')
    vid = cv2.VideoCapture(video)
    fps = vid.get(cv2.CAP_PROP_FPS)  # frame rate
    frame_width = int(vid.get(3))
    frame_height = int(vid.get(4))
    resolution = (frame_width, frame_height)
    pts = deque()
    fontface = cv2.FONT_HERSHEY_SIMPLEX
    fontscale = 1
    fontcolor = (0, 0, 0)
    thickness = 3
    now = datetime.now()
    filename = 'Demo Media//Outputs//ouputvid-{}.avi'.format(now)
    made_shots = 0
    shots_taken = 0

    # ideal arc tracing variables -chris
    all_basket_x_position = []
    all_basket_y_position = []
    all_ball_x_position = []
    all_ball_y_position = []
    calc_angle = 0
    ideal_angle = 51.5  # should be whatever input from the frontend is

    # Define the codec and create VideoWriter object.The output is stored in 'outputs-"date and time".mp4' file.
    # Define the fps to be equal to 10. Also frame size is passed.
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filename, fourcc, fps, resolution)

    while vid.isOpened():
        frameId = vid.get(1)  # current frame number
        print("Processing frame id: {}".format(frameId))
        print('frame_Rate: {}'.format(fps))

        # capture frame by frame
        ret, frame = vid.read()

        # if it doesnt have a frame then break
        if not ret:
            break

        # if has frame then encode it as jpg
        hasFrame, imageBytes = cv2.imencode(".jpg", frame)

        # run each frame against model
        if (hasFrame):

            response = rekognition.detect_custom_labels(
                Image={
                    'Bytes': imageBytes.tobytes(),
                },
                ProjectVersionArn=model,
                MinConfidence=min_confidence
            )
            # Get image shape
            imgHeight, imgWidth, imgChannel  = frame.shape

            # calculate bounding boxes for each detected custom label
            for customLabel in response['CustomLabels']:
                # only looking for basket
                if customLabel['Name'] == 'basket':
                    left_basket = imgWidth * customLabel['Geometry']['BoundingBox']['Left']
                    top_basket = imgHeight * customLabel['Geometry']['BoundingBox']['Top']
                    width_basket = imgWidth * customLabel['Geometry']['BoundingBox']['Width']
                    height_basket = imgHeight * customLabel['Geometry']['BoundingBox']['Height']
                    x_basket = left_basket + (width_basket / 2)
                    y_basket = top_basket + (height_basket / 2)
                    print('Label {}'.format(customLabel['Name']))
                    print('x coor: {}, y coor: {}'.format(x_basket, y_basket))

                    # add basket position to array on top, so i can get coordinates from first frame of basket - chris
                    all_basket_x_position.append(x_basket)
                    all_basket_y_position.append(y_basket)

                    # draw bounding boxes around basket
                    basket_top_left = (int(left_basket), int(top_basket))
                    basket_bot_right = (int(left_basket) + int(width_basket), int(top_basket) + int(height_basket))
                    # basket_box = cv2.rectangle(frame, basket_top_left, basket_bot_right, color=(0, 0, 0), thickness=3)

                # only looking for the ball
                if customLabel['Name'] == 'ball':
                    if 'Geometry' in customLabel:
                        left = imgWidth * customLabel['Geometry']['BoundingBox']['Left']
                        top = imgHeight * customLabel['Geometry']['BoundingBox']['Top']
                        width = imgWidth * customLabel['Geometry']['BoundingBox']['Width']
                        height = imgHeight * customLabel['Geometry']['BoundingBox']['Height']
                        x_ball = left + (width/2)
                        y_ball = top + (height/2)
                        coords = (int(x_ball), int(y_ball))

                        # update the points queue
                        pts.appendleft(coords)

                        print('Label {}'.format(customLabel['Name']))
                        print('x coor: {}, y coor: {}'.format(x_ball, y_ball))

                        # add ball position to array on top, so i can get coordinates from first frame of ball - chris
                        all_ball_x_position.append(int(x_ball))
                        all_ball_y_position.append(int(y_ball))

                        # draw bounding boxes around image
                        im_top_left = (int(left), int(top))
                        im_bot_right = (int(left) + int(width), int(top) + int(height))
                        # cv2.rectangle(frame, im_top_left, im_bot_right,
                        #               color=(0, 0, 0), thickness=3)




        # loop over the set of tracked points if ball if above basket threshold
        # (hardcoded this--NEED TO REVISIT LATER)
        # pts[i][1] is "y" and it represents the position of the ball
        # pts[i][0] is "x" and it represents the position of the ball
        # if y < pts[-1][1] means it is above the first detection of the ball when being shot we want to trace it
        # or if x = to the basket's xcoord and above the baskets ycoord keep tracing
            for i in range(1, len(pts)):
                if pts[i][0] > x_basket:
            # if either of the tracked points are None, ignore them
                    if pts[i - 1] is None or pts[i] is None:
                        continue
                    # otherwise draw the connecting lines
                    cv2.line(frame, pts[i - 1], pts[i], (44, 255, 20), thickness)
                    # compute and draw launch angle
                    (x1, y1) = pts[-1]
                    (x2, y2) = pts[-2]
                    angle = int(math.atan((y1 - y2) / (x1 - x2)) * 180 / math.pi)
                    calc_angle = angle
                    cv2.putText(frame, str(angle), (int(x1-60), int(y1)),
                                fontface, fontscale, color=fontcolor, thickness=thickness)


        # ideal arc tracing portion - chris
        if len(all_ball_x_position) == 0:  # only start if there is a ball x position detected
            continue

        basket_x_position = all_basket_x_position[0]  # initial basket x position
        basket_y_position = all_basket_y_position[0]  # initial basket y position
        ball_initial_x_position = all_ball_x_position[0]  # initial ball x position
        ball_initial_y_position = all_ball_y_position[0]  # initial ball y position
        total_x_distance = (ball_initial_x_position - basket_x_position)  # TO DO: only works for current vid orientation
        total_y_distance = (ball_initial_y_position - basket_y_position)  # TO DO: only works for current vid orientation
        ratio = total_x_distance / 15  # ratio of pixels to feet
        gravity = -32.2 * ratio  # gravity in ft/s^2 is 32.2, this variable gets gravity in px/s^2
        print(gravity/2)
        first_half = total_y_distance - ((math.sin(math.radians(ideal_angle)) * total_x_distance) / math.cos(math.radians(ideal_angle)))  # first part of kinematic equation
        second_half = (gravity / 2) * (total_x_distance ** 2) / (math.cos(math.radians(ideal_angle)) ** 2)  # second part of kinematic equation
        initial_v = math.sqrt(second_half / first_half)  # initial velocity in px/s
        total_time = total_x_distance / (initial_v * math.cos(math.radians(ideal_angle)))  # total travel time of ball
        ideal_initial_velocity = (initial_v / ratio) / 3.281  # ideal initial velocity in m/s^2
        tracker_count = 50  # 50 is just how many points i decided to draw on the arc to connect. more tracker points = smoother curve
        time_increment = total_time / tracker_count

        ### To get actual velocity of ball being tracked ###
        if calc_angle == 0:
            continue

        actual_first_half = total_y_distance - ((math.sin(math.radians(calc_angle)) * total_x_distance) / math.cos(math.radians(calc_angle)))  # first part of kinematic equation
        actual_second_half = (gravity / 2) * (total_x_distance ** 2) / (math.cos(math.radians(calc_angle)) ** 2)  # second part of kinematic equation
        actual_initial_v = math.sqrt(actual_second_half / actual_first_half)
        actual_initial_velocity = (actual_initial_v / ratio) / 3.281  # actual initial velocity in m/s^2

        # test_ball_first_x = all_ball_x_position[3]
        # test_ball_first_y = all_ball_y_position[3]
        # test_total_x_distance = (ball_initial_x_position - test_ball_first_x)
        # test_total_y_distance = (ball_initial_y_position - test_ball_first_y)
        # test_actual_first_half = test_total_y_distance - ((math.sin(math.radians(calc_angle)) * test_total_x_distance) / math.cos(math.radians(calc_angle)))  # first part of kinematic equation
        # test_actual_second_half = (gravity / 2) * (test_total_x_distance ** 2) / (math.cos(math.radians(calc_angle)) ** 2)  # second part of kinematic equation
        # test_actual_initial_v = test_actual_second_half / test_actual_first_half
        # print(test_actual_first_half)
        # test_actual_initial_velocity = (test_actual_initial_v / ratio) / 3.281  # actual initial velocity in m/s^2






        ### end get actual velocity portion ###

        x_differential = [0]  # list of how much the position changes on the x coordinate as time goes on
        y_differential = [0]  # list of how much the position changes on the y coordinate as time goes on

        previous_time = 0
        while len(x_differential) < (tracker_count - 1):
            initial_v_x = initial_v * math.cos(math.radians(ideal_angle))  # initial velocity of x component
            initial_v_y = initial_v * math.sin(math.radians(ideal_angle))  # initial velocity of y component
            previous_time = previous_time + time_increment
            x_changer = initial_v_x * previous_time  # kinematic equation, initial pos 0 and acc is 0
            y_changer = (initial_v_y * previous_time) + ((gravity / 2) * (previous_time ** 2))  # kinematic equation but acc is gravity
            x_differential.append(x_changer)
            y_differential.append(y_changer)

        # iterate through the xdiff and ydiff array, use those values and subtract from initial position, then draw a line from prev point to current point
        counter = 0
        while counter < len(x_differential) - 1:
            cv2.line(frame, (int(ball_initial_x_position - x_differential[counter]), int(ball_initial_y_position - y_differential[counter])), (int(ball_initial_x_position - x_differential[counter + 1]), int(ball_initial_y_position - y_differential[counter + 1])), color=(255, 255, 0), thickness=3)
            counter = counter + 1

        # Shot counter portion
        # x_shot, y_shot = pts[0][0], pts[0][1]
        # if ((left_basket + width_basket) > x_shot > left_basket) and (top_basket < y_shot < (top_basket + height_basket)):
        #     made_shots += 1
        #     shots_taken += 1
        #     cv2.putText(frame, "Shot Made:{}   Shot Taken:{}".format(made_shots, shots_taken), (100, 800),
        #                 fontface, fontscale, color=fontcolor, thickness=thickness)


        # write the video to a file and show the video
        out.write(frame)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(ideal_initial_velocity)
    print(actual_initial_velocity)

    cv2.destroyAllWindows()
    vid.release()
    out.release()


def main():
    #video = "/Users/nwannw/Documents/AWS/Capstone/winnie_shootingtest.mp4"
    video = 'Demo Media/winnie_shooting.mp4'
    model = 'arn:aws:rekognition:us-east-1:333527701433:project/winnie_test_training/version/' \
            'winnie_test_training.2020-04-30T22.35.42/1588300542347'

    new_model = 'arn:aws:rekognition:us-east-1:333527701433:project/capstone_try3/version/' \
                'capstone_try3.2020-07-08T11.36.51/1594222611541'


    min_confidence = 95

    analyzeVideo(video, new_model, min_confidence)


if __name__ == "__main__":
    main()
