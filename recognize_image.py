import boto3
import io
from PIL import Image, ImageDraw, ExifTags, ImageColor, ImageFont
import cv2
import json


def show_custom_labels(model, photo, min_confidence):

    rekognition = boto3.client('rekognition')

    # # Load image from S3 bucket
    # s3_connection = boto3.resource('s3')
    #
    # s3_object = s3_connection.Object(bucket, photo)
    # s3_response = s3_object.get()
    #
    # stream = io.BytesIO(s3_response['Body'].read())

    # read in photo
    frame = cv2.imread(photo)

    # if has frame then resize and encode it as jpg
    hasFrame, imageBytes = cv2.imencode(".jpg", frame)

    # run image against model
    response = rekognition.detect_custom_labels(
        Image={
            'Bytes': imageBytes.tobytes(),
        },
        ProjectVersionArn=model,
        MinConfidence=min_confidence)

    # #convert from cv2 to PIL
    # image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # imgWidth, imgHeight  = image_pil.size
    # draw = ImageDraw.Draw(image_pil)

    imgHeight, imgWidth, imgChannel = frame.shape

    # calculate and display bounding boxes for each detected custom label
    print('Detected custom labels for ' + photo)
    for customLabel in response['CustomLabels']:
        if customLabel['Name'] == 'basket' or customLabel['Name'] == 'ball':
            print('Label ' + str(customLabel['Name']))
            print('Confidence ' + str(customLabel['Confidence']))
            if 'Geometry' in customLabel:
                box = customLabel['Geometry']['BoundingBox']
                left = imgWidth * box['Left']
                top = imgHeight * box['Top']
                width = imgWidth * box['Width']
                height = imgHeight * box['Height']
                x = left + (width / 2)
                y = top + (height / 2)


                # fnt = ImageFont.truetype('/Library/Fonts/Arial.ttf', 50)
                # draw.text((left, top), customLabel['Name'], fill='#00d400', font=fnt)

                print('Left: ' + '{0:.0f}'.format(left))
                print('Top: ' + '{0:.0f}'.format(top))
                print('Label Width: ' + "{0:.0f}".format(width))
                print('Label Height: ' + "{0:.0f}".format(height))
                print('X: {}, Y: {}'.format(x, y))

                points = (
                    (left, top),
                    (left + width, top),
                    (left + width, top + height),
                    (left, top))

                cv2.rectangle(frame,(int(left),int(top)), (int(left) + int(width), int(top) + int(height)), color=(0,0,0))

                # draw.line(points, fil0l='#00d400', width=5)
                print(points)

                #shooters window
                cv2.rectangle(frame, (1400,400), (1500,500), color=(255,0,0), thickness=2)


    cv2.imshow('frame', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return len(response['CustomLabels'])


def main():
    bucket = "custom-labels-console-us-east-1-a4ae15429b"
    photo = "/Users/nwannw/Documents/AWS/Capstone/frame0.jpg"
    model = 'arn:aws:rekognition:us-east-1:333527701433:project/winnie_test_training/version/' \
            'winnie_test_training.2020-04-30T22.35.42/1588300542347'
    min_confidence = 99

    label_count = show_custom_labels(model, photo, min_confidence)
    print("Custom labels detected: " + str(label_count))


if __name__ == "__main__":
    main()
