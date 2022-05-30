import cv2
import logging
import boto3
from botocore.exceptions import ClientError


def upload_file(file_name, bucket, key=None):
  """Upload a file to an S3 bucket

  :param file_name: File to upload
  :param bucket: Bucket to upload to
  :param object_name: S3 object name. If not specified then file_name is used
  :return: True if file was uploaded, else False
  """

  # If S3 object_name was not specified, use file_name
  if key is None:
    key = file_name

  # Upload the file
  s3_client = boto3.client('s3')
  try:
    response = s3_client.put_object(file_name, bucket, key)
  except ClientError as e:
    logging.error(e)
    return False
  return True


# get video from folder and read it in
vid_path = "/Users/nwannw/Documents/AWS/Capstone/winnie_shooting_miss.mp4"
vidcap = cv2.VideoCapture(vid_path)
success,image = vidcap.read()
count = 0

#if vid read was a success
while success:
  #write image to folder
  upload_file(image,'custom-labels-console-us-east-1-a4ae15429b/frames_training')
  # cv2.imwrite( "/Users/nwannw/Documents/AWS/Capstone/long_frame%d.jpg" % count, image) # save frame as JPEG file
  success,image = vidcap.read()
  print('Wrote a new frame: ', success)
  count += 1
  # if count > 100:
  #   success = False

