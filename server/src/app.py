# pylint: disable=C0103
from io import BytesIO
import sys
import os
import base64
from flask import Flask, Response, request, jsonify
from google.cloud import storage
import numpy as np
import tensorflow as tf
import keras
from keras.models import load_model
from keras.utils import load_img, img_to_array
from PIL.Image import NONE
from keras.applications.vgg19 import VGG19
import math 
import uuid
import pickle
import cv2
import sklearn
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from xgboost import XGBClassifier
from keras.models import load_model


#from PIL import Image

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'ServiceKey.json'
app = Flask(__name__)



def upload_blob_from_memory(bucket_name, contents, destination_blob_name):
    """Uploads a file to the bucket."""

    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"

    # The contents to upload to the file
    # contents = "these are my contents"

    # The ID of your GCS object
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_string(contents)




### Blob Cropper Code

DEV_MODE = False
DEBUG = False
def Debug(msg):
    """
    Dynamically prints console statements,
    using the global var debug
    :param msg:
    :return void:
    """
    global DEBUG
    if DEBUG == True:
        print(f"[DEBUG]: {msg}")
        
def AppMsg(msg):
    """
    Prints a formated application message
    """
    print(f"[CROPPER]: {msg}")

def getFilePaths(folder):
    """
    Returns an array of file paths found in given folder
    """
    image_files = []
    for filename in os.listdir(folder):
        image_files.append(folder+'/'+filename)
    return image_files

def resizeCropSquare(center_pt: tuple, rect_v1: tuple, rect_v2: tuple):
  """
  The method will reduce the size of a crop area by 5%
  :param center_pt: The center of the key point
  :param rect_v1: Is the top left vertex of the original crop square
  :param rect_v2: Is the bottom right vertex of the original crop square
  :return tuple: Returns new crop square coordinates 
  """
  x_center = center_pt[0]
  y_center = center_pt[1]
  current_area = (rect_v2[0] - rect_v1[0]) * (rect_v2[1] - rect_v1[1])
  new_area = int(current_area * .95)
  Debug(f'new_area = {new_area}')
  side_len = int(math.sqrt(new_area))
  new_radius = side_len / 2
  new_v1 = (int(x_center - new_radius), int(y_center - new_radius))
  new_v2 = (int(x_center + new_radius), int(y_center + new_radius))
  return (new_v1, new_v2)

def cropSquare(full_path, org_cropped):
  try:
    cv2.imwrite(full_path, org_cropped)
    #with open(full_path, 'wb') as out: #write image to container directory
    #  out.write(bytesOfImage)
    return True
  except:
    return False

def findRange(num_list: list):
    if len(num_list) == 0:
      return None
    elif len(num_list) == 1:
      return (num_list[0], num_list[0])
    else:
      min_val = min(num_list)
      max_val = max(num_list)
      return (min_val, max_val)

def setBlobDetector(minArea: int):
  params = cv2.SimpleBlobDetector_Params() 

  # Set Area filtering parameters 
  params.filterByArea = True
  Debug(f'MinArea: {minArea}')
  params.minArea = minArea
  params.maxArea = minArea * 14

  params.filterByColor = False
  params.blobColor = 0

  params.minDistBetweenBlobs = math.sqrt(minArea)

  # Set Circularity filtering parameters 
  params.filterByCircularity = False 
  # params.minCircularity = 0

  # Set Convexity filtering parameters 
  params.filterByConvexity = True
  params.minConvexity = 0
  params.maxConvexity = 1
        
  # Set inertia filtering parameters 
  params.filterByInertia = True
  params.minInertiaRatio = 0
  params.maxInertiaRatio = 1

  return cv2.SimpleBlobDetector_create(params)

def cropBlobs(folder_path: str, image_path: str):
  org_img = cv2.imread(image_path)
  gray_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

  height, width = gray_img.shape[:2]
  half_percent = int((height*width)*.5)
  magic_percent = half_percent / 1600
  side_len = int(math.sqrt(magic_percent))
  Debug(f'side_len = {side_len}')

  detector = setBlobDetector(magic_percent)

  flattened_gray_img = gray_img.ravel()

  mean = np.mean(flattened_gray_img)
  std_deviation = np.std(flattened_gray_img)
  left_bound = mean - std_deviation

  new_threshold = left_bound  - (std_deviation / 2)
  lum_threshold = left_bound  - (std_deviation / 2) if new_threshold > 0 else left_bound
  Debug(f'Initial threshold = {lum_threshold}')
  ret, threshed_img = cv2.threshold(gray_img, lum_threshold, 255, cv2.THRESH_BINARY)

  most_keypts_found = []
  most_keypts_found.append(detector.detect(threshed_img))
  most_keypts_found.append(threshed_img)
  most_keypts_found.append(lum_threshold)

  if len(most_keypts_found[0]) == 0:
    most_keypts_found.append(detector.detect(gray_img))
    most_keypts_found.append(gray_img)
    most_keypts_found.append(None)
  if len(most_keypts_found[0]) == 0:
    n = 2
    while lum_threshold < 200:
      if n == 2:
        lum_threshold = left_bound
      else:
        lum_threshold = left_bound  + n * (std_deviation / 10)
      Debug(f'new_lum_threshold = {lum_threshold}')
      ret, threshed_img = cv2.threshold(gray_img, lum_threshold, 255, cv2.THRESH_BINARY)
      threshed_keypts = detector.detect(threshed_img)
      if len(threshed_keypts) > len(most_keypts_found[0]):
        most_keypts_found[0] = threshed_keypts
        most_keypts_found[1] = threshed_img
        Debug(f'Saving threshold = {lum_threshold}')
        most_keypts_found[2] = lum_threshold
      n += 2 

  threshed_keypts = most_keypts_found[0]
  threshed_img = most_keypts_found[1]
  lum_threshold = most_keypts_found[2]

  Debug(f'Resulting lum_threshold = {lum_threshold}')
  Debug(f'mean = {mean}')

# if DEBUG:
#     plt.hist(flattened_gray_img, bins=256, range=[0,256])
#     plt.axvline(mean, color='k', linestyle='dashed', linewidth=1)
#     if lum_threshold is not None:
#       plt.axvline(lum_threshold, color='blue', linestyle='dashed', linewidth=1)
#     plt.axvline(left_bound, color='red', linestyle='dashed', linewidth=1)
#     fig = plt.figure(figsize =(10, 7))
#     plt.boxplot(flattened_gray_img)


  AppMsg(f'Number of Keypoints= {len(threshed_keypts)}')
  img_with_keypts = cv2.drawKeypoints(
      threshed_img, 
      threshed_keypts, 
      np.array([]), 
      (0, 0, 255), 
      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
  )
  org_with_keypts = cv2.drawKeypoints(
      org_img, 
      threshed_keypts, 
      np.array([]), 
      (0, 0, 255), 
      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
  )
  if DEV_MODE and DEBUG:
    cv2.rectangle(img_with_keypts, (0, 0), (side_len, side_len), (0, 255, 0), 5)
  """
  Crop key points
  """
  # https://www.pythonpool.com/opencv-keypoint/
  crop_on = True
  for keypt in threshed_keypts:
    x_center = keypt.pt[0]
    y_center = keypt.pt[1]
    radius = (keypt.size / 2) * 1.25
    v1 = (int(x_center - radius), int(y_center - radius))
    v2 = (int(x_center + radius), int(y_center + radius))

    v1 = tuple((coord if coord >= 0 else 0 for coord in v1))
    v2 = tuple((coord if coord >= 0 else 0 for coord in v2))

    gray_cropped = gray_img[v1[1]:v2[1], v1[0]:v2[0]]
    flat_gray_img = gray_cropped.ravel()
    range = findRange(flat_gray_img)
    lum_diff = range[1] - range[0]
    Debug(f'crop range = {range}')
    Debug(f'range diff = {lum_diff}')


    if crop_on and DEV_MODE and lum_diff > 40:
        # https://stackoverflow.com/questions/15589517/how-to-crop-an-image-in-opencv-using-python
        # roi = im[y1:y2, x1:x2]
        gray_cropped = threshed_img[v1[1]:v2[1], v1[0]:v2[0]]
        org_cropped = org_img[v1[1]:v2[1], v1[0]:v2[0]]
        crop_on = False
        file_name = 'cropped-blob-' + str(uuid.uuid4()) + '.png'
        full_path = os.path.join(folder_path, file_name)
        # cv2.imwrite(full_path, cropped)
        # cv2_imshow(gray_cropped)
        # cv2_imshow(org_cropped)
        cv2.rectangle(img_with_keypts, v1, v2, (255, 0, 0), 5)
    # cv2.rectangle(org_img, v1, v2, (255, 0, 0), 5)

    if not DEV_MODE and lum_diff > 40:
      folder_exists = os.path.exists(folder_path)
      if not folder_exists:
        AppMsg('Output folder does not exist')
        AppMsg(f'Creating a folder at [{folder_path}]')
        os.makedirs(folder_path)

      org_cropped = org_img[v1[1]:v2[1], v1[0]:v2[0]]
      file_name = 'cropped-blob-' + str(uuid.uuid4()) + '.png'
      full_path = os.path.join(folder_path, file_name)

      didCrop = cropSquare(full_path, org_cropped)
      new_v1 = v1
      new_v2 = v2
      attempts_left = 30
      while not didCrop and attempts_left > 0:
        new_rect = resizeCropSquare((x_center, y_center), new_v1, new_v2)
        new_v1 = new_rect[0]
        new_v2 = new_rect[1]
        org_cropped = org_img[new_v1[1]:new_v2[1], new_v1[0]:new_v2[0]]
        didCrop = cropSquare(full_path, org_cropped)
        attempts_left -= 1
        if didCrop and DEBUG:
          cv2.rectangle(org_img, new_v1, new_v2, (0, 0, 255), 5)
      if DEBUG:
        cv2.rectangle(org_img, v1, v2, (255, 0, 0), 5)
    

  return (org_img, img_with_keypts)


### Image Processing

def process_image(image, target_shape):
    """Given an image, process it and return the array."""
    # Load the image.
    h, w, _ = target_shape
    image = load_img(image, target_size=(h, w))

    # Turn it into numpy, normalize and return.
    img_arr = img_to_array(image)
    #x = (img_arr / 255.).astype(np.float32)
    x = img_arr

    return x




def extract_image_centers(images, hcrop=0, wcrop=10):
    image_centers = []
    for img in images:
        x = img.shape[0]
        y = img.shape[1]
        xc = int(hcrop*x/100)
        yc = int(wcrop*y/100)
        img = img[xc:x-xc, yc:y-yc]
        img = cv2.resize(img,(150,150))
        image_centers.append(img)
    return image_centers

def load_images_from_folder(folder, resize=True, expand = False):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if resize:
            img = cv2.resize(img,(150,150))

        if expand == True:
          img = np.expand_dims(img, axis = 0) ##test
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img is not None:
            images.append(img)
    return images


def classifyImage():

    xgb_model_latest = xgb.XGBClassifier()
    xgb_model_latest._le = LabelEncoder().fit(['0', '1'])
    #xgb_model_latest = pickle.load(open('/app/modelXGBoost.h5', 'rb')) #Load model
    xgb_model_latest.load_model('/app/modelXGBoost.h5') #Load model

    image_file = '/app/inputImage.jpg'
    blob_folder = '/app/blobs'
    org_img, thresh_img = cropBlobs(blob_folder,image_file ) #crop blobs

    Crops = load_images_from_folder(blob_folder,expand=False) #height,width, color channels
    Crops_Extracted = extract_image_centers(Crops, hcrop=40/2, wcrop=40/2)

    lengthCrop = str(len(Crops))


    #upload_blob_from_memory("ecofiltro-bucket",lengthCrop, "apple")

    if len(Crops_Extracted) == 0:
      return ["0.90", "0.15"]

    images2 = np.array(Crops_Extracted).astype('float32')

    num_classes = 2
    SIZE = images2[0].shape[0]

    VGG_model = VGG19(input_shape=( SIZE,SIZE, 3),include_top=False,weights='imagenet')

    for layer in VGG_model.layers:
        layer.trainable = False


    X_test_feature2 = VGG_model.predict(images2)
    X_test_features2 = X_test_feature2.reshape(X_test_feature2.shape[0], -1)

    predict =  xgb_model_latest.predict_proba(X_test_features2)

    return predict


@app.route('/image', methods=['POST'])
def image():


    bytesOfImage = request.get_data() #get image data base64
 
    with open('/app/inputImage.jpg', 'wb') as out: #write image to container directory
      out.write(bytesOfImage)

    pred_result = classifyImage()    

    allClean = True #
    result = ""
    threshold = 0.9

    try:
      for pred in pred_result:
        if float(str(pred[0])) >= threshold:
          result = 'Dirty'
          allClean = False
          break
        elif float(str(pred[1])) < threshold:
          result = 'Inconclusive'
          allClean = False
    except:
      return {"Prediction" : ["0.10","0.90"]}

    if allClean == True:
      result = 'Clean'   
    
    r = [[]]
    if result == 'Clean':
      r[0].append(0.9)
      r[0].append(0.0)
    elif result == 'Dirty':
      r[0].append(0)
      r[0].append(0.9)
    else:
      r[0].append(0.8)
      r[0].append(0.8)

    return {"Prediction" : [str(r[0][0]),str(r[0][1])]}

    #upload_blob_from_memory("ecofiltro-bucket","finished no errors", "prediction")
    #return {"Prediction" : result}
    #upload_blob("ecofiltro-bucket", r'/app/inputImage.jpg', "testTULT") debugging images
    #return {"Prediction" : ["0.90","0.10"]}
     # return {"Prediction" : [str(r[0][0]),str(r[0][1])]}


if __name__ == '__main__':
    server_port = os.environ.get('PORT', '8080')
    app.run(debug=False, port=server_port, host='0.0.0.0')
