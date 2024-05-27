import time
print("Start Script")
start = time.time()
################################################################################################################################################################
###################################################################### Imports #################################################################################

# info: 45.82518940369288 min duration for 1 sequence

import math
import cv2
import pandas as pd
import os
from scipy.stats import ks_2samp
import cmath
from os.path import exists
from sqlalchemy_api import DriftDetection
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy_api import *
from datetime import datetime
import numpy as np
from PIL import Image
import shutil
import subprocess
from numpy import asarray

###################################################################### Imports #################################################################################
################################################################################################################################################################
# -------------------------------------------------------------------------------------------------------------------------------------------------------------#
################################################################################################################################################################
################################################################## Imports Config ##############################################################################

from configparser import ConfigParser
config = ConfigParser()
config.read_file(open(r'config.txt'))

MinIO_path = config.get('MinIOServer', 'path_local_pc')

sqlData = config.get('SQLServer', 'sqlData')
localhost_SQL = config.get('SQLServer', 'host')
user_SQL = config.get('SQLServer', 'user')
password_SQL = config.get('SQLServer', 'password')
database_SQL = config.get('SQLServer', 'database')

################################################################## Imports Config ##############################################################################
################################################################################################################################################################
# -------------------------------------------------------------------------------------------------------------------------------------------------------------#
################################################################################################################################################################
################################################################### Deklarationen ##############################################################################

import mysql.connector
mydb = mysql.connector.connect(
      host=localhost_SQL,
      user=user_SQL,
      password=password_SQL,
      database=database_SQL
    )

engine = db.create_engine(
  sqlData, echo=False, pool_recycle=3600)
Session = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))
session = Session()

################################################################### Deklarationen ##############################################################################
################################################################################################################################################################
# -------------------------------------------------------------------------------------------------------------------------------------------------------------#
################################################################################################################################################################
############################################################ Funktionen Allgemeingültig ########################################################################

def get_existing_models(MinIO_path_local_PC: str) -> list:
  """Determine all possible model combinations depending on MinIO's model bucket folder structure.
  The determined combinations will be the input for the "determine_golden_data" function.

  :param MinIO_path_local_PC: path storage folder from MinIO on local PC which is written and retrieved from config.txt
  :return possible_combinations: getting all possible model combinations based on the subfolder structure in MinIO's model bucket as python list.
  """
  path_model_bucket_local_PC = MinIO_path_local_PC + '\models'
  possible_combinations = []

  for A5Es in os.listdir(path_model_bucket_local_PC):
    for model_type in os.listdir(path_model_bucket_local_PC + '/' + A5Es):
      for boardside in os.listdir(path_model_bucket_local_PC + '/' + A5Es + '/' + model_type):
        d = os.path.join(path_model_bucket_local_PC, A5Es, model_type, boardside)
        d = d.replace("\\","/")
        d_array = d.split("/")
        d_array = d_array[-3:][:]
        if 'ImgClass' in d_array[1]:
          possible_combinations.append(d_array)

  return possible_combinations


def get_golden_date(SQL_Database, model_type: str, A5E_number: str, board_side: str):
  """Determine oldest date of newest model depending on model type, A5E-number and boardside.
  The determined date will be the seperation point between reference and production dataset.

  :param SQL_Database: database initialized with mysql.connector.connect
  :param model_type: AE or ImgClass, retrieved from subfolder structures in MinIO model bucket.
  :param A5E_number: AE5 number of product, retrieved from subfolder structures in MinIO model bucket and revision number shortened if desired and activated in config.txt.
  :param board_side: X1 or X2, retrieved from subfolder structures in MinIO model bucket.
  :return golden_date: date when the newest model version did its first prediction.
  :return newest_model_version: newest model version of the appropiate model combination (v1, v2, v3....).
  """
  mycursor = SQL_Database.cursor()

  try:
    if 'ImgClass' in model_type:
        
      sql_Query = f"SELECT xray_time, model_version FROM board_pred WHERE boardside='{board_side}' AND model_type='{model_type}' AND type LIKE '%{A5E_number}%'"

      mycursor.execute(sql_Query)
      records = mycursor.fetchall()

      records_df = pd.DataFrame(records, columns=['xray_time', 'model_version'])  # convert to pandas dataframe because determination is easier.

      newest_model_version = records_df['model_version'].max(axis=0)    # get the newest model version
      records_df = records_df[records_df['model_version'] == newest_model_version]  # extract new dataframe which just contain entries with newest model version.
      golden_date = records_df['xray_time'].min(axis=0)   # get the oldest date
  except:
    print(f"Error! No existing version entries in SQL to analyze for {A5E_number} {board_side} {model_type}. Model Monitoring for this combination can't be performed. If analyzation desired, please train an appropriate model on existing data and push it to the according MinIO's subfolder")
    golden_date = "NaN"
    newest_model_version = "NaN"

  return golden_date, newest_model_version


def addDrifttoSQLdriftdetectiontable(drift_type_selected: int, session, model_type: str, version: str, A5E_number: str, board_side: str, ks_test: float, drift_detected_ks_test: str, alpha_level_ks_test: float, percentage_pixels_drifted: float, threshold_pixel_drift: float, overall_pixel_drift_detected: str, URL: str):
  """Adds new entry into drift detection table.

  :param drift_type_selected: 1=Probability Drift, 2=Image Dataset Drift, 3=Pixel Drift
  :param session: session on mysql driftdetection table xray database initialized by scoped_session & sessionmaker library
  :param model_type: ImgClass2D or AE2D
  :param version: newest version of the appropiate model (v1, v2, v3....)
  :param A5E_number: type number of the product
  :param board_side: X1 or X2
  :param ks_test: calculated value by the KS-test
  :param drift_detected_ks_test: True or False as python string
  :param alpha_level_ks_test: determined alpha level (=threshold for true or false outcome) depending on the datasets length
  :param percentage_pixels_drifted: percentage of pixel drifted in one gim
  :param threshold_pixel_drift: threshold manually set for true or false outcome
  :param overall_pixel_drift_detected: True or False as python string
  :param URL: URL of the according pixel drift image which later is pushed on a localhost server for the modle monitoring dashboard
  """
  driftdetection = DriftDetection()

  dateTimeObj = datetime.now()
  current_time = dateTimeObj.strftime("%Y-%m-%d %H:%M:%S")

  # ks_test = ks_test.item()

  if drift_type_selected==1:
    drift_type="Prediction Probability"
    feature="Prediction Probability"
    DriftDetection.addDrift(driftdetection, sess=session, time=current_time, model_type=model_type, type_board=A5E_number, boardside=board_side, version=version, drift_type=drift_type, feature=feature, alpha_level_ks_test=alpha_level_ks_test, ks_test=ks_test, drift_detected_ks_test=drift_detected_ks_test, percentage_pixels_drifted=percentage_pixels_drifted, threshold_pixel_drift=threshold_pixel_drift, overall_pixel_drift_detected=overall_pixel_drift_detected, URL=URL)
  elif drift_type_selected==2:
    drift_type="Images Overall"
    feature="Image"
    DriftDetection.addDrift(driftdetection, sess=session, time=current_time, model_type=model_type, type_board=A5E_number, boardside=board_side, version=version, drift_type=drift_type, feature=feature, alpha_level_ks_test=alpha_level_ks_test, ks_test=ks_test, drift_detected_ks_test=drift_detected_ks_test, percentage_pixels_drifted=percentage_pixels_drifted, threshold_pixel_drift=threshold_pixel_drift, overall_pixel_drift_detected=overall_pixel_drift_detected, URL=URL)
  elif drift_type_selected==3:
    drift_type="Pixel"
    feature="Pixel"
    DriftDetection.addDrift(driftdetection, sess=session, time=current_time, model_type=model_type, type_board=A5E_number, boardside=board_side, version=version, drift_type=drift_type, feature=feature, alpha_level_ks_test=alpha_level_ks_test, ks_test=ks_test, drift_detected_ks_test=drift_detected_ks_test, percentage_pixels_drifted=percentage_pixels_drifted, threshold_pixel_drift=threshold_pixel_drift, overall_pixel_drift_detected=overall_pixel_drift_detected, URL=URL)


def flatten(image: list) -> list:
  """flatten an image from 2D array with [x][y] to 1D [x*y]

  :param image: input image with 2D shape [x][y]
  :return image_flattened: output image with 1D shape [x*y]
  """
  return [item for sublist in image for item in sublist]


def Average(array: list) -> float:
  """calculating average of an 1D array

  :param array: input of 1D array
  :return mean: mean of all values in the 1D array
  """
  return sum(array) / len(array)
  

def two_sample_kolmogorov_kmirnov_test(reference_dataset: list, current_dataset: list, reference_dataset_length: int, current_dataset_length: int):
  """calculating drift of two python 1D arrays with two sample kolmogorov kmirnov test, which analyzes a change in the distribution of the values.

  :param reference_dataset: reference dataset of the values to be compared
  :param current_dataset: current dataset of the values to be compared
  :param reference_dataset_length: length of the reference dataset
  :param current_dataset_length: length of the reference dataset
  :return ks_test: value determined by the KS-test
  :return drift: True (=drift detected) or False (=no drift detected)
  :return alpha_level: threshold for the conclusion of drift detected or no drift detected determined by the length of the datasets


  EXPLANATION OF THE STATISTICAL TEST:
  alternative=two-sided: The null hypothesis is that the two distributions are identical, the alternative is that they are not identical
  If the KS statistic is small or the p-value is high, then we cannot reject the null hypothesis (identical distributions) in favor of the alternative
  -> KS small: identical /// KS high: not identical
  -> p-value small: not identical /// p-value high: identical
  Source: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html
  formula level a where null hypotesis is rejected: squareroot((n+m)/(n*m)) = root((250+250)/(250*250)) = 0.0894
  """
  alpha_level = math.sqrt((reference_dataset_length+current_dataset_length)/(reference_dataset_length*current_dataset_length)) # 0.0894 at 250/250 proportion
  
  ks_test = ks_2samp(reference_dataset, current_dataset, alternative='two-sided').statistic
  
  if ks_test < alpha_level:
    drift = "False"
  else:
    drift = "True"

  return ks_test, drift, alpha_level

############################################################ Funktionen Allgemeingültig ########################################################################
################################################################################################################################################################
# -------------------------------------------------------------------------------------------------------------------------------------------------------------#
################################################################################################################################################################
################################################# Funktionen Prediction Probability Drift Detection ############################################################

def get_amount_of_datapoints(SQL_Database, model_type: str, model_version: str, A5E_number: str, board_side: str, golden_date) -> int:
  """Getting the amount of datapoints in the SQL board_pred table for the appropiate model combination.

  :param SQL_Database: connection to SQL's xraydb database with the mysql.connector libarary
  :param model_type: ImgClass2D or AE2D
  :param model_version: newest version of the appropiate model combination (v1, v2, v3...)
  :param A5E_number: type number of the product
  :param board_side: X1 or X2
  :param golden_date: date when the newest model was pushed
  :return amount_of_datapoints: amount of datapoints available in the SQL board_pred table
  """
  mycursor = SQL_Database.cursor()

  sql_Query = f"SELECT COUNT(*) FROM xraydb.board_pred WHERE model_type='{model_type}' AND model_version='{model_version}' AND type LIKE '%{A5E_number}%' AND boardside='{board_side}' AND xray_time>='{golden_date}'"
  mycursor.execute(sql_Query)
  records = mycursor.fetchall()

  tuple = records[0]
  amount_of_datapoints = tuple[0]

  return amount_of_datapoints


def get_datasets(SQL_Database, model_type: str, model_version: str, A5E_number: str, board_side: str, golden_date) -> list:
  """Get the datasets for the appropiate model combinations.

  :param SQL_Database: connection to SQL's xraydb database with the mysql.connector libarary
  :param model_type: ImgClass2D or AE2D
  :param model_version: newest version of the appropiate model combination (v1, v2, v3...)
  :param A5E_number: type number of the product
  :param board_side: X1 or X2
  :param golden_date: date when the newest model was pushed
  :return reference_dataset: prediction probability of the images / reference dataset as python array
  :return current_dataset: prediction probability of the images / current dataset as python array
  """
  mycursor = SQL_Database.cursor()
  sql_Query = f"SELECT predict_proba FROM xraydb.board_pred WHERE model_type='{model_type}' AND model_version='{model_version}' AND type LIKE '%{A5E_number}%' AND boardside='{board_side}' AND xray_time>='{golden_date}' AND predict_proba<>'None' ORDER BY xray_time ASC LIMIT 250"
  mycursor.execute(sql_Query)
  reference_dataset = mycursor.fetchall()
  for i in range(len(reference_dataset)):
    reference_dataset[i] = float(reference_dataset[i][0])

  sql_Query = f"SELECT predict_proba FROM xraydb.board_pred WHERE model_type='{model_type}' AND model_version='{model_version}' AND type LIKE '%{A5E_number}%' AND boardside='{board_side}' AND xray_time>='{golden_date}' AND predict_proba<>'None' ORDER BY xray_time DESC LIMIT 250"
  mycursor.execute(sql_Query)
  current_dataset = mycursor.fetchall()
  for i in range(len(current_dataset)):
    current_dataset[i] = float(current_dataset[i][0])

  return reference_dataset, current_dataset

################################################# Funktionen Prediction Probability Drift Detection ############################################################
################################################################################################################################################################
# -------------------------------------------------------------------------------------------------------------------------------------------------------------#
################################################################################################################################################################
########################################################### Funktionen Images Drift Detection ##################################################################

def get_imagenames(SQL_Database, MinIO_path, model_type: str, model_version: str, A5E_number: str, board_side: str, golden_date) -> list:
  """Get the image names from the SQL board_pred table. 
  Subsequently just the image names will be taken into account which are existing in the MinIO gim bucket with the right image shape.

  :param SQL_Database: connection to SQL's xraydb database with the mysql.connector libarary
  :param MinIO_path: path storage folder from MinIO on local PC which is written and retrieved from config.txt
  :param model_type: ImgClass2D or AE2D
  :param model_version: newest version of the appropiate model combination (v1, v2, v3...)
  :param A5E_number: type number of the product
  :param board_side: X1 or X2
  :param golden_date: date when the newest model was pushed
  :return reference_dataset: image names of the images / reference dataset as python array
  :return production_dataset: image names of the images / current dataset as python array
  """
  mycursor = SQL_Database.cursor()

  sql_Query = f"SELECT COUNT(*) FROM board_pred WHERE model_type='{model_type}' AND boardside='{board_side}' AND model_version='{model_version}' AND type LIKE '%{A5E_number}%' AND xray_time>='{golden_date}' AND ground_truth='iO'"
  mycursor.execute(sql_Query)
  records = mycursor.fetchall()
  tuple = records[0]
  amount_of_datapoints = tuple[0]
  queryamount = math.floor(amount_of_datapoints/2)

  if "ImgClass" in model_type:
    sql_Query = f"SELECT name FROM board_pred WHERE model_type='{model_type}' AND boardside='{board_side}' AND model_version='{model_version}' AND type LIKE '%{A5E_number}%' AND xray_time>='{golden_date}' AND ground_truth='iO' ORDER BY xray_time ASC LIMIT {queryamount}"
    mycursor.execute(sql_Query)
    reference_dataset_names = mycursor.fetchall()
    
    sql_Query = f"SELECT name FROM board_pred WHERE model_type='{model_type}' AND boardside='{board_side}' AND model_version='{model_version}' AND type LIKE '%{A5E_number}%' AND xray_time>='{golden_date}' AND ground_truth='iO' ORDER BY xray_time DESC LIMIT {queryamount}"
    mycursor.execute(sql_Query)
    current_dataset_names = mycursor.fetchall()

  reference_dataset = []
  for image_name in reference_dataset_names:
    if exists(MinIO_path + '\gims\\' + image_name[0]) == True:
      reference_dataset.append(image_name[0])
  reference_dataset = reference_dataset[:250]

  production_dataset = []
  for image_name in current_dataset_names:
    if exists(MinIO_path + '\gims\\' + image_name[0]) == True:
      production_dataset.append(image_name[0])
  production_dataset = production_dataset[:250]
    
  return reference_dataset, production_dataset


def load_images(MinIO_path_local_PC: str, reference_dataset_image_names: list, current_dataset_image_names: list) -> list:
  """Loading images and creating dataset depending on list of images' names.

  :param MinIO_path_local_PC: path storage folder from MinIO based on local PC which is written and retrieved from config.txt 
  :param reference_dataset_image_names: list which contains all image names to load.
  :param current_dataset_image_names: list which contains all image names to load.
  :return reference_dataset_flattened: reference dataset flattened to 1D arrays/images.
  :return current_dataset_flattened: current dataset flattened to 1D arrays/images.
  :return reference_dataset_mean: mean of each image of the reference dataset.
  :return current_dataset_mean: mean of each image of the reference dataset.
  :return h_min: Shapes of the images differ due to inspection machine. Minimum height in datasets
  :return w_min: Shapes of the images differ due to inspection machine. Miniumum width in datasets
  """
  
  # get the smallest image sizes
  h_min = 999999
  w_min = 999999
  for image_name in reference_dataset_image_names:
    img = cv2.imread(MinIO_path_local_PC + '\gims\\' + image_name)
    h = img.shape[0]
    w = img.shape[1]
    if h < h_min:
      h_min = h
    if w < w_min:
      w_min = w
  for image_name in current_dataset_image_names:
    img = cv2.imread(MinIO_path_local_PC + '\gims\\' + image_name)
    h = img.shape[0]
    w = img.shape[1]
    if h < h_min:
      h_min = h
    if w < w_min:
      w_min = w
  
  # ************* REFERENCE *************
  # loading and cropping all images
  reference_dataset = []
  for image_name in reference_dataset_image_names:
    img = cv2.imread(MinIO_path_local_PC + '\gims\\' + image_name)
    img = img[:,:,0]                # drop 3rd dimension. Other dimension unnecassary
    # start crop
    h = img.shape[0]
    h_diff = int(h - h_min)
    h1 = int(h_diff/2)
    h2 = int(h-(h_diff/2))
    w = img.shape[1]
    w_diff = int(w - w_min)
    w1 = int(w_diff/2)
    w2 = int(w-(w_diff/2))
    crop = img[h1:h2, w1:w2]
    # append crop
    reference_dataset.append(crop)
  # flatten all images
  reference_dataset_flattened = []
  for image in reference_dataset:
    reference_dataset_flattened.append(np.concatenate(image))
  # calculate mean of all images
  reference_dataset_mean = []
  for image_flattened in reference_dataset_flattened:
    reference_dataset_mean.append(Average(image_flattened))

  # ************* CURRENT *************
  # loading and cropping all images
  current_dataset = []
  for image_name in current_dataset_image_names:
    img = cv2.imread(MinIO_path_local_PC + '\gims\\' + image_name)
    img = img[:,:,0]                # drop 3rd dimension. Other dimension unnecassary
    # start crop
    h = img.shape[0]
    h_diff = int(h - h_min)
    h1 = int(h_diff/2)
    h2 = int(h-(h_diff/2))
    w = img.shape[1]
    w_diff = int(w - w_min)
    w1 = int(w_diff/2)
    w2 = int(w-(w_diff/2))
    crop = img[h1:h2, w1:w2]
    # append crop
    current_dataset.append(crop)
  # flatten all images
  current_dataset_flattened = []
  for image in current_dataset:
    current_dataset_flattened.append(np.concatenate(image))
  # calculate mean of all images
  current_dataset_mean = []
  for image_flattened in current_dataset_flattened:
    current_dataset_mean.append(Average(image_flattened))

  return reference_dataset_flattened, current_dataset_flattened, reference_dataset_mean, current_dataset_mean, h_min, w_min

########################################################### Funktionen Images Drift Detection ##################################################################
################################################################################################################################################################
# -------------------------------------------------------------------------------------------------------------------------------------------------------------#
################################################################################################################################################################
########################################################### Funktionen Pixel Drift Detection ###################################################################

def shape_imagelist_to_pixellist(dataset_images_flattened: list) -> list:
  """Array of images shaped that the according pixels of all images are in one dimension.
  
  :param dataset_images_flattened: Array of images where all images are flattened
  :return dataset_images_pixel2pixel: Array of pixels in one dimension.
  """
  dataset_images_pixel2pixel = []
  for i in range(len(dataset_images_flattened[0])):
    identicalpixellist=[]
    for j in range(len(dataset_images_flattened)):
      identicalpixellist.append(dataset_images_flattened[j][i])
    dataset_images_pixel2pixel.append(identicalpixellist)
  return dataset_images_pixel2pixel


def get_drift_image(true_false_list_drift_detected_kstest: list, h_min, w_min) -> Image.Image:
  """Create the drift image to see which parts of the images have drifted.

  :param true_false_list_drift_detected_kstest: Array containing True & False if pixel has drifted or not drifted
  :param h_min: Shapes of the images differ due to inspection machine
  :param w_min: Shapes of the images differ due to inspection machine
  :return drift_image: pixel drift image. Red means drifted, green means not drifted.
  """
  height = h_min
  width = w_min

  true_false_list_drift_detected_kstest = np.array(true_false_list_drift_detected_kstest)
  true_false_list_drift_detected_kstest = np.reshape(true_false_list_drift_detected_kstest, (height, width))

  drift_image = np.zeros(shape=(height, width, 3), dtype=float)  

  for i in range(len(true_false_list_drift_detected_kstest)):
    for j in range(len(true_false_list_drift_detected_kstest[0])):
      if true_false_list_drift_detected_kstest[i][j]=="True":
        drift_image[i][j][0] = 1.0
      elif true_false_list_drift_detected_kstest[i][j]=="False":
        drift_image[i][j][1] = 1.0

  drift_image = Image.fromarray((drift_image * 255).astype(np.uint8))

  return drift_image


def moveimages_and_savepixeldriftimage(pixeldriftimage: Image.Image, A5E_number: str, board_side: str, minio_path_local_pc=MinIO_path):
  """First, move the existing pixel drift images into the archive. Then, save the new pixel drift image. 

  :param pixeldriftimage: pixel drift image. Red means drifted, green means not drifted.
  :param A5E_number: type number of the product
  :param board_side: X1 or X2
  :param minio_path_local_pc: DEFAULT=MinIO_path, which is directly connected to config.txt. path storage folder from MinIO on local PC which is written and retrieved from config.txt
  """
  serving_folder = minio_path_local_pc+"\serving"
  archive_folder = minio_path_local_pc+"\serving_archive"
  dateTimeObj = datetime.now()
  current_time = dateTimeObj.strftime("%Y%m%d%H%M%S")

  if not os.path.exists(serving_folder):
    os.makedirs(serving_folder)
  if not os.path.exists(archive_folder):
    os.makedirs(archive_folder)

  # move the pixeldrift image from the serving folder into the archive folder
  for filename in os.listdir(serving_folder):
    if filename == f"pixeldrift_{A5E_number}_{board_side}.jpg":
      file_path = os.path.join(serving_folder, filename)
      try:
        shutil.move(file_path, os.path.join(archive_folder, f'pixeldrift_{A5E_number}_{board_side}_{current_time}.jpg'))
      except Exception as e:
        print(f'Failed to move {file_path}. Reason: {e}')

  # save the pixeldrift image in the folder
  pixeldriftimage.save(serving_folder + f'\pixeldrift_{A5E_number}_{board_side}.jpg')


def start_http_server(port = "9000", directory=MinIO_path+"\serving"):
  """Startup a HTTP server for the specified directory on the specified port ("9000" recommended to make the setup work)
  """
  try:
    with open(os.devnull, 'w') as t:
      subprocess.Popen(["python","-m","http.server",port], stdout=t, stderr=t, cwd=directory)
    # print("HTTP server was sucessfully started...")
  except:
    # print("HTTP server could not be started")
    pass

########################################################### Funktionen Pixel Drift Detection ###################################################################
################################################################################################################################################################
# -------------------------------------------------------------------------------------------------------------------------------------------------------------#
################################################################################################################################################################
####################################################################### Main ###################################################################################

if __name__ == "__main__":

    ######################################################################################################################################################
    ############################################################# Main - Allgemein #######################################################################

    # getting all possible model combinations based on MinIO model bucket's subfolder structure
    existing_paths_in_MinIO_model_bucket = get_existing_models(MinIO_path)

    # appending all model comnbinations with revision number and without to one list.
    possible_combinations = []
    for i, content in enumerate(existing_paths_in_MinIO_model_bucket):
        a5e_number_without_revisionnumber = content[0][:11]
        possible_combinations.append([a5e_number_without_revisionnumber, content[1], content[2]])
        possible_combinations = pd.DataFrame(possible_combinations)
        possible_combinations = possible_combinations.drop_duplicates()
        possible_combinations = possible_combinations.values.tolist()
    for i, content in enumerate(existing_paths_in_MinIO_model_bucket):
        possible_combinations.append(content)

    # get the first date of the newest model
    combinations1 = []
    for i, content in enumerate(possible_combinations):
        golden_date, newest_version = get_golden_date(mydb, content[1], content[0], content[2])
        combinations1.append([content[1], newest_version, content[0], content[2], golden_date])

    # drop all rows (=combinations) with NaN values which are not analyzable 
    combinations1 = pd.DataFrame(combinations1)
    combinations1 = combinations1.dropna()
    combinations1 = combinations1.values.tolist()

    # check if datapoints in SQL >=500
    combinations2 =[]
    for i, content in enumerate(combinations1):
        amount_of_datapoints = get_amount_of_datapoints(mydb, content[0], content[1], content[2], content[3], content[4])
        if amount_of_datapoints>=500:
            combinations2.append([content[0], content[1], content[2], content[3], content[4]])

    model_combinations_in_MinIO_model_bucket = combinations2

    ############################################################# Main - Allgemein #######################################################################
    ######################################################################################################################################################
    # ---------------------------------------------------------------------------------------------------------------------------------------------------#
    ######################################################################################################################################################
    ##################################### Main - Drift Detection looping over all model combinations existing ############################################

    for i, content in enumerate(model_combinations_in_MinIO_model_bucket):
        print(f"starting {i+1}/{len(model_combinations_in_MinIO_model_bucket)} with following combination: {content[2]} {content[0]} {content[3]} {content[1]}")
        # probability drift
        reference_dataset, current_dataset = get_datasets(mydb, content[0], content[1], content[2], content[3], content[4])
        ks_test, drift_detected_kstest, alpha_level_ks_test = two_sample_kolmogorov_kmirnov_test(reference_dataset, current_dataset, len(reference_dataset), len(current_dataset))
        addDrifttoSQLdriftdetectiontable(drift_type_selected=1,
                                  session=session,
                                  model_type=content[0], 
                                  version=content[1], 
                                  A5E_number=content[2], 
                                  board_side=content[3], 
                                  ks_test=ks_test, 
                                  drift_detected_ks_test=drift_detected_kstest,
                                  alpha_level_ks_test=alpha_level_ks_test,
                                  percentage_pixels_drifted=None,
                                  threshold_pixel_drift=None,
                                  overall_pixel_drift_detected=None,
                                  URL=None)

        # image greyscale drift     
        if content[0]!="ImgClass3D":
            reference_dataset_imagenames, current_dataset_imagenames = get_imagenames(mydb, MinIO_path, content[0], content[1], content[2], content[3], content[4])
            if len(reference_dataset_imagenames) == 250 and len(current_dataset_imagenames) == 250:
                reference_dataset_images_flattened, current_dataset_images_flattened, reference_dataset_images_flattened_mean, current_dataset_images_flattened_mean, h_min, w_min = load_images(MinIO_path, reference_dataset_imagenames, current_dataset_imagenames)
                ks_test, drift_detected_kstest, alpha_level_ks_test = two_sample_kolmogorov_kmirnov_test(reference_dataset_images_flattened_mean, current_dataset_images_flattened_mean, len(reference_dataset_images_flattened_mean), len(current_dataset_images_flattened_mean))
                addDrifttoSQLdriftdetectiontable(drift_type_selected=2,
                                                session=session,
                                                model_type=content[0], 
                                                version=content[1], 
                                                A5E_number=content[2], 
                                                board_side=content[3], 
                                                ks_test=ks_test, 
                                                drift_detected_ks_test=drift_detected_kstest,
                                                alpha_level_ks_test=alpha_level_ks_test,
                                                percentage_pixels_drifted=None,
                                                threshold_pixel_drift=None,
                                                overall_pixel_drift_detected=None,
                                                URL=None)

                # pixel drift
                reference_dataset_images_pixel2pixel = shape_imagelist_to_pixellist(reference_dataset_images_flattened)
                current_dataset_images_pixel2pixel = shape_imagelist_to_pixellist(current_dataset_images_flattened)

                # Calculation of KS-Test for each pixel & saving all values into the according dictionary
                count_true=0
                count_false=0
                pixel_drift_img_true_false_array=[]
                for j in range(len(reference_dataset_images_pixel2pixel)):
                    ks_test, drift_detected_kstest, alpha_level_ks_test = two_sample_kolmogorov_kmirnov_test(reference_dataset_images_pixel2pixel[j], current_dataset_images_pixel2pixel[j], len(reference_dataset_images_pixel2pixel[j]), len(current_dataset_images_pixel2pixel[j]))
                    pixel_drift_img_true_false_array.append(drift_detected_kstest)
                    if drift_detected_kstest=="True":
                        count_true = count_true + 1
                    elif drift_detected_kstest=="False":
                        count_false = count_false + 1
                percentage_pixel_drifted = (count_true/(count_true+count_false))
                threshold_pixel_drift = 0.8
                if (count_true/(count_true+count_false)) > threshold_pixel_drift:
                    pixel_drift_detected_kstest = "True"
                else:
                    pixel_drift_detected_kstest = "False"
                addDrifttoSQLdriftdetectiontable(drift_type_selected=3,
                                                session=session,
                                                model_type=content[0], 
                                                version=content[1], 
                                                A5E_number=content[2], 
                                                board_side=content[3], 
                                                ks_test=None,
                                                drift_detected_ks_test=None,
                                                alpha_level_ks_test=alpha_level_ks_test,
                                                percentage_pixels_drifted=percentage_pixel_drifted,
                                                threshold_pixel_drift=threshold_pixel_drift,
                                                overall_pixel_drift_detected=pixel_drift_detected_kstest,
                                                URL=f"http://localhost:9000/pixeldrift_{content[2]}_{content[3]}.jpg")

                # generate pixel drift gim, save it in minio bucket 
                pixel_drift_image = get_drift_image(pixel_drift_img_true_false_array, h_min=h_min, w_min=w_min)
                moveimages_and_savepixeldriftimage(pixeldriftimage=pixel_drift_image, A5E_number=content[2], board_side=content[3])

    # start http server in serving bucket of minio
    start_http_server()

    ##################################### Main - Drift Detection looping over all model combinations existing ############################################
    ######################################################################################################################################################

    end = time. time()
    print("End Script")
    print(end - start, "sec ///", (end-start)/60, "min")

####################################################################### Main ###################################################################################
################################################################################################################################################################ 