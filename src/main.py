# Main runtime to watch for change on given directories,
# log it, store it, preprocess it and predict it with a given model.
# Tobias Meyer (DI FA MF EWA PE 1), Jan Erik Hagelloch (DI FA MF EWA PE 1)


# import basics of os, xray_time and watchdog
import io
import glob
import logging
import os
import time
from gradcam import *
#import nibabel as nib
import datetime as dt
import pytz
from configparser import ConfigParser
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler
from watchdog.events import LoggingEventHandler
from logging.handlers import RotatingFileHandler
import atexit
# import libraries
import boto3
from comesaConnect import *
from boto3_api import *
import gims
import modelLoad
from minio_api import *
import cv2
import pandas as pd
from pathlib import Path
from predict import *
import minio_api
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy_api import *
from sqlalchemy.sql import exists
from qmeWebService import *
from threading import Timer
from xpi4wesco_API import send_repair_result, generate_Set, send_test_result


class RepeatedTimer(object):
    def __init__(self, interval, function, *args, **kwargs):
        self._timer     = None
        self.interval   = interval
        self.function   = function
        self.args       = args
        self.kwargs     = kwargs
        self.is_running = False
        self.start()

    def _run(self):
        self.is_running = False
        self.start()
        self.function(*self.args, **self.kwargs)

    def start(self):
        if not self.is_running:
            self._timer = Timer(self.interval, self._run)
            self._timer.start()
            self.is_running = True

    def stop(self):
        self._timer.cancel()
        self.is_running = False


def exit_func():
    print("Pipeline stopped!")
    rt.stop()


def get_number_rec(path):
    count = 0
    head_tail = os.path.split(path)
    try:
        for file in os.listdir(head_tail[0]):
            if file.endswith(".rec"):
                count = count + 1
    except:
        pass
    return count


def create_filename(path: str):
    """creates filename in the following order:  FID from dirname + Timestamp from dirname +
    normal filename of rec/ vgi,
    returns adapted filename"""

    raw_filename = os.path.basename(path)
    dirname: str = os.path.dirname(path)

    # extract everything with the beginning of C-... in dirname
    try:
        extract_dirname: str = "C-" + dirname.split("C-", 1)[1]
    
    except (ValueError, IndexError):
        
        if "ERROR" in dirname:
            extract_dirname: str = "ERROR" + dirname.split("ERROR", 1)[1]
        else:
            extract_dirname = 'Test'

    # build new filename
    new_filename: str = extract_dirname + "_" + os.path.splitext(raw_filename)[0] + os.path.splitext(raw_filename)[1]
    return new_filename


def find_type_board(path: str):
    """finds the boardtype (A5E) in given path, based on 'A5E'
    :param path:
    :return: type of board string"""

    raw = "A5E" + path.split("A5E", 1)[1]

    # extract ASE-number from rest of the string
    boardtype = raw.split('_', 1)[0]
    return boardtype


def find_number_board(path: str, dataframe, boardside):
    """finds the boardnumber in given path (gims_name), based on the 4th '_' in the path string.
    Because numberings of X-Ray and QME is different, translate it in QME Pin Layout.
    Currently only valid for X1 Side
    :param path:
    :return: number of board on panel, in QME order int"""

    #if "X1" in path:
    raw_number = path.split('_')[4]
    raw_int = int(raw_number)
    #boardnumber = dictionary.get(raw_int, '')
    try:  # print(df_new)
        boardnumber = (dataframe.loc[
            (dataframe["Component Name"] == boardside) & (dataframe['Component Number'] == raw_int),
            ['Component Block Unit Number']])['Component Block Unit Number'].iloc[0]
    except IndexError:
        #print('No Mapping information for this file!')
        boardnumber = 1000
    return int(boardnumber)


def find_machine(path: str, client):
    """finds the machine in given path, based on the first substring of the triggerd path, beginning with 'F..'
    :param path: string
    :return: name of machine string"""
    listPath = (path.replace('/', '\\')).split('\\')
    indices = []
    for i, elem in enumerate(listPath):
        if client in elem:
            indices.append(i)
    machine_name = listPath[indices[0]]
    return machine_name


def find_boardside(path: str, sideList) -> object:
    """finds the boardside in given path, based on 'X...'
    :param path:
    :return: boardside string"""
    list_para = add_parentheses(sideList)
    for index, value in enumerate(list_para):
        if value in path:
            return sideList[index]


def find_fid(path: str):
    """finds the OFID / numbertype in given path (jpg name) based on 'C-...'
    :param path:
    :return: fid string
    """
    fid = path.split('_', 2)[0]
    return fid


def find_xraytime(path: str):
    """finds the timestamp of Xray-Event in given jpg name, and converts it to local xray_time GMT+1/2 (Berlin),
     based on '_' """
    timestamp = path.split('_', 3)[1]
    date_time_obj = dt.datetime.strptime(timestamp, '%Y%m%d%H%M%S')

    local_tz = pytz.timezone('Europe/Berlin')
    local_dt = date_time_obj.replace(tzinfo=pytz.utc).astimezone(local_tz)
    return local_tz.normalize(local_dt)


def find_test_time(timestamp: str):
    """converts the timestamp from the qme database in UTC and local time zone date time objects"""
    timestamp1 = timestamp.replace('T', '')
    timestamp2 = timestamp1.replace('Z', '')

    date_time_obj_utc = dt.datetime.strptime(timestamp2, '%Y-%m-%d%H:%M:%S')

    local_tz = pytz.timezone('Europe/Berlin')
    local_dt = date_time_obj_utc.replace(tzinfo=pytz.utc).astimezone(local_tz)
    return date_time_obj_utc, local_tz.normalize(local_dt)


def find_comesaTime(path: str):
    """finds the timestamp of Xray-Event in given jpg name, and converts it to to the needed format of the SnInfo,
    response, based on '_' """
    timestamp = path.split('_', 3)[1]
    date_time_obj = dt.datetime.strptime(timestamp, '%Y%m%d%H%M%S')

    local_tz = pytz.timezone('Europe/Berlin')
    time = date_time_obj.replace(tzinfo=pytz.utc).astimezone(local_tz)
    utc_offset = '.000'+ str(time)[-6:]
    time = str(time.strftime("%Y-%m-%dT%H:%M:%S"))
    timeComesa = time + utc_offset

    return timeComesa


def add_parentheses(list1):
    list2 = []
    for index, value in enumerate(list1):
        list2.append('(' + value + ')')
    return list2


def dummyfn(msg="foo"):
    print(msg)


def onQMETrigger(url, username, pw, session, engine, QMEpath, connectionTest):
    nrDays = 3
    defec = []
    dataCache = pd.read_sql('SELECT fid, boardnumber, boardside, use_case FROM data_cache', session.bind)
    if connectionTest is not False:
        for i in range(0, nrDays):
            date = (dt.datetime.now() - dt.timedelta(i)).strftime("%d.%m.%Y")
            defects = getResponse(url, username, pw, date)
            defec = defec + defects
        defectsClean = []
        for i in defec:
            if i not in defectsClean:
                defectsClean.append(i)
            else:
                pass

        for item in defectsClean:

            utcDate, regularDate = find_test_time(item['datum'])
            mandant = item['mandant']
            fid_panel = item['fid_panel']
            board_nr = item['board_nr']
            artikel_nr_fbg = item['artikel_nr_fbg']
            lwo = item['lwo']
            fio = item['fio']
            einbauplatz = item['einbauplatz']
            artikel_nr_be = item['artikel_nr_be']
            be_typ = item['be_typ']
            fehlercode = item['fehlercode']
            fehlerart = item['fehlerart']
            fehlerdetail = item['fehlerdetail']
            session = Session()
            entry = QMEEntry()
            if session.query(exists().where(QMEEntry.fid_panel == fid_panel).where(QMEEntry.board_nr == board_nr).\
                where(QMEEntry.einbauplatz == einbauplatz).where(QMEEntry.mandant == mandant).\
                where(QMEEntry.artikel_nr_fbg == artikel_nr_fbg).where(QMEEntry.datum_utc == utcDate)).scalar() is False:

                QMEEntry.addEntry(entry, sess=session, mandant=mandant, fid_panel=fid_panel, board_nr=int(board_nr),
                                  artikel_nr_fbg=artikel_nr_fbg,
                                  datum=regularDate, datum_utc=utcDate, lwo=lwo, fio=fio, einbauplatz=einbauplatz,
                                  artikel_nr_be=artikel_nr_be,
                                  be_typ=be_typ, fehlercode=fehlercode, fehlerart=fehlerart, fehlerdetail=fehlerdetail)
                session.close()
            session.close()
        session.close()

        qme_data = pd.read_sql('SELECT fid_panel, board_nr, einbauplatz, fehlercode, fehlerart FROM qme_data',
                               session.bind)

        for index, row in dataCache.iterrows():

            fid = row['fid']
            bn = row['boardnumber']
            side = row['boardside']
            case = row['use_case']
            label = None
            fehlercode = None
            label, fehlercode = getLabel(qme_data, fid, bn, side)

            if case == 'collect':
                table = FutureBoard
            elif case == 'pred':
                table = Board

            if label is not None:
                session.query(table).where(table.board_number == bn).where(table.fid == fid). \
                    where(table.boardside == side).update({"ground_truth": label, "fehlercode": fehlercode}, synchronize_session="fetch")
                obj = session.query(DataCache).filter(DataCache.fid == fid). \
                    filter(DataCache.boardside == side). \
                    filter(DataCache.board_number == bn). \
                    first()
                session.delete(obj)

                try:
                    session.commit()
                except:
                    session.rollback()
                    raise
                finally:
                    session.close()
                # session.commit()
            session.close()

        else:
            print('kein Eintrag')

    else:
        fields = ['fid_panel', 'board_nr', 'einbauplatz', 'fehlercode', 'fehlerart']
        qme_data_csv = pd.read_csv(QMEpath, sep=',', engine='python',
                               error_bad_lines=False, warn_bad_lines=False, usecols=fields)
        print('Imported QME Database')

        for index, row in dataCache.iterrows():

            fid = row['fid']
            bn = row['boardnumber']
            side = row['boardside']
            case = row['use_case']
            label = None
            fehlercode = None
            label, fehlercode = getLabel(qme_data_csv, fid, bn, side)

            if case == 'collect':
                table = FutureBoard
            elif case == 'pred':
                table = Board

            if label is not None:
                session.query(table).where(table.board_number == bn).where(table.fid == fid).\
                    where(table.boardside == side).update({"ground_truth": label, "fehlercode": fehlercode}, synchronize_session="fetch")
                obj = session.query(DataCache).filter(DataCache.fid == fid). \
                    filter(DataCache.boardside == side). \
                    filter(DataCache.board_number == bn). \
                    first()
                session.delete(obj)
                try:
                    session.commit()
                except:
                    session.rollback()
                    raise
                finally:
                    session.close()
                #session.commit()
            session.close()
        else:
            print('kein Eintrag')


def on_created(event):

    #print(event.src_path)
    start_ppl = time.time()
    # create new filename (raw and jpg) for Minio based on triggered path
    filename = create_filename(event.src_path)
    boardtype = find_type_board(event.src_path)
    jpg = str(Path(filename).stem) + '.jpg'
    side = str(find_boardside(jpg, availableSides))
    fid = find_fid(jpg)
    entry = DataCache()
    if side == 'None':
        try:
            side = str(find_boardside(jpg, futureBoards[boardtype]))
        except:
            #print('No such boardside in model registry or future boards')
            pass
    # definition of raw files
    raw_files = ['rec', 'vgi']

    if (find_type_board(event.src_path)) in dict_map:
        # print(dict_map[boardtype + '_' + side])
        #mapping = pd.read_table(dict_map[find_type_board(event.src_path)[:-2] + '_' + side], header=1)
        #REC_to_QME_dictionary = pd.Series(mapping.QME.values, index=mapping.REC).to_dict()
        board_nr = find_number_board(jpg, dict_map[boardtype], side)
    else:
        #print('No Metafile for: ' + boardtype + ', ' + side + ' avialable!')
        board_nr = 1000

    if any(x in event.src_path for x in raw_files) and boardtype in futureBoards and side in futureBoards[boardtype]:
        time.sleep(1)
        try:
            upload_object_in_bucket(minio, 'rawfuture', create_filename(event.src_path), event.src_path, stream=False,
                                    size=-1)
            # upload_awsS3(clientS3, event.src_path, create_filename(event.src_path), 'bucketname')
            logging.info(f'Upload from {event.src_path} to Minio Server successful!')
        except ResponseError:
            print('Retry upload to minio')
        if file_exist(minio, 'gimsfuture', jpg) is False:
            processType = 'collect'

            dir_rec = os.path.splitext(event.src_path)[0]



            # start gims.jpg preparation pipeline

            golden_image, height, w, h = gims.Preprocessor().CreateAndReturnGims(dir_rec, side, boardtype, cutArray, level=3,
                                                                           number_pics=1)

            # encode and put into buffer
            # save golden_image as gray image from memory in Buffer
            is_success, buffer = cv2.imencode(".jpg", golden_image)
            buf = io.BytesIO(buffer)  # prepare memory buffer
            buf.seek(0)

            raw_img_size = buf.getbuffer().nbytes  # get img size of the Buffer

            # upload finally gims.jpg in Minio
            upload_object_in_bucket(minio, 'gimsfuture', jpg, buf, stream=True, size=raw_img_size)
            logging.info(f'Upload from {jpg} to Minio Server successful!')

            session = Session()
            board = FutureBoard()

            FutureBoard.addBoard(board, sess=session, fid=find_fid(jpg),
                                 name=jpg,
                                 machine=find_machine(event.src_path, machineClient),
                                 boardside=side, timeXray=find_xraytime(jpg), gims_number=height,
                                 type_board=boardtype,
                                 board_number=board_nr)

            try:
                session.commit()
            except:
                session.rollback()
                raise
            finally:
                session.close()
            print('Uploaded raw files and gims for future Model Training of: '+boardtype + ', '+side)

            DataCache.addBoard(entry, sess=session, fid=str(find_fid(jpg)), board_number=board_nr, boardside=side, case=processType)
            session.close()

    # if rec or vgi triggered this event, then store rec and vgi in Minio
    try:
        if any(x in event.src_path for x in raw_files) and any((str(modelType) + str(side)) in s for s in availableModels[boardtype]) and boardtype in availableModels:
            time.sleep(1)
            try:
                upload_object_in_bucket(minio, 'raw', create_filename(event.src_path), event.src_path, stream=False, size=-1)
                #upload_awsS3(clientS3, event.src_path, create_filename(event.src_path), 'bucketname')
                logging.info(f'Upload from {event.src_path} to Minio Server successful!')
            except ResponseError:
                print('Retry upload to minio')

        if any((str(modelType) + str(side)) in s for s in availableModels[boardtype]) and boardtype in availableModels:
            if '.rec' in event.src_path:
                # trigger complete gims and prediction pipeline only when gims isn't stored in Minio yet,
                # to prevent duplicated predictions event triggered from watchdog

                if file_exist(minio, 'gims', jpg) is False:
                    # only new rec file triggers the complete pipeline, consisting out of:
                    # - Golden Image Prep and upload in Minio Bucket 'gims'
                    # - Prediction of Golden Image
                    # - upload model performance in MySQl and logging of all file changes on the directory of the watch

                    processType = 'pred'
                    currentBoard.append(boardtype)
                    currentBoard_AE.append(boardtype)

                    model, version = modelLoad.loadNewestVersionImgClass(s3, 'models', side, boardtype, modelType, currentBoard, currentSide)
                    if dual_model_classification == True:
                        modelAE, version_AE = modelLoad.loadNewestVersionAutoencoder(s3, 'models', side, boardtype, modelTypeAE, currentBoard_AE, currentSide_AE) 
                    else:
                        version_AE = "None"
                    logging.info(
                        f"Loaded and initialized the " + str(modelType) + " model " + str(boardtype) + " " + str(side)
                        + " " + str(version) + " successfully.")
                    if dual_model_classification == True and version_AE != None:
                        logging.info(
                        f"Loaded and initialized the " + str(modelTypeAE) + " autoencoder " + str(boardtype) + " " + str(side)
                        + " " + str(version_AE) + " successfully.")
                    # sleep specific xray_time to fix delay with data Transfer (rec vs vgi)
                    time.sleep(2)

                    start_gims: float = time.time()
                    dir_rec = os.path.splitext(event.src_path)[0]

                    # start gims.jpg preparation pipeline

                    golden_image, height, w, h = gims.Preprocessor().CreateAndReturnGims(dir_rec, side, boardtype, cutArray, level=3, number_pics=1)

                    # encode and put into buffer
                    # save golden_image as gray image from memory in Buffer
                    is_success, buffer = cv2.imencode(".jpg", golden_image)
                    buf = io.BytesIO(buffer)  # prepare memory buffer
                    buf.seek(0)

                    raw_img_size = buf.getbuffer().nbytes  # get img size of the Buffer
                    duration_gims_creation = round((time.time() - start_gims), 2)
                    logging.info(
                        f"Gims-Creation from " + str(event.src_path) + " took " + str(duration_gims_creation) + " seconds.")

                    # upload finally gims.jpg in Minio
                    upload_object_in_bucket(minio, 'gims', jpg, buf, stream=True, size=raw_img_size)
                    logging.info(f'Upload from {jpg} to Minio Server successful!')

                    session = Session()
                    board = Board()
                    if duration_gims_creation >= 60:
                        duration_gims_creation = 59.9
                    Board.addBoard(board, sess=session, name=jpg, machine=find_machine(event.src_path, machineClient), fid=find_fid(jpg),
                                   boardside=side, timeXray=find_xraytime(jpg), gims_number=height,
                                   time_prep=duration_gims_creation, proba_threshold=threshold,
                                   type_board=find_type_board(event.src_path),
                                   board_number=board_nr, model_type=modelType, version=version, mse_threshold=mse_threshold,
                                   autoencoder_type=modelTypeAE, autoencoder_version=version_AE)

                    try:
                        session.commit()
                    except:
                        session.rollback()
                        raise
                    finally:
                        session.close()

                    # PREDICTION from gims.jpg
                    start_pred: float = time.time()
                    if modelType == 'ImgClass2D':
                        pred, proba = PredictionService().pred_image_Class2D(buf, model, modelType)
                        if dual_model_classification == True and version_AE != None:
                            pred_AE, MSE_AE = PredictionService().pred_image_class2D_AE(buf, modelAE)
                        else:
                            pred_AE = "None"
                            MSE_AE = 0
                    elif modelType == 'ImgClass3D':
                        gims.Preprocessor().RecToNii(dir_rec, filename)
                        pred, proba = PredictionService().pred_image_Class3D(filename, model, modelType)

                    if proba < threshold:
                        pred = 'niO'
                    
                    # generating Gradcam of gim.jpg and push it to http server for the dashboard
                    if gradcam_generation == True:
                        gradcam = grad_cam_generation(model=model, gim=buf, w=w, h=h)
                        emptytheservingfolder_and_savegradcam(gim=buf, gradcam=gradcam)
                    start_http_server()

                    # logging duration of prediction and complete pipeline
                    duration_pred = round((time.time() - start_pred), 2)

                    # update Board Prediction in MySQL
                    session = Session()

                    numberRec = get_number_rec(event.src_path)

                    if len(previous_fid) == 0:
                        previous_fid.append(fid)
                    if previous_fid[-1] == fid:
                        nrRecPerFID[0] = nrRecPerFID[0] + 1

                        if not fid in rec_dict:
                            rec_dict[fid] = {'soll': numberRec, 'ist': nrRecPerFID[-1]}
                        rec_dict[fid]['ist'] = nrRecPerFID[0]
                        rec_dict[fid]['soll'] = numberRec
                        if pred == 'niO':
                            set_list.append(generate_Set(set_list, pred, board_nr, side))

                    elif previous_fid[-1] != fid:
                        if len(payload_list) > 0:
                            send_test_result(payload_list[0])
                            print('Sent previous result')
                        payload_list.clear()
                        nrRecPerFID[0] = 1
                        set_list.clear()
                        del rec_dict[previous_fid[-1]]
                        previous_fid[-1] = fid
                        rec_dict[fid] = {'soll': numberRec, 'ist': nrRecPerFID[-1]}
                        if pred == 'niO':
                            set_list.append(generate_Set(set_list, pred, board_nr, side))

                    #print(rec_dict[fid])
                    if rec_dict[fid]['soll'] == rec_dict[fid]['ist']:
                        # label = None
                        # label_list = []
                        # for index in range(len(set_list)):
                        #     for key in set_list[index]:
                        #         if key == "testresult":
                        #             label_list.append(set_list[index][key])
                        # if 'F' in label_list:
                        #     label = 'F'
                        # else:
                        #label = 'P'
                        head_tail = os.path.split(event.src_path)
                        txtPath = os.path.join(head_tail[0], 'InspectionImage.txt')
                        my_file = Path(txtPath)
                        numberRec = get_number_rec(event.src_path)
                        rec_dict[fid]['soll'] = numberRec
                        '''
                        double check the number of rec files again to make sure all rec files are written before the
                        webservice is sent!
                        '''
                        counter = 0
                        while not os.path.exists(txtPath):
                            time.sleep(0.5)
                            counter = counter + 1
                            if counter == 30:
                                break

                        numberRec = get_number_rec(event.src_path)
                        rec_dict[fid]['soll'] = numberRec
                        payload = [fid, comescoSystem, set_list, find_comesaTime(jpg)]
                        payload_list.append(payload)
                        #if label == 'P' and my_file.is_file() and rec_dict[fid]['soll'] == rec_dict[fid]['ist']:
                        if my_file.is_file() and rec_dict[fid]['soll'] == rec_dict[fid]['ist']:
                            send_test_result(payload_list[0])
                            payload_list.clear()
                            print('sent webservice')

                    print(f'Prediction from {event.src_path} took {duration_pred} sec /// Classification Model Prediction: {pred} with probability of {proba} /// Autoencoder Model Prediction: {pred_AE} with MSE of {MSE_AE}')
                    Board.updateBoard(board, sess=session, name=jpg, time_pred=duration_pred, predict_proba=proba, pred=pred, pred_autoencoder=pred_AE, mse=MSE_AE)

                    try:
                        session.commit()
                    except:
                        session.rollback()
                        raise
                    finally:
                        session.close()

                    logging.info(f'Prediction from {event.src_path} took {duration_pred} sec /// Classification Model Prediction: {pred} with probability of {proba} /// Autoencoder Model Prediction: {pred_AE} with MSE of {MSE_AE}')
                    logging.info(f'Complete Pipeline from {event.src_path} took ' + str(round((time.time() - start_ppl), 2))
                                 + ' seconds.')

                    DataCache.addBoard(entry, sess=session, fid=str(find_fid(jpg)), board_number=board_nr,
                                       boardside=side, case=processType)
                    session.close()

    except KeyError as err:
        #print(err)
        #print("No models in Bucket for this product:" + str(err))
        pass


if __name__ == "__main__":

    #variables
    previous_fid = []
    set_list = []
    nrRecPerFID = [0]
    rec_dict = {}
    payload_list = []

    # load specific paths for logging, watching and predicting
    config = ConfigParser()
    config.read_file(open(r'config.txt'))
    path_log = config.get('watching and logging', 'path_log')
    path_mapping_rec_qme = config.get('Boardlayout', 'path_mapping_rec_qme')
    sqlData = config.get('SQLServer', 'sqlData')
    cutArray = config.get('CutX1Images', 'cutArray')
    cutArray = eval(cutArray)
    path_watch = []
    url = config.get('MinIOServer', 'link')
    firstKey = config.get('MinIOServer', 'firstKey')
    secondKey = config.get('MinIOServer', 'secondKey')
    for i in range(1, int(config.get('watching and logging', 'sum_watchpaths')) + 1):
        constructor = 'path_watch'
        path_watch.append(config.get('watching and logging',
                          (constructor + str(i))))
    modelType = config.get('modelSpecification', 'modelType')
    modelTypeAE = config.get('Clustering_model', 'autoencodertype')
    machineClient = config.get('machineName', 'machineClient')
    pathFutureBoards = config.get('FutureBoards', 'pathFutureBoards')
    urlQMEWebserice = config.get('QME_Webservice', 'url')
    usernameQMEWebservice = config.get('QME_Webservice', 'username')
    pwQMEWebservice = config.get('QME_Webservice', 'password')
    updateInterval = int(config.get('QME_Webservice', 'updateInterval'))
    QMEpath = config.get('QME_offline_Database', 'path')
    threshold = float(config.get('Probability_Threshold', 'threshold'))
    mse_threshold = config.getfloat('Clustering_model', 'threshold')
    comescoSystem = config.get('Comesco_System_Type', 'system')
    dual_model_classification  = config.getboolean('Clustering_model', 'dual_model_classification')
    gradcam_generation = config.getboolean('Gradcam', 'gradcam_generation')
    print(urlQMEWebserice, usernameQMEWebservice, pwQMEWebservice)
    print('Comesco connection system: ', comescoSystem)

    # logging config
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[RotatingFileHandler(path_log, maxBytes=1000000, backupCount=100)])

    # Watchdog config trigger pipeline
    # only raw and vgi files can trigger the raw_handler pipeline
    patterns = ("*.vgi", "*.rec",)
    ignore_patterns = ()
    ignore_directories = True
    case_sensitive = True
    # init Prediction Model
    # init Dictionary for Translation Board Layout on Panel (X-Ray vs QME)
    # read mapping.txt, creating Dictionary to translate REC Board Numbers into QME-Format

    dict_map = {}
    for f in os.listdir(path_mapping_rec_qme):
        if os.path.isfile(os.path.join(path_mapping_rec_qme, f)):
            filename = Path(f).stem
            dict_map[filename] = None
            dict_map[filename] = pd.read_csv(os.path.join(path_mapping_rec_qme, f), skiprows=4, sep=',',
                                             engine='python',
                                             error_bad_lines=False, warn_bad_lines=False)
    print('Mapping Dictionary keys:', dict_map.keys())

    #Get all wanted future products to store the relevant data
    futureBoards = {}
    try:
        with open(pathFutureBoards) as f:
            for line in f:
                line = line.replace(' ', '')
                line = line.strip().split(',')
                futureBoards[line[0]] = []
                for side in line[1:]:
                    futureBoards[line[0]].append(side)
    except FileNotFoundError:
        print('No csv found with future boards')
    print(futureBoards)

    #Make Model and NIfTI Directory
    if not os.path.exists('ModelCache'):
        os.makedirs('ModelCache')
    if not os.path.exists('ModelCacheAE'):
        os.makedirs('ModelCacheAE')
    if not os.path.exists('niiCache'):
        os.makedirs('niiCache')
    # Lists for Model selection
    currentBoard = ['spacer']
    currentBoard_AE = ['spacer']
    currentSide = []
    currentSide_AE = []
    # Minio Api Init and S3 client
    minio = minio_api.init_client(url, firstKey, secondKey)
    s3 = init_s3_client(url, firstKey, secondKey)
    #clientS3 = boto3.client('s3')
    # Get all currently available models in the model registry in a dictionary
    availableModels, availableSides = getModelsAndSides(s3, 'models')
    print(availableModels)
    # Mysql and Sqlalchemy Init
    # Define the MySQL engine using MySQL Connector/Python
    engine = db.create_engine(
        sqlData, echo=False, pool_recycle=3600)
    print(engine)

    # Create a session
    Session = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))

    # init logging handler
    log_handler = LoggingEventHandler()

    # init raw_handler for rec and vgi files
    raw_handler = PatternMatchingEventHandler(patterns, ignore_patterns, ignore_directories, case_sensitive)
    raw_handler.on_created = on_created
    #Test if QME webservice connection is available
    connectionTest = getResponse(urlQMEWebserice, usernameQMEWebservice, pwQMEWebservice)
    if connectionTest is False:
        print('The connection to the QME Webservice is not possible, the csv Database on the SambaShare is used instead!')
    # init observer with two different actions (handling of raw objects and log everything)
    # start on two different X-Ray machines, defined in path_watch
    observer = Observer()
    threads = []

    #Build arguments dict for QME Update
    arguments = {'url': urlQMEWebserice,
                 'username': usernameQMEWebservice,
                 'pw': pwQMEWebservice,
                 'session': Session,
                 'engine': engine,
                 'QMEpath': QMEpath,
                 'connectionTest': connectionTest,
                 }
    try:
        rt = RepeatedTimer(updateInterval, onQMETrigger, **arguments)  # it auto-starts, no need of rt.start()
    except KeyboardInterrupt:
        rt.stop()
        atexit.register(exit_func)
    try:
        for i in path_watch:
            targetPath = str(i)
            observer.schedule(raw_handler, targetPath, recursive=True)
            observer.schedule(log_handler, targetPath, recursive=True)
            threads.append(observer)

        # start finally all observer at the same xray_time
        observer.start()
        print(threads)

        # sleep specific xray_time and watch afterwards for changes
        try:
            while True:
                time.sleep(1)

        except KeyboardInterrupt:
            observer.stop()
        observer.join()
        atexit.register(exit_func)
    finally:
        rt.stop()
        pass
