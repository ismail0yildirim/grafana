import sqlalchemy as db
from sqlalchemy.types import DateTime, Float, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.exc import SQLAlchemyError

Base = declarative_base()


class Board(Base):
    __tablename__ = 'board_pred'
    id = db.Column(Integer, primary_key=True)
    name = db.Column('name', String())
    machine = db.Column('machine', String())
    fid = db.Column('fid', String())
    boardside = db.Column('boardside', String())
    xray_time = db.Column('xray_time', DateTime())
    gims_number = db.Column('gims_number', Integer())
    time_prep = db.Column('time_prep', String())
    time_pred = db.Column('time_pred', String())
    predict_proba = db.Column('predict_proba', Float(precision=32, decimal_return_scale=None))
    proba_threshold = db.Column('proba_threshold', Float(precision=32, decimal_return_scale=None))
    pred = db.Column('pred', String(10))
    type_board = db.Column('type', String())
    board_number = db.Column('boardnumber', Integer())
    model_type = db.Column('model_type', String())
    version = db.Column('model_version', String())
    ground_truth = db.Column('ground_truth', String())
    fehlercode = db.Column('fehlercode', String())
    pred_autoencoder = db.Column('pred_autoencoder', String())
    mse = db.Column('mse', Float(precision=32, decimal_return_scale=None))
    mse_threshold = db.Column('mse_threshold', Float(precision=32, decimal_return_scale=None))
    autoencoder_type = db.Column('autoencoder_type', String())
    autoencoder_version = db.Column('autoencoder_version', String())

    def addBoard(self, sess, name, machine, fid, boardside, timeXray, gims_number, time_prep,proba_threshold, type_board, board_number,
                 model_type, version, mse_threshold, autoencoder_type, autoencoder_version):
        """
        adds new board to the SQL Database, with these specific values

        :param name: name of the jpg, stored in Minio
        :param sess: instance of session should be used for SQL Manipulations
        :param machine: which machine produced XRay
        :param fid: FID number of the board
        :param boardside: which side of the board has scanned
        :param timeXray: on which timestamp GMT+1 xray event has happened
        :param gims_number: which picture of the picture stack has been selected for golden image
        :param time_prep: xray_time took gims preperation
        :param type_board: type Number of the Board (A5E....)
        :param board_number: Number of the given Board on the Panel (A5E....)
        :return:
        """
        try:
            row = Board(name=name, machine=machine, fid=fid, boardside=boardside, xray_time=timeXray,
                        gims_number=gims_number, time_prep=time_prep, proba_threshold=proba_threshold,
                        type_board=type_board, board_number=board_number,
                        model_type=model_type, version=version, mse_threshold=mse_threshold, autoencoder_type=autoencoder_type, autoencoder_version=autoencoder_version)
            sess.add(row)
            sess.commit()
        except SQLAlchemyError as e:
            print(type(e))

    def updateBoard(self, sess, name, time_pred, predict_proba, pred, pred_autoencoder, mse):
        """
        updates the given board on given name with specific values in MySQl server

        :param sess: instance of session should be used for SQL Manupulatings
        :param name: name of the jpg, stored in Minio
        :param time_pred: xray_time took prediction of the gims picture
        :param predict_proba: prediction probability of the specifc decission of the model
        :param pred: prediction or Label of the model for specific jpg
        :return:
        """
        try:
            sess.query(Board).filter_by(name=name).update({"time_pred": time_pred, "predict_proba": predict_proba, "pred": pred, "pred_autoencoder": pred_autoencoder, "mse": mse})
            sess.commit()
        except SQLAlchemyError as e:
            print(type(e))

class FutureBoard(Base):
    __tablename__ = 'future_data_storage'
    id = db.Column(Integer, primary_key=True)
    name = db.Column('name', String())
    machine = db.Column('machine', String())
    fid = db.Column('fid', String())
    boardside = db.Column('boardside', String())
    xray_time = db.Column('xray_time', DateTime())
    gims_number = db.Column('gims_number', Integer())
    type_board = db.Column('type', String())
    board_number = db.Column('boardnumber', Integer())
    ground_truth = db.Column('ground_truth', String())
    fehlercode = db.Column('fehlercode', String())

    def addBoard(self, sess, name, machine, fid, boardside, timeXray, gims_number, type_board, board_number):
        """
        adds new board to the SQL Database, with these specific values

        :param name: name of the jpg, stored in Minio
        :param sess: instance of session should be used for SQL Manipulations
        :param machine: which machine produced XRay
        :param fid: FID number of the board
        :param boardside: which side of the board has scanned
        :param timeXray: on which timestamp GMT+1 xray event has happened
        :param gims_number: which picture of the picture stack has been selected for golden image
        :param type_board: type Number of the Board (A5E....)
        :param board_number: Number of the given Board on the Panel (A5E....)
        :return:
        """
        try:
            row = FutureBoard(name=name, machine=machine, fid=fid, boardside=boardside, xray_time=timeXray,
                        gims_number=gims_number, type_board=type_board, board_number=board_number)
            sess.add(row)
            sess.commit()
        except SQLAlchemyError as e:
            print(type(e))

class QMEEntry(Base):
    __tablename__ = 'qme_data'
    id = db.Column(Integer, primary_key=True)
    mandant = db.Column('mandant', String())
    fid_panel = db.Column('fid_panel', String())
    board_nr = db.Column('board_nr', Integer())
    artikel_nr_fbg = db.Column('artikel_nr_fbg', String())
    datum = db.Column('datum', DateTime())
    datum_utc = db.Column('datum_utc', DateTime())
    lwo = db.Column('lwo', String())
    fio = db.Column('fio', String())
    einbauplatz = db.Column('einbauplatz', String())
    artikel_nr_be = db.Column('artikel_nr_be', String())
    be_typ = db.Column('be_typ', String(10))
    fehlercode = db.Column('fehlercode', String())
    fehlerart = db.Column('fehlerart', String())
    fehlerdetail = db.Column('fehlerdetail', String())

    def addEntry(self, sess, mandant, fid_panel, board_nr, artikel_nr_fbg, datum, datum_utc, lwo, fio, einbauplatz, artikel_nr_be,
                 be_typ, fehlercode, fehlerart, fehlerdetail):

        """
        adds new board to the SQL Database, with these specific values

        :param mandant: Plant where panel was scanned
        :param sess: instance of session should be used for SQL Manipulations
        :param fid_panel: fid of the scanned panel (barcode)
        :param board_nr: Number of the given Board on the Panel
        :param artikel_nr_fbg: A5E number of the pcb assembly
        :param datum: timestamp on which the board was checked
        :param lwo: respective lwo
        :param fio: respective fio
        :param einbauplatz: component on the board where the error occurred
        :param be_typ: component description
        :param fehlercode: code of the respective error
        :param fehlerart: description of the error
        :param fehlerdetail: details about the error
        """
        try:
            row = QMEEntry(mandant=mandant, fid_panel=fid_panel, board_nr=board_nr, artikel_nr_fbg=artikel_nr_fbg,
                           datum=datum, datum_utc=datum_utc, lwo=lwo, fio=fio, einbauplatz=einbauplatz, artikel_nr_be=artikel_nr_be,
                           be_typ=be_typ, fehlercode=fehlercode, fehlerart=fehlerart, fehlerdetail=fehlerdetail)
            sess.add(row)
            sess.commit()
        except SQLAlchemyError as e:
            print(type(e))

class DataCache(Base):
    __tablename__ = 'data_cache'
    id = db.Column(Integer, primary_key=True)
    fid = db.Column('fid', String())
    boardside = db.Column('boardside', String())
    board_number = db.Column('boardnumber', Integer())
    case = db.Column('use_case', String())

    def addBoard(self, sess, fid, boardside, board_number, case):
        """
        adds new board to the SQL Database, with these specific values

        :param sess: instance of session should be used for SQL Manipulations
        :param fid: FID number of the board
        :param boardside: which side of the board has scanned
        :param board_number: Number of the given Board on the Panel (A5E....)
        :return:
        """
        try:
            row = DataCache(fid=fid, boardside=boardside, board_number=board_number, case=case)
            sess.add(row)
            sess.commit()
        except SQLAlchemyError as e:
            print(type(e))

class DriftDetection(Base):
    __tablename__ = 'driftdetection'
    id = db.Column(Integer, primary_key=True)
    time = db.Column('time', DateTime())
    model_type = db.Column('model_type', String())
    type_board = db.Column('type', String())
    boardside = db.Column('boardside', String())
    version = db.Column('model_version', String())
    drift_type = db.Column('drift_type', String())
    feature = db.Column('feature', String())
    drift_detected_ks_test = db.Column('drift_detected_ks_test', String())
    ks_test = db.Column('ks_test', Float(precision=32, decimal_return_scale=None))
    alpha_level_ks_test = db.Column('alpha_level_ks_test', Float(precision=32, decimal_return_scale=None))
    percentage_pixels_drifted = db.Column('percentage_pixels_drifted', Float(precision=32, decimal_return_scale=None))
    threshold_pixel_drift = db.Column('threshold_pixel_drift', Float(precision=32, decimal_return_scale=None))
    overall_pixel_drift_detected = db.Column('overall_pixel_drift_detected', String())
    URL = db.Column('URL', String())

    def addDrift(self, sess, time, model_type, type_board, boardside, version, drift_type, feature, drift_detected_ks_test, ks_test, alpha_level_ks_test, percentage_pixels_drifted, threshold_pixel_drift, overall_pixel_drift_detected, URL):
        """
        adds new drift to the SQL Database, with these specific values

        :param sess: instance of session should be used for SQL Manipulations
        :param id:
        :param time: 
        :param model_type:
        :param type_board:
        :param boardside:
        :param version:
        :param drift_type:
        :param feature:
        :param drift_detected_ks_test:
        :param ks_test:
        :return:
        """
        try:
            row = DriftDetection(time=time, model_type=model_type, type_board=type_board, boardside=boardside, version=version, drift_type=drift_type, feature=feature, drift_detected_ks_test=drift_detected_ks_test, ks_test=ks_test, alpha_level_ks_test=alpha_level_ks_test, percentage_pixels_drifted=percentage_pixels_drifted, threshold_pixel_drift=threshold_pixel_drift, overall_pixel_drift_detected=overall_pixel_drift_detected, URL=URL)
            sess.add(row)
            sess.commit()
        except SQLAlchemyError as e:
            print(type(e))