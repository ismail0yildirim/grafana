from sqlalchemy_api import Omron

import psycopg2
connection = psycopg2.connect(
                            database="Omron", 
                            user="postgres", 
                            password="admin", 
                            host="localhost", 
                            port="5432")

from configparser import ConfigParser
config = ConfigParser()
config.read_file(open(r'config.txt'))
sqlData = config.get('SQLServer', 'sqlData')
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy_api import *
engine = db.create_engine(sqlData, echo=False, pool_recycle=3600)
Session = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))
session = Session()

def addMeasurementtoSQLomrontable(session, fid, type_board, board_number, boardside, linenumber, pin_number, algoname, algoelementname, unit, upperlowerlimit, inspectioncriterion, inspectionmeasurement, result):
    omron = Omron()
    Omron.addMeasurement(omron, sess=session, fid=fid, type_board=type_board, board_number=board_number, boardside=boardside, linenumber=linenumber, pin_number=pin_number, algoname=algoname, algoelementname=algoelementname, unit=unit, upperlowerlimit=upperlowerlimit, inspectioncriterion=inspectioncriterion, inspectionmeasurement=inspectionmeasurement, result=result)

import pandas as pd
path_fids = r'C:\Users\Z004KVJF\Desktop\fidsinomron.csv'
fids_df = pd.read_csv(path_fids)
# print(fids_df)
fids_df = fids_df.drop_duplicates()
# print(len(fids_df))
fids_array = fids_df.values.tolist()
# print(fids_array[0][0])

j = 0

for i in range(len(fids_array)):

    cursor = connection.cursor()
    sql_Query = f"SELECT rb.barcode, cb.programname, cc.blocknum, cc.compname, mt.lineno, fd.pinno, al.algoname, mt.algoelementname, mt.unit, mt.upperlowerlimit, mv.inspectioncriterion, mv.inspectionmeasurement FROM rns_resultboard rb INNER JOIN rns_cadcomponent cc ON rb.cadboardidx=cc.cadboardidx INNER JOIN rns_cadwindow cw ON cc.idx=cw.cadcomponentidx INNER JOIN rns_measurevalue mv ON cw.windowno=mv.windowno INNER JOIN rns_cadboard cb ON cc.cadboardidx=cb.idx INNER JOIN rns_faultdetails fd ON mv.windowno=fd.windowno AND rb.idx=mv.resultboardidx INNER JOIN rns_measuretype mt ON mv.measuretypeidx=mt.idx INNER JOIN rns_algorithm al ON mt.algorithmidx=al.idx WHERE rb.barcode = '{fids_array[i][0]}' and (cc.compname='X1' OR cc.compname='X2')"
    # sql_Query = f"SELECT rb.barcode, cb.programname, cc.blocknum, cc.compname, cw.pinno, rb.inspectionmachine, cb.createmachine, al.algoname, mt.algoelementname, mt.unit, mt.upperlowerlimit, mv.inspectioncriterion, mv.inspectionmeasurement FROM rns_resultboard rb INNER JOIN rns_cadcomponent cc ON rb.cadboardidx=cc.cadboardidx INNER JOIN rns_cadwindow cw ON cc.idx=cw.cadcomponentidx INNER JOIN rns_measurevalue mv ON cw.windowno=mv.windowno INNER JOIN rns_cadboard cb ON cc.cadboardidx=cb.idx INNER JOIN rns_faultdetails fd ON mv.windowno=fd.windowno AND rb.idx=mv.resultboardidx INNER JOIN rns_measuretype mt ON mv.measuretypeidx=mt.idx INNER JOIN rns_algorithm al ON mt.algorithmidx=al.idx WHERE rb.barcode = 'C-P9310485' and (cc.compname='X1' OR cc.compname='X2')"
    cursor.execute(sql_Query)
    records = cursor.fetchall()

    if len(records)==0:
        pass
    else:
        for i in range(len(records)):

            # print(records[i])

            if records[i][9]==0:
                try:
                    if records[i][10]<records[i][11]:
                        result='iO'
                    elif records[i][10]>=records[i][11]:
                        result='niO'
                except:
                    result=None
            elif records[i][9]==1:
                try:
                    if records[i][10]>=records[i][11]:
                        result='iO'
                    elif records[i][10]<records[i][11]:
                        result='niO'
                except:
                    result=None
            
            addMeasurementtoSQLomrontable(session=session, fid=records[i][0], type_board=records[i][1], board_number=records[i][2], boardside=records[i][3], linenumber=records[i][4], pin_number=records[i][5], algoname=records[i][6], algoelementname=records[i][7], unit=records[i][8], upperlowerlimit=records[i][9], inspectioncriterion=records[i][10], inspectionmeasurement=records[i][11], result=result)
    j = j + 1
    print(j, "/", len(fids_df))