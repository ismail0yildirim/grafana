import psycopg2
postgres_ptpclient = psycopg2.connect(
                            database="axi_test", 
                            user="postgres", 
                            password="admin", 
                            host="localhost", 
                            port="5432")

postgres_local = psycopg2.connect(
                            database="xraydb", 
                            user="postgres", 
                            password="admin", 
                            host="localhost", 
                            port="5432")



fid = "C-P9EP5250"


# WRITE DATA INTO LOCAL OMRON
cursor = postgres_ptpclient.cursor()
sql_Query = f"""SELECT rb.barcode, cb.programname, cc.blocknum, cc.compname, cw.pinno, rb.testendtime, al.algoname, mt.algoelementname, mt.unit, mt.upperlowerlimit, mv.inspectioncriterion, mv.inspectionmeasurement 
                FROM rns_resultboard rb 
                INNER JOIN rns_cadcomponent cc ON rb.cadboardidx=cc.cadboardidx 
                INNER JOIN rns_cadwindow cw ON cc.idx=cw.cadcomponentidx 
                INNER JOIN rns_measurevalue mv ON cw.windowno=mv.windowno 
                AND rb.idx=mv.resultboardidx 
                INNER JOIN rns_measuretype mt ON mv.measuretypeidx=mt.idx 
                INNER JOIN rns_algorithm al ON mt.algorithmidx=al.idx 
                INNER JOIN rns_cadboard cb ON cb.idx = cc.cadboardidx 
                WHERE rb.barcode = '{fid}' and (cc.compname='X1' OR cc.compname='X2')"""



cursor.execute(sql_Query)
records = cursor.fetchall()
# print(len(records))
# print(records[0])
j = 3
i = j
# print(f"{records[i][0]}, {records[i][1]}, {records[i][2]}, {records[i][3]}, {records[i][4]}, {records[i][5]}, {records[i][6]}, {records[i][7]}, {records[i][8]}, {records[i][9]}, {records[i][10]}, {records[i][11]}")



################################################################################################### FUNCTION

def insert_into_omron(connection, fid, a5e_number, boardnumber, boardside, pinnumber, testendtime, measurement_element, measurement_type, unit, upperlowerlimit, inspectioncriterion, inspectionmeasurement):
    pinnumber = pinnumber-1
    if inspectioncriterion==None:
        inspectioncriterion='NULL'
    
    if upperlowerlimit==0:
        try:
            if inspectioncriterion<inspectionmeasurement:
                label_test='iO'
            elif inspectioncriterion>=inspectionmeasurement:
                label_test='niO'
        except:
            label_test='null'
    elif upperlowerlimit==1:
        try:
            if inspectioncriterion>=inspectionmeasurement:
                label_test='iO'
            elif inspectioncriterion<inspectionmeasurement:
                label_test='niO'
        except:
            label_test='null'
    
    cursor = connection.cursor()
    
    if label_test=='null':
        postgres_insert_query = f"""INSERT INTO omron (fid, type, boardnumber, boardside, pinnumber, testendtime, measurement_element, measurement_type, unit, upperlowerlimit, inspectioncriterion, inspectionmeasurement) 
                                        VALUES ('{fid}', '{a5e_number}', '{boardnumber}', '{boardside}', {pinnumber}, '{testendtime}', '{measurement_element}', '{measurement_type}', '{unit}', {upperlowerlimit}, {inspectioncriterion}, {inspectionmeasurement});"""
    else:
        postgres_insert_query = f"""INSERT INTO omron (fid, type, boardnumber, boardside, pinnumber, testendtime, measurement_element, measurement_type, unit, upperlowerlimit, inspectioncriterion, inspectionmeasurement, label_test) 
                                        VALUES ('{fid}', '{a5e_number}', '{boardnumber}', '{boardside}', {pinnumber}, '{testendtime}', '{measurement_element}', '{measurement_type}', '{unit}', {upperlowerlimit}, {inspectioncriterion}, {inspectionmeasurement}, '{label_test}');"""
    cursor.execute(postgres_insert_query)
    connection.commit()
    cursor.close()



################################################################################################### FUNCTION


for j in range(len(records)):
    insert_into_omron(connection=postgres_local, fid=records[j][0], a5e_number=records[j][1], boardnumber=records[j][2], boardside=records[j][3], pinnumber=records[j][4], testendtime=records[j][5], 
                    measurement_element=records[j][6], measurement_type=records[j][7], unit=records[j][8], upperlowerlimit=records[j][9], inspectioncriterion=records[j][10], inspectionmeasurement=records[j][11])
    
