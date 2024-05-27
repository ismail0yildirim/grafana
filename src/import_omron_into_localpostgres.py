from configparser import ConfigParser
config = ConfigParser()
config.read_file(open(r'config.txt'))
MinIO_path = config.get('MinIOServer', 'path_local_pc')
sqlData = config.get('SQLServer', 'sqlData')
localhost_SQL = config.get('SQLServer', 'host')
user_SQL = config.get('SQLServer', 'user')
password_SQL = config.get('SQLServer', 'password')
database_SQL = config.get('SQLServer', 'database')
import mysql.connector
my_sql = mysql.connector.connect(
      host=localhost_SQL,
      user=user_SQL,
      password=password_SQL,
      database=database_SQL
    )


import psycopg2
postgres_ptpclient = psycopg2.connect(
                            database="Omron", 
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


def create_payload_query(query_file, xray_time_max):

    with open(query_file, 'r') as f:

        # search last line for last payload sending
        for line in f:
            pass
        last_line = line

        # get query from file
        my_query = last_line
        # print("New SQL query for sending: ", my_query)
        sql_select_Query = str(my_query)
        f.close()

    # add new query for next time to file
    with open(query_file, 'a') as f:

        f.write("\n" + f"SELECT distinct(fid) FROM xraydb.board_pred where xray_time > '{xray_time_max}';")
        f.close()

    return sql_select_Query


################################################################################################### FUNCTION

mycursor = my_sql.cursor()
sql_Query = f"SELECT MAX(xray_time) FROM xraydb.board_pred;"
mycursor.execute(sql_Query)
records = mycursor.fetchall()
xray_time_max = records[0][0]

sql_select_Query = create_payload_query('query_file_omron.txt', xray_time_max)

mycursor = my_sql.cursor()
sql_Query = sql_select_Query
mycursor.execute(sql_Query)
records = mycursor.fetchall()

print("Länge neue FIDs: ", len(records))

fids_board_pred_my_sql = []
for i in range(len(records)):
    fids_board_pred_my_sql.append(records[i][0])

# FILTER IF IN OMRON
fids_in_omron = []
j = 0
for i in range(len(fids_board_pred_my_sql)):
    cursor = postgres_ptpclient.cursor()
    # sql_Query = f"SELECT rb.barcode FROM rns_resultboard rb INNER JOIN rns_cadcomponent cc ON rb.cadboardidx=cc.cadboardidx WHERE rb.barcode = '{fids_board_pred_my_sql[i]}' and (cc.compname='X1' OR cc.compname='X2')"
    sql_Query = f"SELECT rb.barcode, cb.programname, cc.blocknum, cc.compname, cw.pinno, rb.testendtime, al.algoname, mt.algoelementname, mt.unit, mt.upperlowerlimit, mv.inspectioncriterion, mv.inspectionmeasurement FROM rns_resultboard rb INNER JOIN rns_cadcomponent cc ON rb.cadboardidx=cc.cadboardidx INNER JOIN rns_cadwindow cw ON cc.idx=cw.cadcomponentidx INNER JOIN rns_measurevalue mv ON cw.windowno=mv.windowno AND rb.idx=mv.resultboardidx INNER JOIN rns_measuretype mt ON mv.measuretypeidx=mt.idx INNER JOIN rns_algorithm al ON mt.algorithmidx=al.idx INNER JOIN rns_cadboard cb ON cb.idx = cc.cadboardidx WHERE rb.barcode = '{fids_board_pred_my_sql[i]}' and (cc.compname='X1' OR cc.compname='X2')"
    cursor.execute(sql_Query)
    records = cursor.fetchall()
    if len(records)==0:
        pass
    else:
        fids_in_omron.append(fids_board_pred_my_sql[i])
        j = j + 1
        print("FID found in Omron: ", j, " : ", fids_board_pred_my_sql[i])

# FILTER IF ALREADY IN LOCAL OMRON
cursor = postgres_local.cursor()
sql_Query = f"SELECT distinct fid FROM omron"
cursor.execute(sql_Query)
records = cursor.fetchall()

fids_omron_postgres_local = []
for i in range(len(records)):
    fids_omron_postgres_local.append(records[i][0])

fids_missing = []
for i in range(len(fids_in_omron)):
    found = 0
    for j in range(len(fids_omron_postgres_local)):
        if fids_in_omron[i] == fids_omron_postgres_local[j]:
            found = 1
    if found == 0:
        fids_missing.append(fids_in_omron[i])

# WRITE DATA INTO LOCAL OMRON
for i in range(len(fids_missing)):
    cursor = postgres_ptpclient.cursor()
    sql_Query = f"SELECT rb.barcode, cb.programname, cc.blocknum, cc.compname, cw.pinno, rb.testendtime, al.algoname, mt.algoelementname, mt.unit, mt.upperlowerlimit, mv.inspectioncriterion, mv.inspectionmeasurement  FROM rns_resultboard rb INNER JOIN rns_cadcomponent cc ON rb.cadboardidx=cc.cadboardidx INNER JOIN rns_cadwindow cw ON cc.idx=cw.cadcomponentidx INNER JOIN rns_measurevalue mv ON cw.windowno=mv.windowno AND rb.idx=mv.resultboardidx INNER JOIN rns_measuretype mt ON mv.measuretypeidx=mt.idx INNER JOIN rns_algorithm al ON mt.algorithmidx=al.idx INNER JOIN rns_cadboard cb ON cb.idx = cc.cadboardidx WHERE rb.barcode = '{fids_missing[i]}' and (cc.compname='X1' OR cc.compname='X2')"
    cursor.execute(sql_Query)
    records = cursor.fetchall()
    # print(len(records))
    # print(records[0])
    for j in range(len(records)):
        insert_into_omron(connection=postgres_local, fid=records[j][0], a5e_number=records[j][1], boardnumber=records[j][2], boardside=records[j][3], pinnumber=records[j][4], testendtime=records[j][5], 
                    measurement_element=records[j][6], measurement_type=records[j][7], unit=records[j][8], upperlowerlimit=records[j][9], inspectioncriterion=records[j][10], inspectionmeasurement=records[j][11])
    
    print(f"fids imported in omron: {i+1}/{len(fids_missing)}")

