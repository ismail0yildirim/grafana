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
postgres_local = psycopg2.connect(
                            database="xraydb", 
                            user="postgres", 
                            password="admin", 
                            host="localhost", 
                            port="5432")


mycursor = my_sql.cursor()
sql_Query = "SELECT distinct(fid) FROM xraydb.board_pred"
mycursor.execute(sql_Query)
records = mycursor.fetchall()

print("Länge neue FIDs: ", len(records))

fids_board_pred_my_sql = []
for i in range(len(records)):
    fids_board_pred_my_sql.append(records[i][0])

######################---------------------------------------------------------------------------------######################


def insert_into_boardpred(connection, name, machine, type, fid, boardnumber, xray_time, boardside, gims_number, time_prep, time_pred, pred, predict_proba, proba_threshold, model_type, model_version, stamp_created, stamp_updated, ground_truth, fehlercode, mse_threshold, mse, autoencoder_type, autoencoder_version, pred_autoencoder):
        
    cursor = connection.cursor()
    postgres_insert_query = f"""INSERT INTO board_pred (name, machine, type, fid, boardnumber, xray_time, boardside, gims_number, time_prep, time_pred, pred, predict_proba, proba_threshold, model_type, model_version, stamp_created, stamp_updated, ground_truth, fehlercode, mse_threshold, mse, autoencoder_type, autoencoder_version, pred_autoencoder) 
                            VALUES ('{name}', '{machine}', '{type}', '{fid}', {boardnumber}, '{xray_time}', '{boardside}', {gims_number}, '{time_prep}', '{time_pred}', '{pred}', {predict_proba}, {proba_threshold}, '{model_type}', '{model_version}', {stamp_created}, {stamp_updated}, {ground_truth}, '{fehlercode}', {mse_threshold}, {mse}, {autoencoder_type}, {autoencoder_version}, {pred_autoencoder});"""
    cursor.execute(postgres_insert_query)
    connection.commit()
    cursor.close()


for i in range(len(fids_board_pred_my_sql)):
    cursor = my_sql.cursor()
    sql_Query = f"SELECT * FROM xraydb.board_pred where fid='{fids_board_pred_my_sql[i]}';"
    cursor.execute(sql_Query)
    records = cursor.fetchall()
    # print(len(records))
    # print(records[0])
    for j in range(len(records)):
        insert_into_boardpred(connection=postgres_local, name=records[j][1], machine=records[j][2], type=records[j][3], fid=records[j][4], boardnumber=records[j][5], xray_time=records[j][6], 
        boardside=records[j][7], gims_number=records[j][8], time_prep=, time_pred=records[j][9], pred=records[j][10], predict_proba=records[j][11], proba_threshold=records[j][12], model_type=records[j][13], 
        model_version=records[j][14], stamp_created=records[j][15], stamp_updated=records[j][16], ground_truth=records[j][17], fehlercode=records[j][18], mse_threshold=records[j][19], 
        mse=records[j][20], autoencoder_type=records[j][21], autoencoder_version=records[j][22], pred_autoencoder=records[j][23])
    
    print(f"fids imported in board_pred: {i+1}/{len(fids_board_pred_my_sql)}")