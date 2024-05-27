import mysql.connector
mydb = mysql.connector.connect(
      host="localhost",
      user="root",
      password="admin",
      database="xraydb"
    )

sql_Query = f"SELECT * FROM xraydb.board_pred where ground_truth='niO' limit 1;"

mycursor = mydb.cursor()
mycursor.execute(sql_Query)
records = mycursor.fetchall()

print(records)