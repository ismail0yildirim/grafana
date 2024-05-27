import pandas as pd

path = r"C:\Users\Z004KVJF\Desktop\QME_Database.csv"

data = pd.read_csv(path)

# print(data)


# duplicates = data[data.duplicated(['fid_panel'], keep=False)]

duplicates = data[data[['fid_panel', 'board_nr', 'einbauplatz']].duplicated()]


# print(duplicates)

print(data[data["fid_panel"]=="C-PDSW2390"])