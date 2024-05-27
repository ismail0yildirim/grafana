import time


def writeCSV(fid: str, boardnr: str, boardside: str, path: str, comesaTime: str, processType: str):

    while True:
        try:
            with open(path, "r") as f:
                lines = f.readlines()
            with open(path, "w") as f:
                for line in lines:
                    if (line.strip("\n")).rsplit(",", 1)[0] != (fid + "," + boardnr + "," + boardside):
                        f.write(line)
            with open(path,'a+') as f:
                f.seek(0)
                data = f.read(100)
                if len(data) > 0:
                    f.write("\n")
                f.write(fid)
                f.write(',')
                f.write(boardnr)
                f.write(',')
                f.write(boardside)
                f.write(',')
                f.write(comesaTime)
                f.write(',')
                f.write(processType)
        except PermissionError:
            print("Failed writing to csv, try again")
            time.sleep(2)
            continue
        break
