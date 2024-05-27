import requests
import base64
import json

def getResponse(url, username, password, date=None):

    requests.urllib3.disable_warnings()
    connector = username + ':' + password
    connector_bytes = connector.encode("ascii")
    base64_bytes = base64.b64encode(connector_bytes)
    connector64 = base64_bytes.decode("ascii")
    code = 'Basic ' + connector64
    payload = None
    if date is not None:
        payload = {'p_datum': date}
    headers = {
      'Authorization': code
    }

    try:
        response = requests.request("GET", url, headers=headers, params=payload, verify=False)
        result = response.json()
        if result["successful"]:
            print('WebService Connection was successful!')
            return result['defect_data']
        elif not result["successful"]:
            print('WebService Connection was NOT successful!')
            return False

    except:
        return False


def getLabel(qme_data, fid, board_nr, side):

    '''defines label of fid (board_nr, side) stored in (data_chache)SQL.
    
    Label based on a) webservice call from QME b) Local QME database on sharepoint
    
    First (if): Looks for direct hit in QME database and defines label
    
    Second (elif): If no direct hit fid (board_nr, side) is found, then a check only for the FID is done (not board_nr & side), 
    
    if an entry for the fid is found in the qme_data, the label iO is given, as the fid must have been tested but no failure was found for the specific fid (board_nr, side)

    '''

    qme_data['fehlercode'] = qme_data['fehlercode'].astype(int)
    label = None
    errorCode = None
    pseudoerrorlist = [821, 1800, 815]
    mask = (qme_data['fid_panel'] == fid) & (qme_data['board_nr'] == board_nr) & (qme_data['einbauplatz'] == side)
    errorCodeList = (qme_data['fehlercode'][mask]).to_list()
    
    if len(errorCodeList) > 0:
        errorCode = errorCodeList[0]
        
        if errorCode in pseudoerrorlist:
            label = 'iO'
            errorCode = 000 # Our own pseudo-error code in the database
            print(fid, board_nr, side, label)
        else:
            label = 'niO'
            print(fid, board_nr, side, label)
    
    elif len(errorCodeList) == 0:
        mask = (qme_data['fid_panel'] == fid)
        fidList = (qme_data['fid_panel'][mask]).to_list()
        
        if len(fidList) > 0:
            label = 'iO'
            print(fid, board_nr, side, label)
            
    return label, errorCode
