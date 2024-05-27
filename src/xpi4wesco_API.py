from zeep import Client
from zeep.transports import Transport
from datetime import datetime
import ssl
from requests import Session, urllib3
import pytz

import lxml

def xpiTime():
    """finds the timestamp of Xray-Event in given jpg name, and converts it to to the needed format of the SnInfo,
    response, based on '_' """
    #timestamp = path.split('_', 3)[1]
    #date_time_obj = dt.datetime.strptime(timestamp, '%Y%m%d%H%M%S')
    date_time_obj = datetime.utcnow()
    local_tz = pytz.timezone('Europe/Berlin')
    time = date_time_obj.replace(tzinfo=pytz.utc).astimezone(local_tz)
    utc_offset = '.000'+ str(time)[-6:]
    time = str(time.strftime("%Y-%m-%dT%H:%M:%S"))
    timeComesa = time + utc_offset
    return timeComesa


def generate_Set(setList, pred, bn, side):
    seq_nr = len(setList) + 1
    if pred == 'niO':
        label = 'F'
        defect_type = '226'
        defect_group = 'xy'
    elif pred == 'iO':
        label = 'P'
        defect_group = '18'
        defect_type = '00'

    setDict = {"sequence_nr": str(seq_nr),
                       "kind_of_test": "8131", "testresult": label, "board_nr": bn, "reference_designator": side,
                       "defect_type": defect_type}
    return setDict


def send_repair_result(payload_list):

    fid = payload_list[0]
    system_type = payload_list[1]
    set_list = payload_list[2]
    comesa_time = payload_list[3]
    #Definition of the urls dependent on the system type to write in

    if system_type == 'p' or system_type == 'P':
        webservice_wsdl = "https://intra7.amb2.siemens.de/axis2/services/xpi4wescoWs?wsdl"
        webservice_url = 'https://intra7.amb2.siemens.de/axis2/services/xpi4wescoWs/sendProtokoll'
    elif system_type == 'q' or system_type == 'Q':
        webservice_wsdl = "https://intra7-q.amb2.siemens.de/axis2/services/xpi4wescoWs?wsdl"
        webservice_url = 'https://intra7-q.amb2.siemens.de/axis2/services/xpi4wescoWs/sendProtokoll'



    session = Session()
    session.verify = False
    urllib3.disable_warnings()
    try:
        transport = Transport(session=session)
        ssl._create_default_https_context = ssl._create_unverified_context
        timestamp = datetime.utcnow()
        time = str(timestamp.strftime("%Y-%m-%dT%H:%M:%S.%f%z"))
        time = time[:-3]
        time = time + "+00:00"
        client = Client(webservice_wsdl, transport=transport)
        service = client.create_service('{http://com.siemens.make.webService.xpi4wesco/}xpi4wescoBinding',
                                        webservice_url)

        request_xml = {'xpi4wesco':{'header':{'serialnumber': fid, 'datetime': xpiTime(), 'testresult': 'F',
                        'source': 'W', 'protocolType': 'R'}, "repairprot": {"comesaRefMap_p": {"comesa_ref":{"fio": "I98","processType":"O_AI" }}, "test_datetime": comesa_time, "repair_data": {"set":set_list}},'comesaRefMap':{'comesa_ref': {'fio': 'I98', 'processType': 'O_AI'}},
                                    },
                       'ebb_appl': 'Xpi4wescoWsClient_dotNet4.0', 'ebb_appl_v': '2.2.2_WSDL1.10_python_upload'}

        response_xml = service.sit(**request_xml)
        #print('Connection of webservice was succesfull', response_xml['retStatus'])
    except:
        print('Webservice connection was not succesfull')
        pass


def send_test_result(payload_list):

    fid = payload_list[0]
    system_type = payload_list[1]
    set_list = payload_list[2]
    if system_type == 'p' or system_type == 'P':
        webservice_wsdl = "https://intra7.amb2.siemens.de/axis2/services/xpi4wescoWs?wsdl"
        webservice_url = 'https://intra7.amb2.siemens.de/axis2/services/xpi4wescoWs/sendProtokoll'
    elif system_type == 'q' or system_type == 'Q':
        webservice_wsdl = "https://intra7-q.amb2.siemens.de/axis2/services/xpi4wescoWs?wsdl"
        webservice_url = 'https://intra7-q.amb2.siemens.de/axis2/services/xpi4wescoWs/sendProtokoll'
    session = Session()
    session.verify = False
    urllib3.disable_warnings()
    try:
        if len(set_list) == 0:
            transport = Transport(session=session)
            ssl._create_default_https_context = ssl._create_unverified_context
            client = Client(webservice_wsdl, transport=transport)
            service = client.create_service('{http://com.siemens.make.webService.xpi4wesco/}xpi4wescoBinding',
                                            webservice_url)
            request_xml = {'xpi4wesco':{'header':{
                                            'serialnumber': fid, 'datetime': xpiTime(), 'testresult': 'P','source': 'W', 'protocolType': 'T'
                                        },
                                        # "testerprot":{
                                        #
                                        #      "tester_data": {"set":set_list}
                                        # },
                                        'comesaRefMap':{
                                            'comesa_ref': {'fio': 'J73', 'processType': 'O'}
                                            #'sit_ref': {'sit_fio': 'U_I98', 'lwo': 'O'},
                                            #'qme_ref': {'bea': 'O_AI2998'}},
                                         }
                                        # },
                                        },'ebb_appl': 'Xpi4wescoWsClient_dotNet4.0', 'ebb_appl_v': '2.2.2_WSDL1.10_python_upload'}

            response_xml = service.sit(**request_xml)
        else:
            transport = Transport(session=session)
            ssl._create_default_https_context = ssl._create_unverified_context
            client = Client(webservice_wsdl, transport=transport)
            service = client.create_service('{http://com.siemens.make.webService.xpi4wesco/}xpi4wescoBinding',
                                            webservice_url)
            request_xml = {'xpi4wesco': {'header': {
                'serialnumber': fid, 'datetime': xpiTime(), 'testresult': 'F', 'source': 'W', 'protocolType': 'T'
            },
                "testerprot":{

                     "tester_data": {"set":set_list}
                },
                'comesaRefMap': {
                    'comesa_ref': {'fio': 'J73', 'processType': 'O'}
                    # 'sit_ref': {'sit_fio': 'U_I98', 'lwo': 'O'},
                    # 'qme_ref': {'bea': 'O_AI2998'}},
                }
                # },
            }, 'ebb_appl': 'Xpi4wescoWsClient_dotNet4.0', 'ebb_appl_v': '2.2.2_WSDL1.10_python_upload'}

            response_xml = service.sit(**request_xml)
    except:
        print('Webservice connection was not succesfull')
        pass

if __name__ == '__main__':

    print('Connection of webservice was succesfull')


