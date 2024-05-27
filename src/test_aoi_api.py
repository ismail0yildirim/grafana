import requests
import base64

url = "https://intra5.amb2.siemens.de:8033/pls/scout/scout.ws.get_defects_aoi?p_datum=05.09.2023"

payload={}

username = "ewa_pe_xray"
password = "T6Ps.#KA3Wqlk9E2|WZX"
connector = username + ':' + password
connector_bytes = connector.encode("ascii")
base64_bytes = base64.b64encode(connector_bytes)
connector64 = base64_bytes.decode("ascii")
code = 'Basic ' + connector64

headers = {

        'Authorization': code

    }

print(headers)
response = requests.request("GET", url, headers=headers, data=payload, verify=False)
print(response.text)