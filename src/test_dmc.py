from pylibdmtx.pylibdmtx import decode
from PIL import Image

image_path = r"C:\Users\Z004KVJF\Downloads\MicrosoftTeams-image.png"
image = Image.open(image_path)

decoded_data = decode(image)
for d in decoded_data:
    print(d.data.decode())
