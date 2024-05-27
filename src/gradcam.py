import os
import shutil
import subprocess
from gims import *

from PIL import Image
# Doku Library für die Gradcams: https://github.com/jacobgil/pytorch-grad-cam
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms
import numpy as np

from configparser import ConfigParser
config = ConfigParser()
config.read_file(open(r'config.txt'))
path_to_servingbucket = config.get('MinIOServer', 'path_local_pc')


def start_http_server(port = "9000", directory=path_to_servingbucket+"\serving"):
    """Startup a HTTP server for the specified directory on the specified port ("9000" recommended to make the setup work)
    """
    try:
        with open(os.devnull, 'w') as t:
            subprocess.Popen(["python","-m","http.server",port], stdout=t, stderr=t, cwd=directory)
        # print("HTTP server was sucessfully started...")
    except:
        # print("HTTP server could not be started")
        pass


def grad_cam_generation(model, gim, w, h):
    size = (w,h)
    
    image = Image.open(gim)
    
    new_image = Image.new("RGB", size, (255, 255, 255, 0))
    new_image.paste(image, (int((size[0] - image.size[0]) / 2), int((size[1] - image.size[1]) / 2)))

    convert_tensor = transforms.ToTensor()
    im = convert_tensor(new_image)
    im=im.unsqueeze(0)

    input_tensor = im
    target_layers = [model.features[-1]]    # can be different from model to model... please check it before computing
    # target_layers = ["features.28"]
    # for name, _ in model.named_modules(): print(name)     # layers can be printed with this statement. Select accordingly
    targets = None

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)     # if error raises, please check the according target_layer of the model. Target_layer must be the last activation layer.
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    rgb_img = np.array(new_image, np.float32)
    grad_cam = show_cam_on_image(rgb_img/255, grayscale_cam, use_rgb=True)

    return grad_cam


def emptytheservingfolder_and_savegradcam(gim, gradcam, path_to_folder=path_to_servingbucket+"\serving"):
    # delete the gradcamfiles in the folder
    for filename in os.listdir(path_to_folder):
        if filename.startswith("gradcam"):
            file_path = os.path.join(path_to_folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

    # save the last gradcam in the folder
    data = Image.fromarray(gradcam)
    data.save(path_to_folder + '\gradcamYESlatestgim.jpg')

    # save the last gim in the folder
    image = Image.open(gim)
    image.save(path_to_folder + '\gradcamNOlatestgim.jpg')


# path = r"C:\Users\Z004KVJF\Desktop\path_reclustering\iO_ÜBERPRÜFEN1\C-P7N56880_20220721203639_Volume_(X1)_0000000138_0000049990.jpg"
# image = Image.open(path)
# import predict
# model = predict.PredictionService().init_model(r'C:\Users\Z004KVJF\Desktop\git-clone\pseudo_xray\MinIO_Data\models\A5E3737764905\ImgClass2D\X1\v4.pt',
#                                        r'C:\Users\Z004KVJF\Desktop\git-clone\pseudo_xray\MinIO_Data\models\A5E3737764905\ImgClass2D\X1\v4.pth')

# image = Image.open(path)                                     

# gradcam = grad_cam_generation(model, path, image.size[0], image.size[1])
# # emptytheservingfolder_and_savegradcam(image, gradcam)

# import matplotlib.pyplot as plt
# plt.imshow(gradcam)
# plt.show()