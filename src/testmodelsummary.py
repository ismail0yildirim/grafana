from torchvision import models


path_pt = r"C:\Users\Z004KVJF\Desktop\model v5 x1\model_2023_02_07_Training_A5E37377649_X1_ImgClass2D_Model3_40_60_alle_niO.pt"
path_pth = r"C:\Users\Z004KVJF\Desktop\model v5 x1\model_2023_02_07_Training_A5E37377649_X1_ImgClass2D_Model3_40_60_alle_niO.pth"

import predict
model = predict.PredictionService().init_model(path_pt,
                                                path_pth)


print(model)