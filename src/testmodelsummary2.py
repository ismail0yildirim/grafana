from torchvision import models


path_pt = r"C:\Users\Z004KVJF\Desktop\git-clone\pseudo_xray\MinIO_Data\models\A5E3737764905\ImgClass2D\X1\v4.pt"
path_pth = r"C:\Users\Z004KVJF\Desktop\git-clone\pseudo_xray\MinIO_Data\models\A5E3737764905\ImgClass2D\X1\v4.pth"

import predict
model = predict.PredictionService().init_model(path_pt,
                                                path_pth)


print(model)