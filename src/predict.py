# import model libraries
import os
import glob
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torch.nn.functional as f
from pathlib import Path
from sklearn.metrics import mean_squared_error
from monai.data import DataLoader, NiftiDataset
from monai.transforms import (
    AddChannel,
    Compose,
    Resize,
    ScaleIntensity,
    ToTensor,
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from configparser import ConfigParser
config = ConfigParser()
config.read_file(open(r'config.txt'))
image_size = config.getint('Clustering_model', 'image_size')
batch_size = config.getint('Clustering_model', 'batch_size')
mse_threshold = config.getfloat('Clustering_model', 'threshold')
latent_size = config.getint('Clustering_model', 'latent_size')    # latent vector dimension

class PredictionService:
    def __init__(self):
        """
        Funktionen:
            init_model--> Model and weights are loaded and model is set to eval mode
            pred_image_Class3D--> provides the necessary steps for the prediction of .rec/.nii files
            pred_image_Class3D--> provides the necessary steps for the prediction of jpg gims
            providePrediction--> provides the actual prediction service for the 2D and 3D classification
        ___________________________________________________________________________________________________________
         Input:
        """
    def init_model(self, model_path, weights_path) -> object:
        """ loads model in the beginning once and get it in the correct mode (eval) for prediction service
        :param  model_path: path to stored model
        :param weights_path: path to stored weights
        :return: loaded model
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # load model
        model = torch.load(model_path, map_location=torch.device("cpu"))
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        model.eval()

        return model

    def init_autoencoder(self, model_path, weights_path) -> object:
        """ same function as init_model for autoencoder
        :param  model_path: path to stored model
        :param weights_path: path to stored weights
        :return: loaded model
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Autoencoder loaden
        pretrained_model=torch.load(model_path, map_location=torch.device('cpu'))
        resnetAE = ResNetAE()
        resnetAE.load_state_dict(pretrained_model['model'])
        model = resnetAE.eval()

        return model

    def pred_image_Class3D(self, filename, model, type):
        #test_image = nb.load('niiCache/' + filename + '.nii')
        images = [('niiCache/' + Path(filename).stem + '.nii')]
        pred_transforms = Compose([ScaleIntensity(), AddChannel(), Resize((32, 192, 256)), ToTensor()])
        pred_ds = NiftiDataset(image_files=images, labels=[1], transform=pred_transforms)
        pred_loader = DataLoader(pred_ds, batch_size=1, num_workers=0, pin_memory=False)

        label, prob = self.providePrediction(model, pred_loader, type)

        if len(os.listdir('niiCache')) > 0:
            files = glob.glob('niiCache/*')
            for f in files:
                os.remove(f)

        return label, prob

    def pred_image_Class2D(self, jpgdata, model, type):
        """
        Pytorch predicts if Board picture is fail or not. Based on given model.
        :param jpgdata:    object of the jpg.file
        :param model:      loaded model with init_model
        :return:    prediction (pred) and prediction probability of the board
        """

        image = Image.open(jpgdata)

        # function to resize image -> 224x224 for vgg compatability
        size = (224, 224)

        # resize the image so the longest dimension matches our target size
        image.thumbnail(size, Image.ANTIALIAS)
        # Create a new square background image
        new_image = Image.new("RGB", size, "white")  # mode 'L' for black and white images, bg_color="white"
        # Paste the resized image into the center of the square background
        new_image.paste(image, (int((size[0] - image.size[0]) / 2), int((size[1] - image.size[1]) / 2)))
        resized_image = new_image
        # create the test loader
        loader = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        image = loader(resized_image).float()
        image = image.unsqueeze(0)
        image = image.cpu()

        test_loader = torch.utils.data.DataLoader(
            image,
            batch_size=1,
            num_workers=8,
            shuffle=False
        )
        label, prob = self.providePrediction(model, test_loader, type)
        return label, prob

    def providePrediction(self, model, loader, type):
        class_probs = []
        class_preds = []

        with torch.no_grad():
            for data in loader:
                if type == 'ImgClass2D':
                    images = data
                elif type == 'ImgClass3D':
                    images = data[0]
                output = model(images)
                
                class_probs_batch = [f.softmax(el, dim=0) for el in output]
                _, class_preds_batch = torch.max(output, 1)
                class_probs.append(class_probs_batch)
                class_preds.append(class_preds_batch)

        test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
        test_preds = torch.cat(class_preds)

        label: str
        if test_preds == 0:
            label = 'iO'
        elif test_preds == 1:
            label = 'niO'
        prob = (test_probs.numpy()[0][test_preds.numpy()]).item()
        return label, prob

    def pred_image_class2D_AE(self, jpgdata, AE_model):
        """
        Autoencoder predicts if Board picture is fail or not. Based on given model.
        :param jpgdata:    object of the jpg.file
        :param model:      loaded model with init_model
        :return:    prediction (pred) and prediction probability of the board

        Autoencoder model currently has 4 input-dimensions.
        """
        image = Image.open(jpgdata)
        size = (image_size, image_size)
        image = image.resize(size, Image.ANTIALIAS)
        loader = transforms.Compose([transforms.Resize((size)),transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
        image = loader(image).float()
        image = image.unsqueeze(0)          # 4th dimension needs to be added due to trained autoencoder architecture
        image = image.cpu()
        test_loader = torch.utils.data.DataLoader(image, batch_size=batch_size, num_workers=0, shuffle=False)
        label, prob = self.providePredictionAE(AE_model, test_loader)
        py_prob = prob.item()
        return label, py_prob
    
    def providePredictionAE(self, AE_model, loader):    

        latent_vector_list = []

        for data in loader:
            x = data

            latent_vector = AE_model.encode(x)
            reconstructed_image=AE_model.decode(latent_vector)

            latent_vector = latent_vector.cpu().tolist()
            image=x.cpu().tolist()
            reconstructed_image = reconstructed_image.cpu().tolist()

            frame = pd.DataFrame(data=latent_vector, columns=["dim1","dim2","dim3"])
            dim1 = frame["dim1"].values
            dim2 = frame["dim2"].values
            dim3 = frame["dim3"].values

            img = image
            rec_image = reconstructed_image

            img = np.array(img)
            rec_image = np.array(rec_image)

            image_mse = mean_squared_error(img.squeeze(), rec_image.squeeze())      # calculating MSE based on sklearn which requires <= 2 dimensions. Dropping the 2 irrelevant dimensions depth=1 with ".squeeze()"

            latent_vector_list.append([dim1, dim2, dim3, image_mse]) 

        if image_mse < mse_threshold:               # labeling occurs based on threshold which is located in config.txt
            label = "iO"
        else:
            label = "niO"

        return label, image_mse


class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1)):

        super(ResidualBlock, self).__init__()

        self.residual_block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                            kernel_size=kernel_size, stride=stride, padding=1),
            torch.nn.BatchNorm2d(in_channels),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
            torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, 
                            kernel_size=kernel_size, stride=stride, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x):
        return x + self.residual_block(x)

class ResNetEncoder(torch.nn.Module):
    def __init__(self,
                 n_ResidualBlock=8,
                 n_levels=4,
                 input_ch=1,
                 z_dim=128,
                 bUseMultiResSkips=True):

        super(ResNetEncoder, self).__init__()

        self.max_filters = 2 ** (n_levels+3)
        self.n_levels = n_levels
        self.z_dim = z_dim
        self.bUseMultiResSkips = bUseMultiResSkips

        self.conv_list = torch.nn.ModuleList()
        self.res_blk_list = torch.nn.ModuleList()
        self.multi_res_skip_list = torch.nn.ModuleList()

        self.input_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=input_ch, out_channels=8,
                            kernel_size=(3, 3), stride=(1, 1), padding=1),
            torch.nn.BatchNorm2d(8),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        for i in range(n_levels):
            n_filters_1 = 2 ** (i + 3)
            n_filters_2 = 2 ** (i + 4)
            ks = 2 ** (n_levels - i)

            self.res_blk_list.append(
                torch.nn.Sequential(*[ResidualBlock(n_filters_1, n_filters_1)
                                      for _ in range(n_ResidualBlock)])
            )

            self.conv_list.append(
                torch.nn.Sequential(
                    torch.nn.Conv2d(n_filters_1, n_filters_2,
                                    kernel_size=(2, 2), stride=(2, 2), padding=0),
                    torch.nn.BatchNorm2d(n_filters_2),
                    torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
                )
            )

            if bUseMultiResSkips:
                self.multi_res_skip_list.append(
                    torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=n_filters_1, out_channels=self.max_filters,
                                        kernel_size=(ks, ks), stride=(ks, ks), padding=0),
                        torch.nn.BatchNorm2d(self.max_filters),
                        torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    )
                )

        self.output_conv = torch.nn.Conv2d(in_channels=self.max_filters, out_channels=z_dim,
                                           kernel_size=(3, 3), stride=(1, 1), padding=1)
        #self.fc1 = torch.nn.Linear(self.z_dim * self.z_dim * self.z_dim, z_dim)
   

    def forward(self, x):

        x = self.input_conv(x)

        skips = []
        for i in range(self.n_levels):
            x = self.res_blk_list[i](x)
            if self.bUseMultiResSkips:
                skips.append(self.multi_res_skip_list[i](x))
            x = self.conv_list[i](x)

        if self.bUseMultiResSkips:
            x = sum([x] + skips)

        x = self.output_conv(x)
        #x = self.fc1(x.view(-1, self.z_dim * self.z_dim * self.z_dim))

        return x


class ResNetDecoder(torch.nn.Module):
    def __init__(self,
                 n_ResidualBlock=8,
                 n_levels=4,
                 z_dim=128,
                 output_channels=1,
                 bUseMultiResSkips=True):

        super(ResNetDecoder, self).__init__()

        self.max_filters = 2 ** (n_levels+3)
        self.n_levels = n_levels
        self.z_dim = z_dim
        self.bUseMultiResSkips = bUseMultiResSkips

        self.conv_list = torch.nn.ModuleList()
        self.res_blk_list = torch.nn.ModuleList()
        self.multi_res_skip_list = torch.nn.ModuleList()

        self.input_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=z_dim, out_channels=self.max_filters,
                            kernel_size=(3, 3), stride=(1, 1), padding=1),
            torch.nn.BatchNorm2d(self.max_filters),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        for i in range(n_levels):
            n_filters_0 = 2 ** (self.n_levels - i + 3)
            n_filters_1 = 2 ** (self.n_levels - i + 2)
            ks = 2 ** (i + 1)

            self.res_blk_list.append(
                torch.nn.Sequential(*[ResidualBlock(n_filters_1, n_filters_1)
                                      for _ in range(n_ResidualBlock)])
            )

            self.conv_list.append(
                torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(n_filters_0, n_filters_1,
                                             kernel_size=(2, 2), stride=(2, 2), padding=0),
                    torch.nn.BatchNorm2d(n_filters_1),
                    torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
                )
            )

            if bUseMultiResSkips:
                self.multi_res_skip_list.append(
                    torch.nn.Sequential(
                        torch.nn.ConvTranspose2d(in_channels=self.max_filters, out_channels=n_filters_1,
                                                 kernel_size=(ks, ks), stride=(ks, ks), padding=0),
                        torch.nn.BatchNorm2d(n_filters_1),
                        torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    )
                )

        self.output_conv = torch.nn.Conv2d(in_channels=n_filters_1, out_channels=output_channels,
                                           kernel_size=(3, 3), stride=(1, 1), padding=1)
        #self.fc2 = torch.nn.Linear(z_dim, self.z_dim * self.z_dim * self.z_dim)

    def forward(self, z):

        z = z_top = self.input_conv(z)

        for i in range(self.n_levels):
            z = self.conv_list[i](z)
            z = self.res_blk_list[i](z)
            if self.bUseMultiResSkips:
                z += self.multi_res_skip_list[i](z_top)

        z = self.output_conv(z)
        #z = self.fc2(z)

        return z
class ResNetAE(torch.nn.Module):
    def __init__(self,
                 input_shape=(224, 224, 1),
                 n_ResidualBlock=8,
                 n_levels=4,
                 z_dim=128,
                 bottleneck_dim=latent_size,
                 bUseMultiResSkips=True):
        super(ResNetAE, self).__init__()

        assert input_shape[0] == input_shape[1]
        image_channels = input_shape[2]
        self.z_dim = z_dim
        self.img_latent_dim = input_shape[0] // (2 ** n_levels)

        self.encoder = ResNetEncoder(n_ResidualBlock=n_ResidualBlock, n_levels=n_levels,
                                     input_ch=image_channels, z_dim=z_dim, bUseMultiResSkips=bUseMultiResSkips)
        self.decoder = ResNetDecoder(n_ResidualBlock=n_ResidualBlock, n_levels=n_levels,
                                     output_channels=image_channels, z_dim=z_dim, bUseMultiResSkips=bUseMultiResSkips)

        self.fc1 = torch.nn.Linear(self.z_dim * self.img_latent_dim * self.img_latent_dim, bottleneck_dim)
        self.fc2 = torch.nn.Linear(bottleneck_dim, self.z_dim * self.img_latent_dim * self.img_latent_dim)

    def encode(self, x):
        h = self.encoder(x)
        
        h = self.fc1(h.view(-1, self.z_dim * self.img_latent_dim * self.img_latent_dim))
        
        return h

    def decode(self, z):
        h = self.decoder(self.fc2(z).view(-1, self.z_dim, self.img_latent_dim, self.img_latent_dim))
        h=torch.sigmoid(h)
        return h 

    def forward(self, x):
        return self.decode(self.encode(x))



# example:
# model = init_model(os.path.join(evalDir, 'model.pt'))
# pred_image(Image.open(buf), model)
